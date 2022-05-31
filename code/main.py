import os
import shutil
from copy import deepcopy
import random
import json
import glob

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from nltk.tokenize import TweetTokenizer

from utils.functions import load_model, WordSplitTokenizer
from utils.args_helper import get_parser, print_opts
from utils.data_utils import load_sequence_classification_dataset, SequenceClassificationDataset
from utils.metrics import sentiment_metrics_fn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

###
# modelling functions
###
def get_lr(args, optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)

if __name__ == "__main__":
    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True
    
    # Parse args
    args = get_parser()
    # args = append_dataset_args(args)

    data_dir_path = f'datasets/{args["task"]}/{args["dataset"]}'

    # create directory
    output_dir = '{}/{}/{}/{}_{}_{}'.format(args["model_dir"],args["task"],args["dataset"],args['model_checkpoint'].replace('/','-'),args['seed'],args["num_sample"])
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir, exist_ok=True)
    # elif args['force']:
    #     print(f'overwriting model directory `{model_dir}`')
    # else:
    #     raise Exception(f'model directory `{model_dir}` already exists, use --force if you want to overwrite the folder')

    # Set random seed
    set_seed(args['seed'])  # Added here for reproductibility    
        
    # Prepare derived args
    if args["task"] == 'sentiment':
        strlabel2int = {'negative': 0, 'neutral': 1, 'positive': 2}
    elif args["task"] == 'lid':
        strlabel2int = {
            'indonesian': 0, 'balinese': 1, 'acehnese': 2, 'maduranese': 3, 'banjarese': 4, 'javanese': 5, 
            'buginese': 6, 'sundanese': 7, 'ngaju': 8, 'minangkabau': 9, 'toba_batak': 10, 'english': 11
        }
    else:
        raise ValueError(f'Unknown value `{args["task"]}` for key `--task`')
    
    args["num_labels"] = len(strlabel2int)

    # load model
    model, tokenizer, vocab_path, config_path = load_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    if args['device'] == "cuda":
        model = model.cuda()

    print("=========== TRAINING PHASE ===========")
    train_dataset, valid_dataset, test_dataset = load_sequence_classification_dataset(data_dir_path, strlabel2int, tokenizer, args["num_sample"], args['seed'])

    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    logging_dir = "logs"

    # Train
    training_args = TrainingArguments(
        output_dir=output_dir,          # output directory
        dataloader_num_workers=8,
        num_train_epochs=args["n_epochs"],              # total number of training epochs
        per_device_train_batch_size=args["train_batch_size"],  # batch size per device during training
        per_device_eval_batch_size=args["eval_batch_size"],   # batch size for evaluation
        learning_rate=args["lr"],              # number of warmup steps for learning rate scheduler
        weight_decay=args["gamma"],               # strength of weight decay
        gradient_accumulation_steps=args["grad_accum"], # Gradient accumulation
        logging_dir=logging_dir,            # directory for storing logs
        logging_strategy="epoch",
        evaluation_strategy='steps',
        save_strategy="steps",
        # logging_steps=logging_steps,
        eval_steps=150,
        save_steps=150,
        load_best_model_at_end = True,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=sentiment_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()
    valid_res = trainer.predict(valid_dataset)
    print(valid_res.metrics)

    # Evaluate
    print("=========== EVALUATION PHASE ===========")
    
    eval_metrics = {}
    for task_lang_path in glob.glob(f'datasets/{args["task"]}/*'):
        lang = task_lang_path.split('/')[-1]
        train_dataset, valid_dataset, test_dataset = load_sequence_classification_dataset(task_lang_path, strlabel2int, tokenizer)
        
        print(f'Run eval on `{lang}`')
        test_res = trainer.predict(test_dataset)
        eval_metrics[lang] = test_res.metrics
        
        print(f'Test results: {test_res.metrics}')

    log_output_path = output_dir + "/test_results.json"
    with open(log_output_path, "w+") as f:
        json.dump({"valid": valid_res.metrics, "test": eval_metrics}, f)
