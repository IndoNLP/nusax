import os
import shutil
from copy import deepcopy
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import AdamW, T5Tokenizer
from nltk.tokenize import TweetTokenizer
from modules.tokenization_indonlg import IndoNLGTokenizer
# from indobenchmark import IndoNLGTokenizer
from modules.tokenization_mbart52 import MBart52Tokenizer
from utils.functions import load_generation_model
from utils.args_helper import get_generation_parser, print_opts, append_generation_dataset_args, append_generation_model_args

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

###
# Training & Evaluation Function
###
    
# Evaluate function for validation and test
def evaluate(model, data_loader, forward_fn, metrics_fn, model_type, tokenizer, beam_size=1, max_seq_len=512, is_test=False, device='cpu'):
    model.eval()
    torch.set_grad_enabled(False)
    
    total_loss, total_correct, total_labels = 0, 0, 0

    list_hyp, list_label = [], []

    pbar = tqdm(iter(data_loader), leave=True, total=len(data_loader))
    for i, batch_data in enumerate(pbar):
        batch_seq = batch_data[-1]
        loss, batch_hyp, batch_label = forward_fn(model, batch_data, model_type=model_type, tokenizer=tokenizer, device=device, is_inference=is_test, 
                                                      is_test=is_test, skip_special_tokens=True, beam_size=beam_size, max_seq_len=max_seq_len)
        
        # Calculate evaluation metrics
        list_hyp += batch_hyp
        list_label += batch_label

        if not is_test:
            # Calculate total loss for validation
            test_loss = loss.item()
            total_loss = total_loss + test_loss

            # pbar.set_description("VALID {}".format(metrics_to_string(metrics)))
            pbar.set_description("VALID LOSS:{:.4f}".format(total_loss/(i+1)))
        else:
            pbar.set_description("TESTING... ")
            # pbar.set_description("TEST LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))
    
    metrics = metrics_fn(list_hyp, list_label)        
    if is_test:
        return total_loss/(i+1), metrics, list_hyp, list_label
    else:
        return total_loss/(i+1), metrics

# Training function and trainer
def train(model, train_loader, valid_loader, optimizer, forward_fn, metrics_fn, valid_criterion, tokenizer, n_epochs, evaluate_every=1, early_stop=3, step_size=1, gamma=0.5, max_norm=10, grad_accum=1, beam_size=1, max_seq_len=512, model_type='bart', model_dir="", exp_id=None, fp16=False, device='cpu'):
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_metric = -100
    count_stop = 0

    if fp16:
        scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(n_epochs):
        model.train()
        torch.set_grad_enabled(True)
        
        total_train_loss = 0
        list_hyp, list_label = [], []
        
        train_pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            if fp16:
                with torch.cuda.amp.autocast():
                    loss, batch_hyp, batch_label = forward_fn(model, batch_data, model_type=model_type, tokenizer=tokenizer, 
                                                device=device, skip_special_tokens=False, is_test=False)
                    
                # Scales the loss, and calls backward() to create scaled gradients
                scaler.scale(loss).backward()
                
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
                # Unscales gradients and calls or skips optimizer.step()
                scaler.step(optimizer)

                # Updates the scale for next iteration
                scaler.update()                    
            else:
                loss, batch_hyp, batch_label = forward_fn(model, batch_data, model_type=model_type, tokenizer=tokenizer, 
                                            device=device, skip_special_tokens=False, is_test=False)
            
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            # print(batch_hyp)
            # print(batch_label)

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label
            
            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                total_train_loss/(i+1), get_lr(args, optimizer)))
            
            if (i + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                                   
        metrics = metrics_fn(list_hyp, list_label)
        print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), metrics_to_string(metrics), get_lr(args, optimizer)))
        
        # Decay Learning Rate
        scheduler.step()

        # evaluate
        if ((epoch+1) % evaluate_every) == 0:
            val_loss, val_metrics = evaluate(model, valid_loader, forward_fn, metrics_fn, model_type, tokenizer, is_test=False, 
                                                 beam_size=beam_size, max_seq_len=max_seq_len, device=device)

            print("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1), val_loss, metrics_to_string(val_metrics)))            
            # Early stopping
            val_metric = val_metrics[valid_criterion]
            if best_val_metric < val_metric:
                best_val_metric = val_metric
                # save model
                if exp_id is not None:
                    torch.save(model.state_dict(), model_dir + "/best_model_" + str(exp_id) + ".th")
                else:
                    torch.save(model.state_dict(), model_dir + "/best_model.th")
                count_stop = 0
            else:
                count_stop += 1
                print("count stop:", count_stop)
                if count_stop == early_stop:
                    break

if __name__ == "__main__":
    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True
    
    # Parse args
    args = get_generation_parser()
    args = append_generation_dataset_args(args)
    args = append_generation_model_args(args)

    # create directory
    model_dir = '{}/mt/{}/{}'.format(args["model_dir"],args["dataset"],args['experiment_name'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    elif args['force']:
        print(f'overwriting model directory `{model_dir}`')
    else:
        raise Exception(f'model directory `{model_dir}` already exists, use --force if you want to overwrite the folder')

    # Set random seed
    set_seed(args['seed'])  # Added here for reproductibility    
    
    metrics_scores = []
    result_dfs = []
    # load model
    model, tokenizer, vocab_path, config_path = load_generation_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    # set a specific cuda device
    if "cuda" in args["device"]:
        torch.cuda.set_device(int(args["device"][4:]))
        args["device"] = "cuda"

    if args['device'] == "cuda":
        model = model.cuda()

    if type(tokenizer) == IndoNLGTokenizer:
        src_lid = tokenizer.special_tokens_to_ids[args['source_lang']]
        tgt_lid = tokenizer.special_tokens_to_ids[args['target_lang']]
        
        # Inject lang id as bos token in `model.generate()` function
        tokenizer.bos_token = args['target_lang']
        model.config.decoder_start_token_id = tgt_lid
    elif type(tokenizer) == MBart52Tokenizer:
        src_lid = tokenizer.lang_code_to_id[args['source_lang_bart']]
        tgt_lid = tokenizer.lang_code_to_id[args['target_lang_bart']]  
        model.config.decoder_start_token_id = tgt_lid      
    elif type(tokenizer) == T5Tokenizer: # mT5 baseline goes here because it doesn't need any language token
        src_lid = -1
        tgt_lid = -1
        tokenizer.bos_token = tokenizer.decode([model.config.decoder_start_token_id])
    else:
        ValueError(f'Unknown tokenizer type `{type(tokenizer)}`')
        
    print("=========== TRAINING PHASE ===========")

    train_dataset = args['dataset_class'](args['train_set_src_path'], args['train_set_tgt_path'], tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'], 
                                    speaker_1_id=args['speaker_1_id'], speaker_2_id=args['speaker_2_id'], separator_id=args['separator_id'],
                                    max_token_length=args['max_seq_len'], swap_source_target=args['swap_source_target'] if 'swap_source_target' in args else False)
    train_loader = args['dataloader_class'](dataset=train_dataset, model_type=args['model_type'], tokenizer=tokenizer, max_seq_len=args['max_seq_len'], batch_size=args['train_batch_size'], src_lid_token_id=src_lid, tgt_lid_token_id=tgt_lid, num_workers=8, shuffle=True)  

    valid_dataset = args['dataset_class'](args['valid_set_src_path'], args['valid_set_tgt_path'], tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'], 
                                    speaker_1_id=args['speaker_1_id'], speaker_2_id=args['speaker_2_id'], separator_id=args['separator_id'],
                                    max_token_length=args['max_seq_len'], swap_source_target=args['swap_source_target'] if 'swap_source_target' in args else False)
    valid_loader = args['dataloader_class'](dataset=valid_dataset, model_type=args['model_type'], tokenizer=tokenizer, max_seq_len=args['max_seq_len'], batch_size=args['valid_batch_size'], src_lid_token_id=src_lid, tgt_lid_token_id=tgt_lid, num_workers=8, shuffle=False)

    test_dataset = args['dataset_class'](args['test_set_src_path'], args['test_set_tgt_path'], tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'], 
                                    speaker_1_id=args['speaker_1_id'], speaker_2_id=args['speaker_2_id'], separator_id=args['separator_id'],
                                    max_token_length=args['max_seq_len'], swap_source_target=args['swap_source_target'] if 'swap_source_target' in args else False)
    test_loader = args['dataloader_class'](dataset=test_dataset, model_type=args['model_type'], tokenizer=tokenizer, max_seq_len=args['max_seq_len'], batch_size=args['test_batch_size'], src_lid_token_id=src_lid, tgt_lid_token_id=tgt_lid, num_workers=8, shuffle=False)

    # Train
    train(model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, forward_fn=args['forward_fn'], metrics_fn=args['metrics_fn'], valid_criterion=args['valid_criterion'], tokenizer=tokenizer, n_epochs=args['n_epochs'], evaluate_every=1, early_stop=args['early_stop'], grad_accum=args['grad_accumulate'], step_size=args['step_size'], gamma=args['gamma'], max_norm=args['max_norm'], model_type=args['model_type'], beam_size=args['beam_size'], max_seq_len=args['max_seq_len'], model_dir=model_dir, exp_id=0, fp16=args['fp16'], device=args['device'])

    # Save Meta
    if vocab_path:
        shutil.copyfile(vocab_path, f'{model_dir}/vocab.txt')
    if config_path:
        shutil.copyfile(config_path, f'{model_dir}/config.json')
        
    # Load best model
    model.load_state_dict(torch.load(model_dir + "/best_model_0.th"))

    # Evaluate
    print("=========== EVALUATION PHASE ===========")
    test_loss, test_metrics, test_hyp, test_label = evaluate(model, data_loader=test_loader, forward_fn=args['forward_fn'], metrics_fn=args['metrics_fn'], 
            model_type=args['model_type'], tokenizer=tokenizer, beam_size=args['beam_size'], max_seq_len=args['max_seq_len'], is_test=True, device=args['device'])

    metrics_scores.append(test_metrics)
    result_dfs.append(pd.DataFrame({
        'hyp': test_hyp, 
        'label': test_label
    }))
    
    result_df = pd.concat(result_dfs)
    metric_df = pd.DataFrame.from_records(metrics_scores)
    
    print('== Prediction Result ==')
    print(result_df.head())
    print()
    
    print('== Model Performance ==')
    print(metric_df.describe())
    
    result_df.to_csv(model_dir + "/prediction_result.csv")
    metric_df.describe().to_csv(model_dir + "/evaluation_result.csv")
