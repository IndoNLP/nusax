from argparse import ArgumentParser
from transformers import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification, AlbertModel
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertForPreTraining, BertModel
from transformers import XLMConfig, XLMTokenizer, XLMForSequenceClassification, XLMForTokenClassification, XLMModel
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaModel

# generation model
from transformers import BartConfig, BartTokenizer, BartModel, BartForConditionalGeneration
from transformers import MBartTokenizer, MBartConfig, MBartForConditionalGeneration
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers import MT5ForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from modules.tokenization_indonlg import IndoNLGTokenizer
# from indobenchmark import IndoNLGTokenizer
from modules.tokenization_mbart52 import MBart52Tokenizer

import json
import numpy as np
import torch

class WordSplitTokenizer():
    def tokenize(self, string):
        return string.split()
    
class SimpleTokenizer():
    def __init__(self, vocab, word_tokenizer, lower=True):
        self.vocab = vocab
        self.lower = lower
        idx = len(self.vocab.keys())
        self.vocab["<bos>"] = idx+0
        self.vocab["<|endoftext|>"] = idx+1
        self.vocab["<speaker1>"] = idx+2
        self.vocab["<speaker2>"] = idx+3
        self.vocab["<pad>"] = idx+4
        self.vocab["<cls>"] = idx+5
        self.vocab["<sep>"] = idx+6

        self.inverted_vocab = {int(v):k for k,v in self.vocab.items()}
        assert len(self.vocab.keys()) == len(self.inverted_vocab.keys())
        
        # Define word tokenizer
        self.tokenizer = word_tokenizer
        
        # Add special token attribute
        self.cls_token_id = self.vocab["<cls>"]
        self.sep_token_id = self.vocab["<sep>"]   

    def __len__(self):
        return len(self.vocab.keys())+1

    def convert_tokens_to_ids(self,tokens):
        if(type(tokens)==list):
            return [self.vocab[tok] for tok in tokens]
        else:
            return self.vocab[tokens]

    def encode(self,text,text_pair=None,add_special_tokens=False):
        if self.lower:
            text = text.lower()
            text_pair = text_pair.lower() if text_pair else None

        if not add_special_tokens:
            tokens = [self.vocab[tok] for tok in self.tokenizer.tokenize(text)]
            if text_pair:
                tokens += [self.vocab[tok] for tok in self.tokenizer.tokenize(text_pair)]
        else:
            tokens = [self.vocab["<cls>"]] + [self.vocab[tok] for tok in self.tokenizer.tokenize(text)] + [self.vocab["<sep>"]]
            if text_pair:
                tokens += [self.vocab[tok] for tok in self.tokenizer.tokenize(text_pair)] + [self.vocab["<sep>"]]
        return tokens     
    
    def encode_plus(self,text,text_pair=None,add_special_tokens=False, return_token_type_ids=False):
        if self.lower:
            text = text.lower()
            text_pair = text_pair.lower() if text_pair else None
        
        if not add_special_tokens:
            tokens = [self.vocab[tok] for tok in self.tokenizer.tokenize(text)]
            if text_pair:
                tokens_pair = [self.vocab[tok] for tok in self.tokenizer.tokenize(text_pair)]
                token_type_ids = len(tokens) * [0] + len(tokens_pair) * [1]
                tokens += tokens_pair
        else:
            tokens = [self.vocab["<cls>"]] + [self.vocab[tok] for tok in self.tokenizer.tokenize(text)] + [self.vocab["<sep>"]]
            if text_pair:
                tokens_pair = [self.vocab[tok] for tok in self.tokenizer.tokenize(text_pair)] + [self.vocab["<sep>"]]
                token_type_ids = (len(tokens) * [0]) + (len(tokens_pair) * [1])
                tokens += tokens_pair
        
        encoded_inputs = {}
        encoded_inputs['input_ids'] = tokens
        if return_token_type_ids:
            encoded_inputs['token_type_ids'] = token_type_ids
        return encoded_inputs

    def decode(self,index,skip_special_tokens=True):
        return " ".join([self.inverted_vocab[ind] for ind in index])

    def save_pretrained(self, save_dir): 
        with open(save_dir+'/vocab.json', 'w') as fp:
            json.dump(self.vocab, fp, indent=4)

def gen_embeddings(vocab_list, emb_path, emb_dim=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = None
    count, pre_trained = 0, 0
    vocab_map = {}
    for i in range(len(vocab_list)):
        vocab_map[vocab_list[i]] = i

    found_word_map = {}

    print('Loading embedding file: %s' % emb_path)
    for line in open(emb_path).readlines():
        sp = line.split()
        count += 1
        if count == 1 and emb_dim is None:
            # header <num_vocab, emb_dim>
            emb_dim = int(sp[1])
            embeddings = np.random.rand(len(vocab_list), emb_dim)
            print('Embeddings: %d x %d' % (len(vocab_list), emb_dim))
        else:
            if count == 1:
                embeddings = np.random.rand(len(vocab_list), emb_dim)
                print('Embeddings: %d x %d' % (len(vocab_list), emb_dim))
                continue

            if(len(sp) == emb_dim + 1): 
                if sp[0] in vocab_map:
                    found_word_map[sp[0]] = True
                    embeddings[vocab_map[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print("Error:", sp[0], len(sp))
    pre_trained = len(found_word_map)
    print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / len(vocab_list)))
    return embeddings

def load_vocab(path):
    vocab_list = []
    with open(path, "r") as f:
        for word in f:
            vocab_list.append(word.replace('\n',''))

    vocab_map = {}
    for i in range(len(vocab_list)):
        vocab_map[vocab_list[i]] = i
        
    return vocab_list, vocab_map

def get_model_class(model_type, task):
    if 'babert-lite' in model_type:
        base_cls = AlbertModel
        if 'sequence_classification' == task:
            pred_cls = AlbertForSequenceClassification
        elif 'token_classification' == task:
            pred_cls = AlbertForWordClassification
        elif 'multi_label_classification' == task:
            pred_cls = AlbertForMultiLabelClassification     
    elif 'xlm-mlm' in model_type:
        base_cls = XLMModel
        if 'sequence_classification' == task:
            pred_cls = XLMForSequenceClassification
        elif 'token_classification' == task:
            pred_cls = XLMForWordClassification
        elif 'multi_label_classification' == task:
            pred_cls = XLMForMultiLabelClassification
    elif 'xlm-roberta' in model_type:
        base_cls = XLMRobertaModel
        if 'sequence_classification' == task:
            pred_cls = XLMRobertaForSequenceClassification
        elif 'token_classification' == task:
            pred_cls = XLMRobertaForWordClassification
        elif 'multi_label_classification' == task:
            pred_cls = XLMRobertaForMultiLabelClassification
    else: # 'babert', 'bert-base-multilingual', 'word2vec', 'fasttext', 'scratch'
        base_cls = BertModel
        if 'sequence_classification' == task:
            pred_cls = BertForSequenceClassification
        elif 'token_classification' == task:
            pred_cls = BertForWordClassification
        elif 'multi_label_classification' == task:
            pred_cls = BertForMultiLabelClassification
    return base_cls, pred_cls

def load_model(args):
    if 'bert-base-multilingual' in args['model_checkpoint']:
        # bert-base-multilingual-uncased or bert-base-multilingual-cased
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = BertTokenizer.from_pretrained(args['model_checkpoint'])
        config = BertConfig.from_pretrained(args['model_checkpoint'])
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']
        
        # Instantiate model
        model = BertForSequenceClassification.from_pretrained(args['model_checkpoint'], config=config)
    elif 'xlm-mlm' in args['model_checkpoint']:
        # xlm-mlm-100-1280
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = XLMTokenizer.from_pretrained(args['model_checkpoint'])            
        config = XLMConfig.from_pretrained(args['model_checkpoint'])
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        model = XLMForSequenceClassification.from_pretrained(args['model_checkpoint'], config=config)
    elif 'xlm-roberta' in args['model_checkpoint']:
        # xlm-roberta-base or xlm-roberta-large
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = XLMRobertaTokenizer.from_pretrained(args['model_checkpoint'])                                                        
        config = XLMRobertaConfig.from_pretrained(args['model_checkpoint'])
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']
        
        # Instantiate model
        model = XLMRobertaForSequenceClassification.from_pretrained(args['model_checkpoint'], config=config)
    elif 'fasttext' in args['model_checkpoint']:
        # Prepare config & tokenizer
        vocab_path = args['vocab_path']
        config_path = None
        
        word_tokenizer = args['word_tokenizer_class']()
        emb_path = args['embedding_path'][args['model_checkpoint']]

        _, vocab_map = load_vocab(vocab_path)
        tokenizer = SimpleTokenizer(vocab_map, word_tokenizer, lower=args["lower"])
        vocab_list = list(tokenizer.vocab.keys())

        config = BertConfig.from_pretrained('bert-base-uncased') 
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']
        config.num_hidden_layers = args["num_layers"]

        embeddings = gen_embeddings(vocab_list, emb_path, emb_dim=300)
        config.hidden_size = 300
        config.num_attention_heads = 10
        config.vocab_size = len(embeddings)

        # Instantiate model
        model = BertForSequenceClassification(config)
        model.bert.embeddings.word_embeddings.weight.data.copy_(torch.FloatTensor(embeddings))
            
    elif 'scratch' in args['model_checkpoint']: 
        vocab_path, config_path = None, None
        
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        config = BertConfig.from_pretrained("bert-base-uncased")
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']
        config.num_hidden_layers = args["num_layers"]
        config.hidden_size = 300
        config.num_attention_heads = 10
        
        model = BertForSequenceClassification(config=config)
    elif 'indobenchmark' in args['model_checkpoint']:
        # indobenchmark models
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = BertTokenizer.from_pretrained(args['model_checkpoint'])
        config = BertConfig.from_pretrained(args['model_checkpoint'])
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']
        
        # Instantiate model
        model_class = None
        model_class = AlbertForSequenceClassification if 'lite' in args['model_checkpoint'] else BertForSequenceClassification
        model = model_class.from_pretrained(args['model_checkpoint'], config=config) 
    elif 'indolem' in args['model_checkpoint']:
        # indobenchmark models
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = BertTokenizer.from_pretrained(args['model_checkpoint'])
        config = BertConfig.from_pretrained(args['model_checkpoint'])
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']
        
        # Instantiate model
        model = BertForSequenceClassification.from_pretrained(args['model_checkpoint'], config=config) 
    return model, tokenizer, vocab_path, config_path

def load_generation_model(args, resize_embedding=True):
    # IndoNLG Tokenizer vocabulary
    vocab_size = 40011
    special_tokens_to_ids = {
        "[javanese]": 40000, 
        "[sundanese]": 40001, 
        "[indonesian]": 40002,
        "<mask>": 40003,
        "[english]": 40004,
        "[acehnese]": 40005,
        "[balinese]": 40006,
        "[banjarese]": 40007,
        "[bugis]": 40008,
        "[madurese]": 40009,
        "[minang]": 40010,
        "[ngaju]": 40011
    }
    special_ids_to_tokens = {v: k for k, v in special_tokens_to_ids.items()}

    # Store Language token ID
    javanese_token, javanese_token_id = '[javanese]', 40000
    sundanese_token, sundanese_token_id = '[sundanese]', 40001
    indonesian_token, indonesian_token_id = '[indonesian]', 40002
    english_token, english_token_id = '[english]', 40004
    acehnese_token, acehnese_token_id = '[acehnese]', 40005
    balinese_token, balinese_token_id = '[balinese]', 40006
    banjarese_token, banjarese_token_id = '[banjarese]', 40007
    bugis_token, bugis_token_id = '[bugis]', 40008
    madurese_token, madurese_token_id = '[madurese]', 40009
    minang_token, minang_token_id = '[minang]', 40010
    ngaju_token, ngaju_token_id = '[ngaju]', 40011

    ##############################################

    if 'transformer' in args['model_type']:
        # baseline transformer models
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = IndoNLGTokenizer(vocab_file=args['vocab_path'])
        if 'local' in args['model_type']:
            config = BartConfig.from_pretrained('pretrained_models/facebook/bart-base') # Use Bart config, because there is no MBart-base
        else:
            config = BartConfig.from_pretrained('facebook/bart-base') # Use Bart config, because there is no MBart-base
        config.vocab_size = vocab_size
        tokenizer.special_tokens_to_ids = special_tokens_to_ids
        tokenizer.special_ids_to_tokens = {v: k for k, v in special_tokens_to_ids.items()}
        
        # Instantiate model
        model = MBartForConditionalGeneration(config=config)
        if args['model_checkpoint']:
            bart = BartModel(config=config)
        
    elif 'baseline' in args['model_type']:
        vocab_path, config_path = None, None
        if 'mbart' in args['model_type']:
            # mbart models
            # Prepare config & tokenizer
            tokenizer = MBart52Tokenizer.from_pretrained(args['model_checkpoint'], src_lang=args['source_lang_bart'], tgt_lang=args['target_lang_bart'])
            model = MBartForConditionalGeneration.from_pretrained(args['model_checkpoint'])
            
            # Added new language token For MT
            if resize_embedding:
                # model.resize_token_embeddings(model.config.vocab_size + 4) # For su_SU, jv_JV, <speaker_1>, <speaker_2>
                model.resize_token_embeddings(model.config.vocab_size + 15) # Adding new languages
            
            # Freeze Layer
            if args['freeze_encoder']:
                for parameter in model.model.encoder.parameters():
                    parameter.requires_grad = False
            if args['freeze_decoder']:
                for parameter in model.model.decoder.parameters():
                    parameter.requires_grad = False
            
        elif 'mt5' in args['model_type']:
            # mt5 models
            # Prepare config & tokenizer
            tokenizer = T5Tokenizer.from_pretrained(args['model_checkpoint'])
            model = MT5ForConditionalGeneration.from_pretrained(args['model_checkpoint'])   
            
            if 'small' not in args['model_type']:
                # Freeze Layer
                if args['freeze_encoder']:
                    for parameter in model.encoder.parameters():
                        parameter.requires_grad = False
                if args['freeze_decoder']:
                    for parameter in model.decoder.parameters():
                        parameter.requires_grad = False
    elif 'indo-bart' in args['model_type']:
        # bart models
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        # tokenizer = IndoNLGTokenizer(vocab_file=args['vocab_path'])
        if 'local' in args['model_type']:
            tokenizer = IndoNLGTokenizer.from_pretrained('pretrained_models/indobenchmark/indobart-v2')
            config = BartConfig.from_pretrained('pretrained_models/facebook/bart-base') # Use Bart config, because there is no MBart-base
        else:
            tokenizer = IndoNLGTokenizer.from_pretrained('indobenchmark/indobart-v2')
            config = BartConfig.from_pretrained('facebook/bart-base') # Use Bart config, because there is no MBart-base
        config.vocab_size = vocab_size
        tokenizer.special_token_ids = [
            tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id, 
            tokenizer.unk_token_id, tokenizer.pad_token_id, tokenizer.mask_token_id, 
            english_token_id, javanese_token_id, sundanese_token_id, indonesian_token_id,
            acehnese_token_id, balinese_token_id, banjarese_token_id,
            bugis_token_id, madurese_token_id, minang_token_id,
            ngaju_token_id
        ]
        tokenizer.special_tokens_to_ids = special_tokens_to_ids
        tokenizer.special_ids_to_tokens = {v: k for k, v in special_tokens_to_ids.items()}
        
        # Instantiate model
        if 'local' in args['model_type']:
            model = MBartForConditionalGeneration.from_pretrained('pretrained_models/indobenchmark/indobart-v2')
        else:
            model = MBartForConditionalGeneration.from_pretrained('indobenchmark/indobart-v2')
        # model = MBartForConditionalGeneration(config=config)
        # if args['model_checkpoint']:
        #     bart = BartModel(config=config)
        #     bart.load_state_dict(torch.load(args['model_checkpoint'])['model'], strict=False)
        #     bart.shared.weight = bart.encoder.embed_tokens.weight
        #     model.model = bart

    elif 'indo-gpt2' in args['model_type']:
        # gpt2 models
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        
        if 'local' in args['model_type']:
            tokenizer = IndoNLGTokenizer.from_pretrained('pretrained_models/indobenchmark/indogpt')
            config = GPT2Config.from_pretrained('pretrained_models/gpt2')
        else:
            tokenizer = IndoNLGTokenizer.from_pretrained('indobenchmark/indogpt')
        # tokenizer = IndoNLGTokenizer(vocab_file=args['vocab_path'])
            config = GPT2Config.from_pretrained('gpt2')
        config.vocab_size = vocab_size
        tokenizer.special_token_ids = [
            tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id, 
            tokenizer.unk_token_id, tokenizer.pad_token_id, tokenizer.mask_token_id, 
            english_token_id, javanese_token_id, sundanese_token_id, indonesian_token_id,
            acehnese_token_id, balinese_token_id, banjarese_token_id,
            bugis_token_id, madurese_token_id, minang_token_id,
            ngaju_token_id
        ]
        tokenizer.special_tokens_to_ids = special_tokens_to_ids
        tokenizer.special_ids_to_tokens = {v: k for k, v in special_tokens_to_ids.items()}
        
        # Instantiate model
        if 'local' in args['model_type']:
            model = GPT2LMHeadModel.from_pretrained('pretrained_models/indobenchmark/indogpt')
        else:
            model = GPT2LMHeadModel.from_pretrained('indobenchmark/indogpt')
        old_vocab_size = model.transformer.wte.weight.size()[0]
        embedding_dim = model.transformer.wte.weight.size()[1]
        # print(model.transformer.wte.weight.size())
        model.transformer.wte.weight = torch.nn.Parameter(torch.cat((model.transformer.wte.weight, torch.randn(vocab_size-old_vocab_size, embedding_dim))))
        # if args['model_checkpoint']:
        #     state_dict = torch.load(args['model_checkpoint'])
        #     model.load_state_dict(state_dict)
        
    return model, tokenizer, vocab_path, config_path