import torch

###
# Forward Function
###

# Forward function for sequence classification
def forward_sequence_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    # Unpack batch data
    if len(batch_data) == 3:
        (subword_batch, mask_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 4:
        (subword_batch, mask_batch, token_type_batch, label_batch) = batch_data
    
    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    label_batch = torch.LongTensor(label_batch)
            
    if device == "cuda":
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
        label_batch = label_batch.cuda()

    # Forward model
    outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
    loss, logits = outputs[:2]
    
    # generate prediction & label list
    list_hyp = []
    list_label = []
    hyp = torch.topk(logits, 1)[1]
    for j in range(len(hyp)):
        list_hyp.append(i2w[hyp[j].item()])
        list_label.append(i2w[label_batch[j][0].item()])
        
    return loss, list_hyp, list_label

###
# Inputs:
#    batch_data - one batch of data
#    tokenizer - the tokenizer use for decoding tokens into text
#    model_type - type of the model (only handle special case for indo_gpt2 as it is the only decoder only model)
#    is_inference - whether to run inference on decoder or else use teacher forcing for training
#    is_test - use beam search with the specified `beam_size` if is test else greedy search
#    skip_special_tokens - whether to skip decoding special tokens for producing hypothesis and label strings
#    beam_size - size of beam search
#    max_seq_len - maximum allowed length of the decoding sequence
#    device - device to run the forward function
#
# Outputs
#    loss - loss from the forward function, 0 when doing performing inference
#    list_hyp - list of hypothesis string
#    list_label - list of label string
###
def forward_generation(model, batch_data, tokenizer, model_type, is_inference=False, is_test=False, 
                           skip_special_tokens=True, beam_size=1, max_seq_len=512, device='cpu', length_penalty=1.0, 
                           top_p=1.0, top_k=50, repetition_penalty=1.0, do_sample=False,  **kwargs):
    # Unpack batch data
    if len(batch_data) == 6:
        (id, enc_batch, dec_batch, enc_mask_batch, dec_mask_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 7:
        (id, enc_batch, dec_batch, enc_mask_batch, dec_mask_batch, label_batch, token_type_batch) = batch_data
    
    # Prepare input & label
    enc_batch = torch.LongTensor(enc_batch) if enc_batch is not None else None
    dec_batch = torch.LongTensor(dec_batch)
    enc_mask_batch = torch.FloatTensor(enc_mask_batch) if enc_mask_batch is not None else None
    dec_mask_batch = torch.FloatTensor(dec_mask_batch) if dec_mask_batch is not None else None
    label_batch = torch.LongTensor(label_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
            
    if device == "cuda":
        enc_batch = enc_batch.cuda() if enc_batch is not None else None
        dec_batch = dec_batch.cuda()
        enc_mask_batch = enc_mask_batch.cuda() if enc_mask_batch is not None else None
        dec_mask_batch = dec_mask_batch.cuda() if dec_mask_batch is not None else None
        label_batch = label_batch.cuda()
        token_type_batch = token_type_batch.cuda()  if token_type_batch is not None else None

    # Forward model   
    if model_type == 'indo-gpt2':
        # GPT2 should go here
        if not is_inference:
            outputs = model(input_ids=dec_batch, attention_mask=dec_mask_batch, token_type_ids=token_type_batch, labels=label_batch)
            loss, logits = outputs[:2]
            hyps = logits.topk(1, dim=-1)[1]
        else:
            loss = 0
            hyps = model.generate(input_ids=enc_batch, attention_mask=enc_mask_batch, num_beams=beam_size if is_test else 1, 
                                    max_length=max_seq_len * 2, early_stopping=True, length_penalty=length_penalty, 
                                    repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k, do_sample=do_sample,
                                    use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            hyps = hyps[:, enc_batch.shape[1]:] # Remove prefix
    else: # model_type == 't5' or model_type == 'bart' or model_type == 'baseline-mt5' or model_type == 'baseline-mbart' 
        # BART and T5 should go here!!
        if not is_inference:
            outputs = model(input_ids=enc_batch, attention_mask=enc_mask_batch, decoder_input_ids=dec_batch, 
                    decoder_attention_mask=dec_mask_batch, labels=label_batch)
            loss, logits = outputs[:2]
            hyps = logits.topk(1, dim=-1)[1]
        else:
            loss = 0
            hyps = model.generate(input_ids=enc_batch, attention_mask=enc_mask_batch, num_beams=beam_size if is_test else 1, 
                                    max_length=max_seq_len, early_stopping=True, length_penalty=length_penalty, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    # generate prediction & label list
    list_hyp = []
    list_label = []
    for j in range(len(hyps)):
        hyp = hyps[j]
        label = label_batch[j,:].squeeze()
        if not is_inference:
            hyp = hyp.squeeze()[label != -100]
        list_hyp.append(tokenizer.decode(hyp, skip_special_tokens=skip_special_tokens))
        list_label.append(tokenizer.decode(label[label != -100], skip_special_tokens=skip_special_tokens))
        
    return loss, list_hyp, list_label