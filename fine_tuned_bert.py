
# coding: utf-8

# ## Get started

# In[1]:


from __future__ import absolute_import, division, print_function
from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from sklearn import metrics
import glob
import logging
import os
os.environ['PYTHONHASHSEED'] = str(2019)

import random
random.seed(2019)

import json
from torch import nn
import numpy as np
np.random.seed(2019)

import torch
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)

torch.backends.cudnn.benchmark = False

torch.backends.cudnn.deterministic = True

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm_notebook, trange, tqdm

# from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,
#                                   XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
#                                   XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
#                                   RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

# from pytorch_transformers import AdamW, WarmupLinearSchedule
from transformers import *

from utils import (convert_examples_to_features,
                        output_modes, processors)

logger = logging.get_logger(__name__)

from torch.utils.tensorboard import SummaryWriter
import operator





dataset_name = ''
args = {
    'data_dir': 'fine_tuned_data/',
    'model_type':  'roberta',
    'model_name': 'roberta-base',
    'task_name': 'binary',
    'output_dir': 'fine_tuned_outputs/',
    'cache_dir': 'fine_tuned_cache/',
    'do_train': True,
    'do_eval': True,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 512,
    'output_mode': 'classification',
    'train_batch_size': 4,
    'eval_batch_size': 64,

    'gradient_accumulation_steps': 1,
    # 'num_train_epochs': 8,
    'weight_decay': 0,
    'learning_rate': 1e-4,
    'adam_epsilon': 1e-8,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    'logging_steps': int(180/4),
    'evaluate_during_training': False,
    'save_steps': int(180/4),
    'eval_all_checkpoints': True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,
    'notes': 'Using %s dataset'%(dataset_name),
    'cuda_num': 1
}

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('args.json', 'w') as f:
    json.dump(args, f)

if os.path.exists(args['output_dir']) and os.listdir(args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:
    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args['output_dir']))
    
    
task = args['task_name']

processor = processors[task]()
label_list = processor.get_labels()
num_labels = len(label_list)





def load_and_cache_examples(task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = args['output_mode']
    
    mode = 'dev' if evaluate else 'train'
    cached_features_file = os.path.join(args['data_dir'], f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")
    
    if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
               
    else:
        logger.info("Creating features from dataset file at %s", args['data_dir'])
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args['data_dir']) if evaluate else processor.get_train_examples(args['data_dir'])
        
        features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode,
            cls_token_at_end=bool(args['model_type'] in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
            pad_on_left=bool(args['model_type'] in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0)
        
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset





def train(train_dataset, model, tokenizer,dataset_name,saving_steps,learning_rate,no_epochs):
    tb_writer = SummaryWriter()
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])
    
    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * no_epochs
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total)
    
    if args['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])
        
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", no_epochs)
    logger.info("  Total train batch size  = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss, tr_losses = 0.0, 0.0, []
    model.zero_grad()
    train_iterator = trange(int(no_epochs), desc="Epoch")
    idx_iter = 0
    for _ in train_iterator:
        idx_iter += 1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        temp_loss =0.0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.cuda(args['cuda_num']) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            print("\r%f" % loss, end='')

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

            tr_loss += loss.item()
            temp_loss+=loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if saving_steps > 0 and global_step % saving_steps == 0:
                    # Log metrics
                    if args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        results,auc = evaluate(model, tokenizer)

                        for key, value in results[0].items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/saving_steps, global_step)
                    logging_loss = tr_loss

                if saving_steps > 0 and global_step % saving_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'],'%s-%s'%(dataset_name,str(global_step)))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
        tr_losses.append(temp_loss/len(train_dataloader))
        

    return global_step, tr_loss / global_step, tr_losses





from scipy.stats import pearsonr

def get_mismatched(labels, preds):
    mismatched = labels != preds
    examples = processor.get_dev_examples(args['data_dir'])
    wrong = [i for (i, v) in zip(examples, mismatched) if v]
    
    return wrong

def get_eval_report(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        'acc': (tp+tn)/(tp+tn+fp+fn)
    }, get_mismatched(labels, preds)

def compute_metrics(task_name, preds, probs, labels):
    assert len(preds) == len(labels)
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    result_dict = {
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        'acc': (tp+tn)/(tp+tn+fp+fn)
    }
                  
   
    probs_pos = np.array(probs)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs_pos, pos_label=1)
    auc = metrics.auc(fpr, tpr)


                                        
    return result_dict, auc
                                        
    

def evaluate(model, tokenizer, prefix=""):
    
    eval_output_dir = args['output_dir']

    results = {}
    EVAL_TASK = args['task_name']

    train_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, evaluate=False)
    eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, evaluate=True)

    # eval_dataset = train_dataset.concatenate(eval_dataset)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['eval_batch_size'])

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Eval!
    logger.info("***** Running evaluation based on training dataset{} *****".format(prefix))
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])   
                                        
    train_eval_loss = 0.0
    train_nb_eval_steps = 0
    train_preds = None
    train_probs = None
    train_out_label_ids = None 
    sm = nn.Softmax(dim=1)
    
                                        
    for batch in tqdm(train_dataloader, desc="Evaluating Training"):
        model.eval()
        batch = tuple(t.cuda(args['cuda_num']) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            train_eval_loss += tmp_eval_loss.mean().item()
        train_nb_eval_steps += 1
        if train_preds is None:
            train_preds = logits.detach().cpu().numpy()
            train_probs = sm(logits.detach().cpu())                            
            train_out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            temp_preds = logits.detach().cpu().numpy()                          
            train_preds = np.append(train_preds,temp_preds, axis=0)
            train_probs = np.append(train_probs,sm(logits.detach().cpu()), axis=0)
            train_out_label_ids = np.append(train_out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    train_eval_loss = train_eval_loss / train_nb_eval_steps
    if args['output_mode'] == "classification":
        train_preds = np.argmax(train_preds, axis=1)
    elif args['output_mode'] == "regression":
        train_preds = np.squeeze(train_preds)
    train_result_dict, train_auc = compute_metrics(EVAL_TASK, train_preds, train_probs,train_out_label_ids)

    
                                        
                                        
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
                                        
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    probs = None                                              

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.cuda(args['cuda_num']) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            probs = sm(logits.detach().cpu())                                       
            out_label_ids = inputs['labels'].detach().cpu().numpy()
            
        else:
            temp_preds = logits.detach().cpu().numpy()                                       
            preds = np.append(preds, temp_preds, axis=0)
            probs = np.append(probs,sm(logits.detach().cpu()), axis=0)                                       
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args['output_mode'] == "classification":
        preds = np.argmax(preds, axis=1)
    elif args['output_mode'] == "regression":
        preds = np.squeeze(preds)
    result_dict,auc = compute_metrics(EVAL_TASK, preds, probs,out_label_ids)
    

 
    return train_result_dict, train_auc ,result_dict, auc





def fine_tuned(dataset_name,pre_trained,saving_steps,learning_rate,no_epochs):

    
    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    }

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]

    config = config_class.from_pretrained(args['model_name'], num_labels=2, finetuning_task=args['task_name'])
    tokenizer = tokenizer_class.from_pretrained(args['model_name'])

    model = model_class.from_pretrained(pre_trained)

    model.cuda(args['cuda_num'])

    task = args['task_name']

    processor = processors[task]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args['do_train']:
        train_dataset = load_and_cache_examples(task, tokenizer)
        global_step, tr_loss, tr_losses = train(train_dataset, model, tokenizer,dataset_name,saving_steps,learning_rate,no_epochs)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        
    results = {}
    accs = {}
    train_accs = {}
    aucs = {}
    train_aucs = {}
    eval_confusion_matrices = {}
    # aucs = {}                               
    if args['do_eval']:
        checkpoints = [args['output_dir']]
        if args['eval_all_checkpoints']:

            # checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
            checkpoints = glob.glob('./fine_tuned_outputs/*')
            checkpoints.sort(key=os.path.getctime,reverse=True)
            logging.get_logger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        print(checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            print(checkpoint)
            model = model_class.from_pretrained(checkpoint)
            model.cuda(args['cuda_num'])

            ## train_result/result is a dictionary storing the confusion matrix and acc of the training set/testing set respectively
            ## train_auc/auc is the AUC score of the training set/testing set respectively

            train_result, train_auc, result, auc = evaluate(model, tokenizer, prefix=global_step)
    #         result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #         results.update(result)

            accs[checkpoint]=result['acc']
            train_accs[checkpoint]=train_result['acc']
            aucs[checkpoint]=auc
            train_aucs[checkpoint]=train_auc
            eval_confusion_matrices[checkpoint]=result


            # aucs[checkpoint]=auc


            ## return the checkpoint with highest performance in training set as well as the performance of all checkpoints
    return max(train_aucs.items(), key=operator.itemgetter(1))[0], accs, train_accs, aucs, train_aucs, tr_losses, eval_confusion_matrices

