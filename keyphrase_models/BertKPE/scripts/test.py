import os
import sys
import time
import tqdm
import json
import torch

import logging
import argparse
import traceback
import numpy as np
from tqdm import tqdm

sys.path.append("..")

import keyphrase_models.BertKPE.scripts.config as config
import keyphrase_models.BertKPE.scripts.utils as utils
from keyphrase_models.BertKPE.scripts.utils import pred_arranger, pred_saver

from keyphrase_models.BertKPE.bertkpe import tokenizer_class, Idx2Tag, Tag2Idx, Decode_Candidate_Number
from keyphrase_models.BertKPE.bertkpe import dataloader, generator, evaluator
from keyphrase_models.BertKPE.scripts.model import KeyphraseSpanExtraction

torch.backends.cudnn.benchmark = False
from torch.utils.data.distributed import DistributedSampler

def select_decoder(name):
    if name in ["bert2rank", "bert2joint"]:
        return bert2rank_decoder

    raise RuntimeError("Invalid retriever class: %s" % name)

# Bert2Rank & Bert2Joint
def bert2rank_decoder(
    args,
    data_loader,
    dataset,
    model,
    test_input_refactor,
    pred_arranger,
    mode,
    stem_flag=False,
):
    test_time = utils.Timer()
    if args['dataset_class'] == "kp20k":
        stem_flag = True

    tot_examples = 0
    tot_predictions = []
    for step, batch in enumerate(data_loader):
        inputs, indices, lengths = test_input_refactor(batch, model.args['device'])
        try:
            logit_lists = model.test_bert2rank(inputs, lengths)
        except:
            logging.error(str(traceback.format_exc()))
            continue

        params = {
            "examples": dataset.examples,
            "logit_lists": logit_lists,
            "indices": indices,
            "return_num": Decode_Candidate_Number[args['dataset_class']],
            "stem_flag": stem_flag,
        }

        batch_predictions = generator.rank2phrase(**params)
        tot_predictions.extend(batch_predictions)

    candidate = pred_arranger(tot_predictions)
    return candidate

args = {"run_mode":"test",
        "dataset_class":"kp20k",
        "model_class":"bert2joint",
        "pretrain_model_type":"roberta-base",
        "preprocess_folder":r"E:\Backend\BertKPE\BertKPE\data\prepro_dataset",
        "pretrain_model_path":r"E:\Backend\BertKPE\BertKPE\data\pretrain_model",
        "save_path":"../results",
        "cached_features_dir":r"E:\Backend\BertKPE\BertKPE\data\cached_features",
        "no_cuda":True,
        "local_rank":-1,
        "data_workers":0,
        "seed":42,
        "max_token":510,
        "max_train_epochs":10,
        "max_train_steps":"0",
        "per_gpu_train_batch_size":8,
        "per_gpu_test_batch_size":32,
        "gradient_accumulation_steps":4,
        "tag_pooling":"min",
        "eval_checkpoint":r"E:\Backend\BertKPE\BertKPE\checkpoints\bert2joint\bert2joint.openkp.roberta.checkpoint",
        "max_phrase_words":5}

    # optim = parser.add_argument_group('Optimizer')
    # optim.add_argument("--learning_rate", default=5e-5, type=float,
    #                     help="The initial learning rate for Adam.")
    # optim.add_argument("--weight_decay", default=0.01, type=float,
    #                     help="Weight deay if we apply some.")
    # optim.add_argument("--warmup_proportion", default=0.1, type=float,
    #                    help="Linear warmup over warmup_ratio warm_step / t_total.")
    # optim.add_argument("--adam_epsilon", default=1e-8, type=float,
    #                     help="Epsilon for Adam optimizer.")
    # optim.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")

    # general = parser.add_argument_group('General')
    # general.add_argument('--use_viso', action='store_true', default=False,
    #                      help='Whether use tensorboadX to log loss.')
    # general.add_argument('--display_iter', type=int, default=200,
    #                      help='Log state after every <display_iter> batches.')
    # general.add_argument('--load_checkpoint', action='store_true', default=False,
    #                     help='Path to a checkpoint for generation .')
    # general.add_argument("--checkpoint_file", default='', type=str, 
    #                      help="Load checkpoint model continue training.")
    # general.add_argument('--save_checkpoint', action='store_true', default=False,
    #                     help='If true, Save model + optimizer state after each epoch.')
    # general.add_argument('--server_ip', type=str, default='', 
    #                      help="For distant debugging.")
    # general.add_argument('--server_port', type=str, default='', 
    #                      help="For distant debugging.")
    # general.add_argument('--fp16', action='store_true', default=False,
    #                      help="Whether to use 16-bit float precision instead of 32-bit")
    # general.add_argument('--fp16_opt_level', type=str, default='O1',
    #                     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    #                          "See details at https://nvidia.github.io/apex/amp.html")

args['cache_dir'] = os.path.join(args['pretrain_model_path'], args['pretrain_model_type'])
args['preprocess_folder'] = os.path.join(args['preprocess_folder'], args['dataset_class'])
    
config.init_args_config()

args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()
if args['local_rank'] == -1 or args['no_cuda']:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args['no_cuda'] else "cpu"
    )
    args['n_gpu'] = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args['local_rank'])
    device = torch.device("cuda", args['local_rank'])
    torch.distributed.init_process_group(backend='nccl')
    args['n_gpu'] = 1
args['device'] = device

tokenizer = tokenizer_class[args['pretrain_model_type']].from_pretrained(
    args['cache_dir']
)

batchify_features_for_train, batchify_features_for_test = dataloader.get_class(
    args['model_class']
)

args['test_batch_size'] = args['per_gpu_test_batch_size'] * max(1, args['n_gpu'])


def testing_gold(preprocessed_text: dict):

    eval_dataset = dataloader.build_dataset(
        **{"text": preprocessed_text, "args": args, "tokenizer": tokenizer, "mode": "eval"}
    )

    eval_sampler = torch.utils.data.sampler.SequentialSampler(eval_dataset)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args['test_batch_size'],
        sampler=eval_sampler,
        num_workers=args['data_workers'],
        collate_fn=batchify_features_for_test,
        pin_memory=args['cuda'],
    )

    try:
        model, checkpoint_epoch = KeyphraseSpanExtraction.load_checkpoint(
            args['eval_checkpoint'], args
        )
        model.set_device()
    except ValueError:
        print("Could't Load Pretrain Model %s" % args['eval_checkpoint'])

    if args['local_rank'] == 0:
        torch.distributed.barrier()

    if args['n_gpu'] > 1:
        model.parallelize()

    if args['local_rank'] != -1:
        model.distribute()

    candidate_decoder = select_decoder(args['model_class'])
    evaluate_script, main_metric_name = utils.select_eval_script(args['dataset_class'])
    _, test_input_refactor = utils.select_input_refactor(args['model_class'])

    # eval generator
    eval_candidate = candidate_decoder(
        args,
        eval_data_loader,
        eval_dataset,
        model,
        test_input_refactor,
        pred_arranger,
        "eval",
    )
    return eval_candidate[0]["KeyPhrases"]
 