import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import logging
import time
import torch

from data import data_process
from inference import keyphrases_selection
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration


def get_setting_dict():
    setting_dict = {}
    setting_dict["max_len"] = 512
    setting_dict["temp_en"] = "Book:"
    setting_dict["temp_de"] = "This book mainly talks about "
    setting_dict["model"] = "base"
    setting_dict["enable_filter"] = False
    setting_dict["enable_pos"] = True
    setting_dict["position_factor"] = 1.2e8
    setting_dict["length_factor"] = 0.6
    return setting_dict

# def parse_argument():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_dir",
#                         default=None,
#                         type=str,
#                         required=True,
#                         help="The input dataset.")
#     parser.add_argument("--dataset_name",
#                         default=None,
#                         type=str,
#                         required=True,
#                         help="The input dataset name.")
#     parser.add_argument("--batch_size",
#                         default=None,
#                         type=int,
#                         required=True,
#                         help="Batch size for testing.")
#     parser.add_argument("--log_dir",
#                         default=None,
#                         type=str,
#                         required=True,
#                         help="Path for Logging file")
#     args = parser.parse_args()
#     return args

def main(text):
    setting_dict = get_setting_dict()
    # args = parse_argument()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #log = Logger(args.log_dir + args.dataset_name + '.log')
    # start = time.time()
    #log.logger.info("Start Testing ...")

    # dataset, doc_list= data_process(setting_dict, args.dataset_dir, args.dataset_name)
    # dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size)
    dataset, doc_list= data_process(setting_dict,text ,"gold")
    dataloader = DataLoader(dataset, num_workers=4, batch_size=128)
    model = T5ForConditionalGeneration.from_pretrained("t5-"+ setting_dict["model"])
    model.to(device)

    return keyphrases_selection(setting_dict, doc_list,model, dataloader, device)

    # end = time.time()
    #log_setting(log, setting_dict)
    #log.logger.info("Processing time: {}".format(end-start))

# def log_setting(log, setting_dict):
#     for i, j in setting_dict.items():
#         log.logger.info(i + ": {}".format(j))

# class Logger(object):

#     def __init__(self, filename, level='info'):

#         level = logging.INFO if level == 'info' else logging.DEBUG
#         self.logger = logging.getLogger(filename)
#         self.logger.propagate = False
#         # # format_str = logging.Formatter(fmt)  
#         # if args.local_rank == 0 :
#         #     level = level
#         # else:
#         #     level = 'warning'
#         self.logger.setLevel(level)  # 

#         th = logging.FileHandler(filename,'w')
#         # formatter = logging.Formatter('%(asctime)s => %(name)s * %(levelname)s : %(message)s')
#         # th.setFormatter(formatter)

#         #self.logger.addHandler(sh)  # 
#         self.logger.addHandler(th)  # 
        
# if __name__ == "__main__":
#     torch.multiprocessing.set_sharing_strategy('file_system')
#     main("Machine learning classifiers for human-facing tasks such as comment toxicity and misinformation often score highly on metrics such as ROC AUC but are received poorly in practice. Why this gap? Today, metrics such as ROC AUC, precision, and recall are used to measure technical performance; however, human-computer interaction observes that evaluation of human-facing systems should account for people’s reactions to the system. In this paper, we introduce a transformation that more closely aligns machine learning classification metrics with the values and methods of user-facing performance measures. The disagreement deconvolution takes in any multi-annotator (e.g., crowdsourced) dataset, disentangles stable opinions from noise by estimating intra-annotator consistency, and compares each test set prediction to the individual stable opinions from each annotator. Applying the disagreement deconvolution to existing social computing datasets, we find that current metrics dramatically overstate the performance of many human-facing machine learning tasks: for example, performance on a comment toxicity task is corrected from .95 to .73 ROC AUC.")
    
