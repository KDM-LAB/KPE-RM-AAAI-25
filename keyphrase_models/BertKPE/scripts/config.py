import os
import sys
import time
import logging
import argparse

logger = logging.getLogger()

def init_args_config():
    
    # bert pretrained model

    # logging file
    # args.log_file = os.path.join(args.save_folder, 'logging.txt')
    logger = logging.getLogger() 
    # logger.setLevel(logging.INFO) # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.WARNING) # logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    
    console = logging.StreamHandler() 
    console.setFormatter(fmt) 
    logger.addHandler(console) 

    # logger.info('COMMAND: %s' % ' '.join(sys.argv))
    # logger.info("preprocess_folder = {}".format(args.preprocess_folder))
    # logger.info("Pretrain Model Type = {}".format(args.pretrain_model_type))