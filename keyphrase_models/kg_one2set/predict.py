# import argparse
import os
import time
import torch

import math
import keyphrase_models.kg_one2set.config
from keyphrase_models.kg_one2set.inference.evaluate import evaluate_greedy_generator
from keyphrase_models.kg_one2set.pykp.model import Seq2SeqModel
from keyphrase_models.kg_one2set.pykp.utils.io import build_interactive_predict_dataset
from keyphrase_models.kg_one2set.utils.data_loader import load_vocab, build_data_loader
from keyphrase_models.kg_one2set.utils.functions import common_process_opt, read_tokenized_src_file
from keyphrase_models.kg_one2set.utils.functions import time_since


def process_opt(opt):
    opt = common_process_opt(opt)

    if not os.path.exists(opt["pred_path"]):
        os.makedirs(opt["pred_path"])

    if torch.cuda.is_available():
        if not opt["gpuid"]:
            opt["gpuid"] = 0
        opt["device"] = torch.device("cuda:%d" % opt["gpuid"])
    else:
        opt["device"] = torch.device("cpu")
        opt["gpuid"] = -1
        # print("CUDA is not available, fall back to CPU.")

    return opt


def init_pretrained_model(opt):
    model = Seq2SeqModel(opt)
    model.load_state_dict(torch.load(opt["model"], map_location=torch.device('cpu')))
    model.to(opt["device"])
    model.eval()
    return model


def predict(test_data_loader, model, opt):
    if opt["fix_kp_num_len"]:
        from keyphrase_models.kg_one2set.inference.set_generator import SetGenerator
        generator = SetGenerator.from_opt(model, opt)
    else:
        from keyphrase_models.kg_one2set.inference.sequence_generator import SequenceGenerator
        generator = SequenceGenerator.from_opt(model, opt)
    return evaluate_greedy_generator(test_data_loader, generator, opt)


def main(opt, text):
    vocab = load_vocab(opt)
    src_file = opt["src_file"]
    tokenized_src = read_tokenized_src_file(src_file, text=text, remove_title_eos=opt["remove_title_eos"])
    # print(f"{tokenized_src = }")

    if opt["one2many"]:
        mode = 'one2many'
    else:
        mode = 'one2one'

    test_data = build_interactive_predict_dataset(tokenized_src, opt, mode=mode, include_original=True)

    torch.save(test_data, open(opt["exp_path"] + "/test_%s.pt" % mode, 'wb'))

    test_loader = build_data_loader(data=test_data, opt=opt, shuffle=False, load_train=False)
    # logging.info('#(test data size: #(batch)=%d' % (len(test_loader)))

    # init the pretrained model
    model = init_pretrained_model(opt)

    # Print out predict path
    # logging.info("Prediction path: %s" % opt["pred_path"])

    # predict the keyphrases of the src file and output it to opt.pred_path/predictions.txt
    start_time = time.time()
    return predict(test_loader, model, opt)
    training_time = time_since(start_time)
    # logging.info('Time for training: %.1f' % training_time)

def discard_subsets(phrases):  #consider only the superset and remove all subsets
    subsets_removed = []
    for i in range(len(phrases)):
        is_subset = False
        for j in range(i,len(phrases)):
            if i != j and set(phrases[i].split()).issubset(set(phrases[j].split())):
                is_subset = True
                break
        if not is_subset:
            subsets_removed.append(phrases[i])
    return subsets_removed


# if __name__ == '__main__':
def predict_one2set(text: str):
    # path = r"E:\Backend\reviewer-assignment\data\txts\0a485fd94b2cb554e281d0f8d7e9f71db4891ce0.txt"
    # with open(path, mode="r", encoding='utf-8') as fi:
    #     text = fi.read()
    # text = "a cyber medical center . <eos> this paper describes the design and implementation of a cyber medical center ( cmc ) using web technology . the intention is to overcome the inefficiency of the traditional filing system for patients ' medical records , which are considered to be time and space consuming . another objective is to enhance the interaction between the user the patient in this case and the medical center personnel the doctors and staff . this task is achieved by developing a cyber medical center interfaced with the internet to provide direct public access . the traditional filing system is replaced by a database system for maintaining the electronic medical records of all patients in the medical center . the doctors or staff can either view the medical records or update them through the intranet internet . this system has been successfully developed , implemented and tested on the intranet platform . it will be available in a university medical center for practical use . ( c ) <digit> elsevier ireland ltd ."

    text_len = len(text)
    chunks = math.ceil(text_len / 10000)        

    # load settings for training
    # parser = argparse.ArgumentParser(
    #     description='interactive_predict.py',
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # config.vocab_opts(parser)
    # config.model_opts(parser)
    # config.predict_opts(parser)
    # opt = parser.parse_args()

    opt = {"vocab_size":50000,
            "max_unk_words":1000,
            "word_vec_size":512,
            "enc_layers":6,
            "dec_layers":6,
            "dropout":0.1,
            "d_model":512,
            "n_head":8,
            "dim_ff":2048,
            "copy_attention":True,
            "max_kp_len":6,
            "max_kp_num":20,
            "fix_kp_num_len":True,
            "seperate_pre_ab":True,
            "src_file":"E:/Backend/reviewer-assignment/keyphrase_models/kg_one2set/data/testsets/kp20k/test_src.txt",
            "remove_title_eos":True,
            "vocab":"E:/Backend/reviewer-assignment/keyphrase_models/kg_one2set/data/kp20k_separated/Full",
            "model":"E:/Backend/reviewer-assignment/keyphrase_models/kg_one2set/output/train/Full_One2set_Copy_Seed27_Dropout0.1_LR0.0001_BS12_MaxLen6_MaxNum20_LossScalePre0.2_LossScaleAb0.1_Step2_SetLoss/best_model.pt",
            "pred_path":"E:/Backend/reviewer-assignment/keyphrase_models/kg_one2set/output/test/Full_One2set_Copy_Seed27_Dropout0.1_LR0.0001_BS12_MaxLen6_MaxNum20_LossScalePre0.2_LossScaleAb0.1_Step2_SetLoss/kp20k",
            "exp_path":"E:/Backend/reviewer-assignment/keyphrase_models/kg_one2set/output/test/Full_One2set_Copy_Seed27_Dropout0.1_LR0.0001_BS12_MaxLen6_MaxNum20_LossScalePre0.2_LossScaleAb0.1_Step2_SetLoss/kp20k",
            "gpuid":0,
            "seed":9527,
            "batch_size":20,
            "batch_workers":0,
            "beam_size":200,
            "n_best":None,
            "max_length":6,
            "one2many":True,
            "replace_unk":True}

    opt = process_opt(opt)
    # logging = config.init_logging(log_file=opt.exp_path + '/output.log', stdout=True)
    # logging.info('Parameters:')
    # [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    kp = []
    for i in range(chunks):
        t = text[i*10000:(i+1)*10000]

        try:
            keyphrases = main(opt.copy(), text=t)
            # print(i)
            keyphrase = keyphrases[0].replace("\n","")
            kp.extend(keyphrase.split(";"))
        except Exception as e:
            print(f"Error occured in predict_one2set: {e}")
            continue
        
    # print(discard_subsets(kp))
    return discard_subsets(kp)

    # print(main(opt.copy(), text=" ".join(kp)))

