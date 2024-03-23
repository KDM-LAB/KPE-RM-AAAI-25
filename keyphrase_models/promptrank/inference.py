import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer
from nltk import PorterStemmer
from sys import exit

pd.options.mode.chained_assignment = None

MAX_LEN = None
enable_pos = None
temp_en = None
temp_de = None
length_factor = None
position_factor = None
tokenizer = None

def init(setting_dict):
    '''
    Init template, max length and tokenizer.
    '''

    global MAX_LEN, temp_en, temp_de, tokenizer
    global enable_pos, length_factor, position_factor

    MAX_LEN = setting_dict["max_len"]
    temp_en = setting_dict["temp_en"]
    temp_de = setting_dict["temp_de"]
    enable_pos = setting_dict["enable_pos"]
    position_factor = setting_dict["position_factor"]
    length_factor = setting_dict["length_factor"]

    tokenizer = T5Tokenizer.from_pretrained("t5-" + setting_dict["model"], model_max_length=MAX_LEN)


# def get_PRF(num_c, num_e, num_s):
#     F1 = 0.0
#     P = float(num_c) / float(num_e) if num_e!=0 else 0.0
#     R = float(num_c) / float(num_s) if num_s!=0 else 0.0
#     if (P + R == 0.0):
#         F1 = 0
#     else:
#         F1 = 2 * P * R / (P + R)
#     return P, R, F1

# def print_PRF(P, R, F1, N, log):

#     log.logger.info("\nN=" + str(N))
#     log.logger.info("P=" + str(P))
#     log.logger.info("R=" + str(R))
#     log.logger.info("F1=" + str(F1) + "\n")
#     return 0

def keyphrases_selection(setting_dict, doc_list, model, dataloader, device, log=''):
    # print(f"{doc_list=}")
    init(setting_dict)
    # print("inference function called")
    # print(len(doc_list))

    model.eval()

    cos_similarity_list = {}
    candidate_list = []
    cos_score_list = []
    doc_id_list = []
    pos_list = []
    output_embedding_list = []
    last_layer_decoder_output_embedding_list = []
    
    num_c_5 = num_c_10 = num_c_15 = 0
    num_e_5 = num_e_10 = num_e_15 = 0
    num_s = 0
    
    template_len = tokenizer(temp_de, return_tensors="pt")["input_ids"].shape[1] - 3
    # print(f"{template_len=}")
    # print(f"{temp_de=}")
    # for id, [en_input_ids, en_input_mask, de_input_ids, dic] in enumerate(tqdm(dataloader,desc="Evaluating:")):
    for id, [en_input_ids, en_input_mask, de_input_ids, dic] in enumerate(dataloader):
        # print(f"{id=}")
        # print(f"{dic['de_input_len']=}")
        en_input_ids = en_input_ids.to(device)
        # print(f"{en_input_ids.shape=}")
        # print(f"{en_input_ids=}")
        en_input_mask = en_input_mask.to(device)
        de_input_ids = de_input_ids.to(device)
        # print(f"{de_input_ids.shape=}")
        # print(f"{de_input_ids=}")

        score = np.zeros(en_input_ids.shape[0])
        # print(f"{score.shape=}")
        
        with torch.no_grad():
            import time
            # start = time.perf_counter()
            op = model(input_ids=en_input_ids, attention_mask=en_input_mask, decoder_input_ids=de_input_ids)
            # print(f"{op.decoder_hidden_states[-2].shape=}")
            # print(f"{op.encoder_hidden_states[-2].shape=}")
            output = op[0]
            decoder_last_layer_output = op.decoder_hidden_states[-1]
            # end = time.perf_counter()
            # print("model running time:", end-start)
            # print(f"{output.shape=}")
            for i in range(template_len, de_input_ids.shape[1] - 3):
                logits = output[:, i, :]
                logits = logits.softmax(dim=1)
                logits = logits.cpu().numpy()
                # print(f"{logits.shape=}")

                for j in range(de_input_ids.shape[0]):
                    if i < dic["de_input_len"][j]:
                        score[j] = score[j] + np.log(logits[j, int(de_input_ids[j][i + 1])])
                    elif i == dic["de_input_len"][j]:
                        score[j] = score[j] / np.power(dic["de_input_len"][j] - template_len, length_factor)
            
            log_out = output[:, template_len:de_input_ids.shape[1] - 3, :]
            filtered_decoder_last_layer_output = decoder_last_layer_output[:, template_len:de_input_ids.shape[1] - 3, :]
            # print(f"{log_out.shape=}")
            log_out = torch.mean(log_out, axis=1, dtype=torch.float32)
            filtered_decoder_last_layer_output = torch.mean(filtered_decoder_last_layer_output, axis=1, dtype=torch.float32)
            log_out = log_out.numpy()
            filtered_decoder_last_layer_output = filtered_decoder_last_layer_output.numpy()
            # print(f"{log_out.shape=}")

            doc_id_list.extend(dic["idx"])
            candidate_list.extend(dic["candidate"])
            # print(f"{dic['candidate']=}")
            cos_score_list.extend(score)
            pos_list.extend(dic["pos"])
            output_embedding_list.extend([v for v in log_out])
            last_layer_decoder_output_embedding_list.extend([v for v in filtered_decoder_last_layer_output])

    cos_similarity_list["doc_id"] = doc_id_list
    cos_similarity_list["candidate"] = candidate_list
    # print(f"{candidate_list=}")
    cos_similarity_list["score"] = cos_score_list
    cos_similarity_list["pos"] = pos_list
    cos_similarity_list["output_embeddings"] = output_embedding_list
    cos_similarity_list["last_layer_decoder_output_embedding"] = last_layer_decoder_output_embedding_list
    cosine_similarity_rank = pd.DataFrame(cos_similarity_list)
    # print(f"{len(doc_list)=}")
    # print(f"{cosine_similarity_rank}")
    # print(f"{cosine_similarity_rank=}, {type(cosine_similarity_rank)=}")

    for i in range(len(doc_list)):
        doc_len = len(doc_list[i].split())
        
        doc_results = cosine_similarity_rank.loc[cosine_similarity_rank['doc_id']==i]
        # print(f"{doc_results=}, {type(doc_results)=}")
        #doc_results = cosine_similarity_rank.loc[cosine_similarity_rank]
        if enable_pos == True:
            doc_results["pos"] = doc_results["pos"] / doc_len + position_factor / (doc_len ** 3)
            doc_results["score"] = doc_results["pos"] * doc_results["score"]

        # print(f"{doc_results=}, {type(doc_results)=}")
        ranked_keyphrases = doc_results.sort_values(by='score', ascending=False)
        # print(f"{ranked_keyphrases=}, {type(ranked_keyphrases)=}")
        top_k = ranked_keyphrases.reset_index(drop=True)
        # print(f"{top_k=}, {type(top_k)=}")
        top_k_can = top_k.loc[:, ['candidate']].values.tolist()
        top_k_can_embeddings = top_k.loc[:, ['output_embeddings']].values.tolist()
        top_k_can_last_layer_decoder_embeddings = top_k.loc[:, ['last_layer_decoder_output_embedding']].values.tolist()
        # print(f"{top_k_can=}")
        # print(f"{len(top_k_can_embeddings)=}")
        # print(f"{len(top_k_can_embeddings[0])=}")
        # print(f"{top_k_can_embeddings[0]=}")

        candidates_set = set()
        candidates_dedup = []
        embeddings_dedup = []
        last_layer_decoder_embeddings_dedup = []
        for idx, temp in enumerate(top_k_can):
            temp = temp[0].lower()
            if temp in candidates_set:
                continue
            else:
                candidates_set.add(temp)
                candidates_dedup.append(temp)
                embeddings_dedup.append(top_k_can_embeddings[idx][0])
                last_layer_decoder_embeddings_dedup.append(top_k_can_last_layer_decoder_embeddings[idx][0])

        # print(f"{candidates_dedup=}")
        # print(f"{embeddings_dedup=}")
        j = 0
        # Matched = candidates_dedup[:15]
        Matched = candidates_dedup[:30] # Taking first 30 keyphrases
        Matched_embeddings = embeddings_dedup[:30] # Taking first 30 keyphrases embeddings
        Matched_last_layer_decoder_embeddings = last_layer_decoder_embeddings_dedup[:30]
        # print(len(Matched_embeddings), len(Matched), Matched_embeddings[0].shape)
        porter = PorterStemmer()
        # for id, temp in enumerate(candidates_dedup[0:15]):
        for id, temp in enumerate(candidates_dedup[0:30]):
            tokens = temp.split()
            tt = ' '.join(porter.stem(t) for t in tokens)
            
            if (j < 5):
                num_c_5 += 1
                num_c_10 += 1
                num_c_15 += 1

            elif (j < 10 and j >= 5):
                num_c_10 += 1
                num_c_15 += 1

            elif (j < 15 and j >= 10):
                num_c_15 += 1
            j += 1

        #log.logger.info("TOP-K {}: {} \n".format(i, Matched))
        ToReturn = Matched
        # print(Matched)
        
        if (len(top_k[0:5]) == 5):
            num_e_5 += 5
        else:
            num_e_5 += len(top_k[0:5])

        if (len(top_k[0:10]) == 10):
            num_e_10 += 10
        else:
            num_e_10 += len(top_k[0:10])

        if (len(top_k[0:15]) == 15):
            num_e_15 += 15
        else:
            num_e_15 += len(top_k[0:15])

        # print(f"{ToReturn=}")
        return ToReturn, Matched_embeddings, Matched_last_layer_decoder_embeddings

        





    
