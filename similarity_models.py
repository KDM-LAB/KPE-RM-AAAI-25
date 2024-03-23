import spacy
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.metrics import jaccard_distance
from nltk.stem import WordNetLemmatizer
from sentbert_embed import sentbert_embeddings

lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_md')
glove_embed_dim = 300 # This model uses 300 dimensional representation space
# epsilon = 1e-2 # to prevent zero division when all of the tokens are in out of vocabulary (oov). This mostly happens when the abstract/title/text is chinese
epsilon = 0 # 1e-2 was producing noticible deflection. For ex, 2.43 was getting converted to 2.37. Also, I belive with all the checks in place, this should not come in use anyways.


# ------------------------------------------------------------------- #
# utility functions
def get_k_keyphrases(keyphrases, sim_args):
    return keyphrases[:abs(int(sim_args[4]))]

def lemmatize(keyphrases, sim_args):
    pos = sim_args[3]
    pos = ['n', 'v', 'a'] if pos=="l" else [pos.lstrip("l")]
    temp1 = []
    for i in keyphrases:
        temp2 = []
        for j in i.split(" "):
            for p in pos:
                j = lemmatizer.lemmatize(j, pos=p)
            temp2.append(j)
        temp1.append(" ".join(temp2))
    return temp1

def discard_subsets(keyphrases, sim_args): #consider only the superset and remove all subsets (both direction)
    '''
    ex: ['model prompt tuning', 'tuning model prompt', 'embeddings prompt tuning', 'model tuning prompt', 'tuning prompt embeddings'] -> ['model tuning prompt', 'tuning prompt embeddings']
    ex: ["computer", 'computer vision', "vision"] -> ['computer vision']
    '''
    supersets = []
    for i in range(len(keyphrases)):
        not_subset = True
        for j in range(i+1,len(keyphrases)):
            if set(keyphrases[i].split()).issubset(set(keyphrases[j].split())):
                not_subset = False
                break
        if not_subset:
            supersets.append(keyphrases[i])

    supersets = supersets[::-1]
    supersets_refined = []
    for i in range(len(supersets)):
        not_subset = True
        for j in range(i+1,len(supersets)):
            if set(supersets[i].split()).issubset(set(supersets[j].split())):
                not_subset = False
                break
        if not_subset:
            supersets_refined.append(supersets[i])
    return supersets_refined[::-1]

def cosine_similarity(vec_1: np.ndarray, vec_2: np.ndarray) -> float:
    return np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2) + epsilon)


# ------------------------------------------------------------------- #
# jaccard
def jaccard_partial_match(m_kp, p_kp):
    m_kp = set(word_tokenize(m_kp.lower()))
    p_kp = set(word_tokenize(p_kp.lower()))
    return 1 - jaccard_distance(m_kp, p_kp)

def jaccard(manuscript_kp, past_paper_kp, sim_args):
    if sim_args[1] == "cross":
        pos_factor = float(sim_args[6])
        similarity=0
        for idx_m, m_kp in enumerate(manuscript_kp):
            sim=[]
            for idx_p, p_kp in enumerate(past_paper_kp):
                pos = abs(idx_p - idx_m)
                sim.append(jaccard_partial_match(m_kp, p_kp) * 2/(1+np.exp(pos_factor*pos)))
            similarity+=max(sim)
        return (similarity/len(manuscript_kp)) * 5
    else: # when sim_args[1] == "avg"
        manuscript_kp = " ".join(manuscript_kp)
        past_paper_kp = " ".join(past_paper_kp)
        return jaccard_partial_match(manuscript_kp, past_paper_kp) * 5


# cos_glove
def cos_glove_avg_vector_representation(keyphrases: list[str]) -> np.ndarray:
    single_keywords = []
    multi_keywords = []
    for item in keyphrases:
        if " " in item:
            multi_keywords.append(item)
        else:
            single_keywords.append(item)

    single_keywords_vector_sum = np.zeros((glove_embed_dim,))
    multi_keywords_vector_sum = np.zeros((glove_embed_dim,))

    single_keywords_data = nlp(" ".join(single_keywords))
    single_keywords_data_count = 0
    for token in single_keywords_data:
        if not token.is_oov:
            single_keywords_vector_sum += token.vector
            single_keywords_data_count += 1

    multi_keywords_data_count = 0
    for keyphrase in multi_keywords:
        # if x = nlp("computer science"), then x.vector is the average gloVe(maybe) vector of "computer" and "science", which is the method we're using to deal with keyphrases
        if (vec := nlp(keyphrase)).vector_norm != 0.0:
            multi_keywords_vector_sum += vec.vector
            multi_keywords_data_count += 1

    if single_keywords_data_count + multi_keywords_data_count == 0:
        return None
    return (single_keywords_vector_sum + multi_keywords_vector_sum) / (single_keywords_data_count + multi_keywords_data_count)

def cos_glove_cross_vector_representation(keyphrases: list[str]) -> np.ndarray:
    keywords_vector = []
    for item in keyphrases:
        if " " in item:
            if (vec := nlp(item)).vector_norm != 0.0:
                keywords_vector.append(vec.vector)
            else:
                keywords_vector.append(None)
        else:
            # Can't compute all vectors for single word keyphrases cause it'll mess up the position of keyphrases needed for positional weights. Also, number of single words keyphrases seems to be less on average.
            if (vec := nlp(item)).vector_norm != 0.0:
                keywords_vector.append(vec.vector)
            else:
                keywords_vector.append(None)
    return keywords_vector

def cosine_glove(manuscript_kp, past_paper_kp, sim_args):
    if sim_args[1] == "cross":
        pos_factor = float(sim_args[6])
        vec_m = cos_glove_cross_vector_representation(manuscript_kp)
        vec_p = cos_glove_cross_vector_representation(past_paper_kp)
        total_similarity = 0
        total_count = 0
        for idx_m, v_m in enumerate(vec_m):
            if v_m is not None:
                max_sim_list = []
                for idx_p, v_p in enumerate(vec_p):
                    if v_p is not None:
                        pos = abs(idx_p - idx_m)
                        max_sim_list.append(cosine_similarity(v_m, v_p) * (2/(1+np.exp(pos_factor*pos))))
                if max_sim_list:
                    total_similarity += max(max_sim_list)
                    total_count += 1
        if total_count == 0:
            return None
        return (total_similarity / total_count) * 5
    else: # sim_args[1] == "avg"
        vec_m = cos_glove_avg_vector_representation(manuscript_kp)
        vec_p = cos_glove_avg_vector_representation(past_paper_kp)
        if (vec_m is None) or (vec_p is None):
            return None
        return cosine_similarity(vec_m, vec_p) * 5


# cos_sentbert
def cos_sentbert_vector_representation(keyphrases: list[str]) -> np.ndarray:
    keywords_vector = []
    for item in keyphrases:
        keywords_vector.append(sentbert_embeddings([item]))
    return keywords_vector

def cosine_sentbert(manuscript_kp, past_paper_kp, sim_args):
    manuscript_kp = [i for i in manuscript_kp if i] # getting rid of empty strings
    past_paper_kp = [i for i in past_paper_kp if i]
    if sim_args[1] == "cross":
        pos_factor = float(sim_args[6])
        vec_m = cos_sentbert_vector_representation(manuscript_kp)
        vec_p = cos_sentbert_vector_representation(past_paper_kp)
        total_similarity = 0
        total_count = 0
        for idx_m, v_m in enumerate(vec_m):
            max_sim_list = []
            for idx_p, v_p in enumerate(vec_p):
                pos = abs(idx_p - idx_m)
                max_sim_list.append(cosine_similarity(v_m, v_p) * (2/(1+np.exp(pos_factor*pos))))
            total_similarity += max(max_sim_list)
            total_count += 1
        return (total_similarity / total_count) * 5
    else: # sim_args[1] == "avg"
        vec_m = cos_sentbert_vector_representation(manuscript_kp)
        vec_p = cos_sentbert_vector_representation(past_paper_kp)
        return cosine_similarity(np.mean(vec_m, axis=0), np.mean(vec_p, axis=0)) * 5


# cos_selfembed
def cosine_selfembed(vec_m, vec_p, sim_args):
    if sim_args[1] == "cross":
        pos_factor = float(sim_args[6])
        total_similarity = 0
        for idx_m, v_m in enumerate(vec_m):
            max_sim_list = []
            for idx_p, v_p in enumerate(vec_p):
                pos = abs(idx_p - idx_m)
                max_sim_list.append((cosine_similarity(v_m, v_p) * (2/(1+np.exp(pos_factor*pos)))))
            total_similarity += max(max_sim_list)
        return (total_similarity / vec_m.shape[0]) * 5
    else: # sim_args[1] == "avg"
        vec_m = np.mean(vec_m, axis=0)
        vec_p = np.mean(vec_p, axis=0)
        return cosine_similarity(vec_m.flatten(), vec_p.flatten()) * 5


# ------------------------------------------------------------------- #
# main functions
def similarity_computation(x: str|np.ndarray, y: str|np.ndarray, func_list: list, sim_args: list) -> float:
    if len(x) == 0 or len(y) == 0:
        raise(Exception(f"A paper with zero keyphrases is found. This should not happen. Please check.\nmanuscript_kp: {x}\npast_paper_kp: {y}"))
    if sim_args[0] != "cos-selfembed":
        x = x.split(";")
        y = y.split(";")
    for fun in func_list[:-1]:
        x = fun(x, sim_args)
        y = fun(y, sim_args)
    return func_list[-1](x, y, sim_args)

def validate_similarity_args(sim_args):
    val_list = [False]*8
    try:
        val_list[0] = True if sim_args[0] in ["jaccard", "cos-glove", "cos-sentbert", "cos-selfembed"] else False
        val_list[1] = True if sim_args[1] in ["avg", "cross"] else False
        val_list[2] = True if sim_args[2] in ["nd", "d"] else False
        val_list[3] = True if sim_args[3] in ["nl", "l", "ln", "lv", "la"] else False
        val_list[4] = True if int(sim_args[4]) != 0 else False
        val_list[5] = True if int(sim_args[5]) == 20 else False
        val_list[6] = True if float(sim_args[6]) >= 0 else False
        val_list[7] = True if sim_args[7] in ["mean", "max"] else False

        if all(val_list):
            return True
        return False
    except:
        return False

def get_similarity_function_list(similarity_name: str) -> tuple[list, list]:
    """
    ARGUMENT
    similarity_name: similarity argument string with below format:
    {}_{}_{}_{}_{}_{}_{} with index values as:
    0 -> "jaccard", "cos-glove", "cos-sentbert" or "cos-selfembed"
    1 -> "avg" or "cross" for the type of similarity computation
    2 -> "nd" or "d" refering to either not to discard subsets or to discard subsets respectively
    3 -> "nl", "l", "ln", "lv" or "la" refering to not to lemmatize, to lemmatize with 'n, v, a', to lemmatize with 'n', to lemmatize with 'v', to lemmatize with 'a' respectively
    4 -> any positive integer value refering to the number of keyphrases to use after lemmatization and discard subset. any negative value for before lemmatization and discard subset.
    5 -> number of past papers to use for similarity calculation. Has to be 20 only as of now
    6 -> positional weight factor value for cross similarity calculation, typical value range [0.05, 0.5]. 0 to not have positional weight
    7 -> "mean" or "max" for the mode of similarity calculation

    An example can be:
    similarity_name = "jaccard_avg_nd_l_15_20_0.2_mean"
    similarity_name = "cos-sentbert_cross_d_nl_-5_20_0_max"

    RETURN
    A tuple of list of functions corresponding to similarity name and list of sim_args
    """
    sim_args = similarity_name.split("_")
    if not validate_similarity_args(sim_args):
        raise(Exception("Failed to generate similarity function list, please ensure to have correct input format"))
        
    similarity_name_dict = {"jaccard":jaccard, "cos-glove":cosine_glove, "cos-sentbert":cosine_sentbert, "cos-selfembed":cosine_selfembed}
    func_list = []
    if int(sim_args[4]) < 0:
        func_list.append(get_k_keyphrases)
    if sim_args[3] != "nl":
        func_list.append(lemmatize)
    if sim_args[2] == "d":
        func_list.append(discard_subsets)
    if int(sim_args[4]) > 0:
        func_list.append(get_k_keyphrases)
    
    func_list.append(similarity_name_dict[sim_args[0]])
    return func_list, sim_args