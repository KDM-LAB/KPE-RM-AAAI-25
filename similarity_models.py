import spacy
import numpy as np

nlp = spacy.load('en_core_web_md')
# This model uses 300 dimensional representation space
embed_dim = 300
epsilon = 1e-2 # to prevent zero division when all of the tokens are in out of vocabulary (oov). This mostly happens when the abstract/title/text is chinese

# mean of mean similarity: inner mean is for keyphrases(mean of keywords making the keyphrase) while outer mean is for the mean of all the vectors obtained
# cross attention mean of mean similarity: inner mean is for keyphrases(mean of keywords making the keyphrase) while for the vectors, we do a cross attention kind of mean.

def cosine_similarity(vec_1: np.ndarray, vec_2: np.ndarray) -> float:
    return np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2) + epsilon)

def mm_vector_representation(keyphrases: str) -> np.ndarray:
    keyphrases = keyphrases.split(";")

    single_keywords = []
    multi_keywords = []
    for item in keyphrases:
        if " " in item:
            multi_keywords.append(item)
        else:
            single_keywords.append(item)

    single_keywords_vector_sum = np.zeros((embed_dim,))
    multi_keywords_vector_sum = np.zeros((embed_dim,))

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

    return (single_keywords_vector_sum + multi_keywords_vector_sum) / (single_keywords_data_count + multi_keywords_data_count + epsilon)

def mean_of_mean_similarity(text1: str, text2: str) -> float:
    vec1 = mm_vector_representation(text1)
    vec2 = mm_vector_representation(text2)
    return round(cosine_similarity(vec1, vec2) * 5, 2)

def camm_vector_representation(keyphrases: str) -> np.ndarray:
    keyphrases = keyphrases.split(";")

    single_keywords = []
    multi_keywords = []
    for item in keyphrases:
        if " " in item:
            multi_keywords.append(item)
        else:
            single_keywords.append(item)

    single_keywords_vector = []
    multi_keywords_vector = []

    single_keywords_data = nlp(" ".join(single_keywords))
    for token in single_keywords_data:
        if not token.is_oov:
            single_keywords_vector.append(token.vector)

    for keyphrase in multi_keywords:
        # if x = nlp("computer science"), then x.vector is the average gloVe(maybe) vector of "computer" and "science", which is the method we're using to deal with keyphrases
        if (vec := nlp(keyphrase)).vector_norm != 0.0:
            multi_keywords_vector.append(vec.vector)

    single_keywords_vector.extend(multi_keywords_vector)
    return single_keywords_vector

def ca_mean_of_mean_similarity(text1: str, text2: str) -> float:
    """
    ca stands for Cross Attention.
    It's named so because the operation done before computing mean is similar to cross attention.
    It's an experimental feature, I'm not sure if it's used anywhere else.
    """
    vec1 = camm_vector_representation(text1)
    vec2 = camm_vector_representation(text2)
    total_similarity = 0
    for x in vec1:
        for y in vec2:
            total_similarity += cosine_similarity(x, y)
    return round((total_similarity / (len(vec1) * len(vec2) + epsilon)) * 5, 2)

def jaccard_similarity(text1: str, text2: str) -> float:
    keywords_a = set(text1.split(";"))
    keywords_b = set(text2.split(";"))

    intersection = len(keywords_a.intersection(keywords_b))
    union = len(keywords_a.union(keywords_b))
    return round((intersection / union) * 5, 2)

similarity_dict = {"mean_of_mean-max":{"sim":mean_of_mean_similarity, "mode":"max"},
                "mean_of_mean-mean":{"sim":mean_of_mean_similarity, "mode":"mean"},
                "ca_mean_of_mean-max":{"sim":ca_mean_of_mean_similarity, "mode":"max"},
                "ca_mean_of_mean-mean":{"sim":ca_mean_of_mean_similarity, "mode":"mean"},
                "jaccard-max":{"sim":jaccard_similarity, "mode":"max"},
                "jaccard-mean":{"sim":jaccard_similarity, "mode":"mean"}}