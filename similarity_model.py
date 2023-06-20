import spacy

nlp = spacy.load('en_core_web_md')
epsilon = 1e-4 # to prevent zero division when all of the tokens are in out of vocabulary (oov). This mostly happens when the abstract is chinese

def keyword_similarity(kw1, kw2):
    kw1 = set(kw1.split(";"))
    kw2 = set(kw2.split(";"))
    kw1 = " ".join(kw1)
    kw2 = " ".join(kw2)

    tokens1 = nlp(kw1)
    tokens2 = nlp(kw2)

    total_similarity = 0
    terms = 0

    # assuming none of the tokens are empty string
    for t1 in tokens1:
        for t2 in tokens2:
            if not (t1.is_oov or t2.is_oov): # checking if both words are in spacy vocab and has embeddings
                total_similarity += t1.similarity(t2)
                terms += 1
                
    return round((total_similarity/(terms + epsilon) * 5), 2) # it returns similarity [0,1] so to make similarity [0,5]

