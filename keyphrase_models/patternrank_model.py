from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT

kw_model = KeyBERT()

def patternrank_keywords(text: str, type: str | None = None):
    """
    text: The text string
    type: Can be either 'tit' for title or 'abs' for abstract. Anything else will be for pdf text.
    """
    if type == "tit":
        top_n = 3
    elif type == "abs":
        top_n = 27
    else:
        top_n = 30

    keywords = kw_model.extract_keywords(text, vectorizer=KeyphraseCountVectorizer(), top_n=top_n)

    keyphrases = [words[0] for words in keywords]

    length_keyphrases = 0
    for idx, keyphrase in enumerate(keyphrases):
        length_keyphrases += len(keyphrase)
        if length_keyphrases >= 900: # for 1000 char limit in db to store keywords
            return keyphrases[:idx]

        if "-" in keyphrase:
            keyphrases[idx] = keyphrase.replace("-", " ")

    return keyphrases

