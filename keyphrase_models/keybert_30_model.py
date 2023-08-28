# Two changes: 
# Extracting 30 words. Changed extract_keywords method same to patternrank

from keybert import KeyBERT

kw_model = KeyBERT()

def keybert_30_keywords(text: str, type: str | None = None):
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

    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), top_n=top_n)

    keyphrases = [words[0] for words in keywords]

    length_keyphrases = 0
    for idx, keyphrase in enumerate(keyphrases):
        length_keyphrases += len(keyphrase)
        if length_keyphrases >= 900: # for 1000 char limit in db to store keywords
            return keyphrases[:idx]

        if "-" in keyphrase:
            keyphrases[idx] = keyphrase.replace("-", " ")

    return keyphrases

