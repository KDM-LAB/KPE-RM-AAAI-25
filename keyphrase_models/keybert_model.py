from keybert import KeyBERT

kw_model = KeyBERT()

def keybert_keywords(text: str, type: str | None = None):
    """
    text: The text string
    type: Can be either 'tit' for title or 'abs' for abstract. Anything else will be for pdf text.
    """
    if type == "tit":
        top_n = 3
    elif type == "abs":
        top_n = 12
    else:
        top_n = 15

    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1),
            stop_words='english',
            use_maxsum=True,
            nr_candidates=top_n+1,
            top_n=top_n)

    return [words[0] for words in keywords]
