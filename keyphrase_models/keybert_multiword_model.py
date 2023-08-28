# Two changes: 
# 1. Number of keyphrases is 30, to match non-dl methods (I tried with 30 keywords but it was taking a lot of time and memory, also methods like keybart and bertkpe were generating 10-15 keyphrases so i stick with 15 keyphrases).
# Also (1,3) range was adding on the time.
# 2. Instead of the model producing only unigrams, I changed the range to (1,3), i.e. it'll output unigram, biagram or even trigram

from keybert import KeyBERT

kw_model = KeyBERT()

def keybert_multiword_keywords(text: str, type: str | None = None):
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

    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3),
            stop_words='english',
            use_maxsum=True,
            nr_candidates=top_n+1,
            top_n=top_n)

    keyphrases = [words[0] for words in keywords]

    length_keyphrases = 0
    for idx, keyphrase in enumerate(keyphrases):
        length_keyphrases += len(keyphrase)
        if length_keyphrases >= 900: # for 1000 char limit in db to store keywords
            return keyphrases[:idx]

        if "-" in keyphrase:
            keyphrases[idx] = keyphrase.replace("-", " ")

    return keyphrases

