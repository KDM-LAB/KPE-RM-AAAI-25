from keybert import KeyBERT

kw_model = KeyBERT()

def keyword_from_paper(text: str):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1))
    return [words[0] for words in keywords]
