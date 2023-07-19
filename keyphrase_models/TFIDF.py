import spacy
import pke

nlp = spacy.load('en_core_web_sm')
nlp.max_length = 2000000 # to get rid of max_length error which came in paper_pk 2097
extractor = pke.unsupervised.TfIdf()

def TFIDF_keywords(text: str, type: str | None = None):
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

    extractor.load_document(input=text, language='en', spacy_model=nlp)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=top_n)

    return [keyphrase[0].replace("-", " ") if "-" in keyphrase[0] else keyphrase[0] for keyphrase in keyphrases]
