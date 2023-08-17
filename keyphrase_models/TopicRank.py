import spacy
import pke

nlp = spacy.load('en_core_web_sm')
nlp.max_length = 2000000 # to get rid of max_length error which first came in paper_pk 2097
extractor = pke.unsupervised.TopicRank()

def TopicRank_keywords(text: str, type: str | None = None):
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

    result = []
    for keyphrase in keyphrases:
        if len(keyphrase[0]) <= 30: # there are some keywords which are more than 30 characters and seems useless. also this prevents 1000 char overflow in db
            if "-" in keyphrase[0]:
                result.append(keyphrase[0].replace("-", " "))
            else:
                result.append(keyphrase[0])

    return result

