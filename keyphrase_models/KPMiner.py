import pke

extractor = pke.unsupervised.KPMiner()

def KPMiner_keywords(text: str, type: str | None = None):
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

    extractor.load_document(input=text, language='en')
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=top_n)

    return [keyphrase[0].replace("-", " ") if "-" in keyphrase[0] else keyphrase[0] for keyphrase in keyphrases]
