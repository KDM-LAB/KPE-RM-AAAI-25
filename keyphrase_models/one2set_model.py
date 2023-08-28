from keyphrase_models.kg_one2set.predict import predict_one2set

def one2set_keywords(text: str) -> str:
    keyphrases = predict_one2set(text)

    length_keyphrases = 0
    for idx, keyphrase in enumerate(keyphrases):
        length_keyphrases += len(keyphrase)
        if length_keyphrases >= 900: # for 1000 char limit in db to store keywords
            return ";".join(keyphrases[:idx])

        if "-" in keyphrase:
            keyphrases[idx] = keyphrase.replace("-", " ")

        if "," in keyphrase:
            keyphrases[idx] = keyphrase.replace(",", "")

    return ";".join(keyphrases)
