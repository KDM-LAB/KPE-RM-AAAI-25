from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ml6team/keyphrase-generation-keybart-inspec")
key_model = AutoModelForSeq2SeqLM.from_pretrained("ml6team/keyphrase-generation-keybart-inspec")

def keybart_keywords(text: str, type: str | None = None):
    """
    text: The text string.
    type: Can be either 'tit' for title or 'abs' for abstract. Anything else will be for pdf text.
    But since keybart doesn't have a "top_n" argument, "type" argument won't do anything. It's just there to maintain similarity.
    """
    inputs = tokenizer([text], truncation=True, padding="longest", return_tensors="pt")
    outputs = key_model.generate(**inputs, max_length=350, num_beams=20, early_stopping=True) # seems like max_length of 350 gives the maximum keyphrases even in pdf_text. num_beams changes the keyphrases.
    keywords = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    keywords = keywords.replace("-", " ")
    keywords = keywords.split(" ; ")
    if " ;" in keywords[-1]:
        keywords[-1] = keywords[-1].replace(" ;", "")
    return keywords
