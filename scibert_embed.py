import torch
from transformers import AutoTokenizer, AutoModel

scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states = True)
scibert_model.eval()

def scibert_embeddings(text):
    # tokens = scibert_tokenizer.tokenize(text[0])
    # print(f"{tokens =}")
    encoding = scibert_tokenizer.batch_encode_plus(text, padding=True, truncation=False, return_tensors="pt", add_special_tokens=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding["attention_mask"]
    # print(input_ids.shape, attention_mask.shape)

    with torch.no_grad():
        outputs = scibert_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[2]  # This contains the hidden states
    
    token_embeddings = torch.stack(hidden_states, dim=0) # converting the tuple into pytorch tensor
    token_embeddings = token_embeddings[-2][0] # taking second last layer output and first sentence (only sentence)
    # print(f"Shape of Word Embeddings: {token_embeddings.shape}")
    token_embeddings = token_embeddings[1:-1,:] # removing CLS and SEP token embeddings
    keyphrase_embedding = torch.mean(token_embeddings, dim=0)
    # print(f"Shape of Keyphrase Embeddings: {keyphrase_embedding.shape}")
    return keyphrase_embedding