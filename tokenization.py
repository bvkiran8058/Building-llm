import os
from natsort import natsorted
folder_path = r"Mahabharatha Datasets"
files = natsorted(os.listdir(folder_path))
corpus = ''
for file in files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read() 
        print(f"Path: {file_path} | Length: {len(content)}")
        corpus += content + '\n'

vocab = list(set(corpus))
print(sorted(vocab))
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}
def encode(text):
    return [token_to_id[token] for token in text]
def decode(token_ids):
    return ''.join([id_to_token[token_id] for token_id in token_ids])

tokens = encode("Bhisma is very wise.")
print(f"Tokens: {tokens}")
text = decode(tokens)
print(f"Text: {text}")