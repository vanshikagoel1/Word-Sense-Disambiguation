import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import json

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings for a sentence
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Load dataset
df = pd.read_csv('semcor_dataset_with_embeddings.csv')

# Function to convert embeddings to string
def embeddings_to_string(embedding):
    return ','.join(map(str, embedding))

# Apply the function to convert embeddings to strings
bert_embeddings_str = df['embedding'].apply(lambda x: embeddings_to_string(json.loads(x)))

# Update the 'embedding' column with the string representations
df['embedding'] = bert_embeddings_str

# Save dataset with embeddings as strings
df.to_csv('semcor_dataset_with_embeddings.csv', index=False)
