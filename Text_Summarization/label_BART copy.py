# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("digit82/kobart-summarization")

import tensorflow as tf
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
nltk.download('punkt')

train_data = pd.read_csv('Data/train_data.csv')

review_list = []
for i in range(0, len(train_data['Review'])):
    review_list.append(train_data['Review'][i])

data = {
    'text': review_list
}

df = pd.DataFrame(data)

def summary(row):
    text = str(row['text'])
    raw_input_ids = tokenizer.encode(text)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
    summary_ids = model.generate(torch.tensor([input_ids]),
                                do_sample = True,
                                num_beams = 21,  
                                repetition_penalty = 1000.1,
                                temperature = 0.7,
                                min_length = 10,
                                max_length = 32,
                                top_k = 20,
                                top_p = 0.95,
                                eos_token_id = 1)
    decoded_output = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens = True)
    predicted = nltk.sent_tokenize(decoded_output.strip())[0]
    print(predicted)
    return predicted

df['predicted'] = df.apply(summary, axis=1)

print(df[['text', 'predicted']])

df.to_csv('train_review_label_BART2.csv', encoding= 'utf-8-sig', index=False, header=True)