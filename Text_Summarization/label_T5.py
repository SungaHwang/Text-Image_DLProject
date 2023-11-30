# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk

train_data = pd.read_csv('Data/train_data.csv')

review_list = []
for i in range(0, len(train_data['Review'])):
    review_list.append(train_data['Review'][i])

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
nltk.download('punkt')

model_dir = "lcw99/t5-base-korean-text-summary"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 128

data = {
    'text': review_list
}

df = pd.DataFrame(data)

def summary(row):
    text = str(row['text'])
    input_ids = tokenizer.encode(text)
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

# Print the resulting DataFrame
print(df[['text', 'predicted']])

df.to_csv('train_review_label_T5.csv', encoding= 'utf-8-sig', index=False, header=True)