import warnings
warnings.filterwarnings('ignore')

from transformers import logging
logging.set_verbosity_error()

import tensorflow as tf
import numpy as np
import pandas as pd
np.set_printoptions(threshold = np.inf, linewidth = np.inf)

with open('test_data_상의.txt', encoding='utf-8') as f:
    docs = [doc.strip().split('\t') for doc in f ]
    docs = [(doc[0]) for doc in docs if len(doc) == 1]
    docs

#print(docs)

from transformers import AutoTokenizer, TFElectraForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")


test_data_상의_tokenized = tokenizer(docs, return_tensors="np", max_length=64, padding='max_length', truncation=True)
print(test_data_상의_tokenized[0])


checkpoint_filepath = "./checkpoints/checkpoint_electra_kr"
model = TFElectraForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022", num_labels=2, from_pt=True)
model.load_weights(checkpoint_filepath)

y_preds = model.predict(dict(test_data_상의_tokenized))
prediction_probs_total = tf.nn.softmax(y_preds.logits,axis=1).numpy()

print(len(docs))

with open('score/results_total.txt', 'w') as results_total:   
    for value in prediction_probs_total:
        for value2 in value:
            results_total.write(f"{value2}\n")


with open('score/results_total.txt', encoding='utf-8') as f:
    docs = [doc.strip().split('\t') for doc in f ]
    docs = [(doc[0]) for doc in docs if len(doc) == 1]
    docs

print(len(docs) /2)

긍정 = []
for i in range(0, len(docs)):
    if i % 2 == 1:
        긍정.append(float(docs[i]))
print(긍정)
print(len(긍정))

total = 0
for i in range(0, len(긍정)):
    total = total+ 긍정[i]
total_score = total / len(긍정)
print(total_score)


if(total_score> 0.5):
    print("{:.2f}% 긍정입니다.".format(total_score * 100))
else:
    print("{:.2f}% 부정입니다.".format((1 - total_score) * 100))

with open('score/total_score.txt', 'w', encoding="UTF-8") as text_file:
    if(total_score> 0.5):
        print("{:.2f}% 긍정입니다.".format(total_score * 100), file = text_file)
    else:
        print("{:.2f}% 부정입니다.".format((1 - total_score) * 100), file = text_file)