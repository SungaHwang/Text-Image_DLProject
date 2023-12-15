import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

train_data = pd.read_csv('train_data.csv')
print(train_data)
embedder = SentenceTransformer("jhgan/ko-sbert-sts")

train_data_embedding = embedder.encode(train_data['Review'].astype(str))
print(train_data_embedding)

model = DBSCAN(eps=0.1, min_samples=6, metric="cosine")
cluster = model.fit_predict(train_data_embedding)
print(model)
print(cluster)

train_data['dbscan'] = cluster

for cluster_num in range(6):
    if cluster_num == -1:
        continue
    else:
        print("Cluster num: {}".format(cluster_num))
        temp_df = train_data[train_data['dbscan'] == cluster_num]
        for review in temp_df['Review']:
            print(review)
        print()
