from bertopic import BERTopic
import pandas as pd

df = pd.read_csv('/Users/sungahwang/Text-Image_DLProject/Text_Summarization/Data/train_data.csv')
print(df)

num_topics = 9
topic_model = BERTopic(language="korean", nr_topics=num_topics)

topics, probabilities = topic_model.fit_transform(df['Review'].tolist())

topic_model.get_topic_info()

topic_model.visualize_barchart(n_words = 10)