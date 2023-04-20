
# -*- coding: windows-1251 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
import csv

# ������� ��'� ����� �� ������ ��������� ��������
filename = "data.csv"
fields = ['url', 'text', 'label']

# ������ �����
rows = [
        ['http://www.ua-football.com/', '������', '�����'],
        ['http://dynamo.kiev.ua/', '�������', '�����'],
        ['http://sportnews.com.ua/', '������', '�����'],
        ['http://terrikon.com/', '������� ���', '�����'],
        ['http://1927.kiev.ua/', '������', '�����'],
        ['http://sportnews.com.ua/', '������', '���'],
        ['https://klopotenko.com/', '������������', '���'],
        ['https://picantecooking.com/', '����', '���'],
        ['http://patelnya.com.ua/', '�����', '���'],
        ['https://www.smachno.in.ua/', '����', '���'],
        ['http://cookery.com.ua/', '������������', '���'],
        ['https://weheartit.com/', '����', '������'],
        ['https://www.behance.net/', '���������', '������'],
        ['https://dribbble.com/', '�������', '������'],
        ['https://www.awwwards.com/', '������', '������'],
        ['https://www.smashingmagazine.com/', '���-������', '������'],
        ['https://designmodo.com/', '�������', '������'],
        ['https://www.nytimes.com/', '�������', '�������'],
        ['https://www.bbc.com/', '���������', '�������'],
        ['https://www.cnn.com/', '���', '�������'],
        ['https://www.reuters.com/', '��������', '�������'],
        ['https://www.theguardian.com/', '�����', '�������'],
        ['https://www.washingtonpost.com/', '��������', '�������'],
        ['https://www.themuse.com/', '������� ������', '������'],
        ['https://www.entrepreneur.com/', '������������', '������'],
        ['https://www.forbes.com/', '�����-������', '������'],
        ['https://www.bloomberg.com/', '�������', '������'],
        ['https://www.ft.com/', '�������', '������'],
        ['https://techcrunch.com/', '�������㳿', '�������㳿'],
        ['https://www.theverge.com/', '�������', '�������㳿'],
        ['https://www.wired.com/', '�������', '�������㳿'],
        ['https://www.cnet.com/', '����\'�����', '�������㳿'],
        ['https://www.techradar.com/', '������ �������', '�������㳿'],
        ['https://www.apple.com/', '�������', '�������㳿'],
        ['https://www.playstation.com/', '����', '����'],
        ['https://www.xbox.com/', '�������', '����'],
        ['https://www.nintendo.com/', '������', '����'],
        ['https://corp.xumo.com/', '�����', '³���'],
        ['https://www.apple.com/ua/apple-tv-plus/', '�����', '³���'],
        ['https://sweet.tv/', '�����', '³���'],
        ['https://www.imdb.com/', '����', '³���'],
        ['https://megogo.net/ua', '�����������', '³���'],
        ['https://www.youtube.com/?gl=UA&hl=ru', '������ ������', '³���'],
        ['https://www.asos.com/', '������', '������'],
        ['https://www.ikea.com/', '������', '������'],
        ['https://best.aliexpress.com/?lan=en&spm=a2g0o.best.1000002.1.39e72c25Dlp1Wz', '���������', '������'],
        ['https://www.ebay.co.uk/', '�볺���', '������'],
        ['https://www.walmart.com/?veh=aff&wmlspartner=imp_143303&sharedid=&affiliates_ad_id=565706&campaign_id=9383&irgwc=1&sourceid=imp_3dnwjjVbexyNWIQzAexm0VcCUkAQQzQ1AzebVc0&veh=aff&wmlspartner=imp_143303&clickid=3dnwjjVbexyNWIQzAexm0VcCUkAQQzQ1AzebVc0&sharedid=&affiliates_ad_id=565706&campaign_id=9383', '���������', '������'],
        ['https://rakuten.ua/', '�������', '������'],
]
#�������� ��� � ����
with open(filename, 'w', newline='', encoding='windows-1251') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

#����������� ���
data = pd.read_csv(filename, encoding='windows-1251')

#�������� ���������������� ������� �� ����� � ��������� ����������
data.dropna(subset=['url', 'text', 'label'], inplace=True)
data.drop(['url'], axis=1, inplace=True)

#��������� ��� �� ��������� �� ������� ������
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

#�������� ������� TF-IDF
tfidf_vectorizer = TfidfVectorizer()
train_vectors = tfidf_vectorizer.fit_transform(train_data['text'])
test_vectors = tfidf_vectorizer.transform(test_data['text'])

#�������� �� ���������� �����
clf = MLPClassifier(hidden_layer_sizes=(70,), max_iter=700, alpha=0.0001,
solver='adam', random_state=42, learning_rate='adaptive', verbose=True)
clf.fit(train_vectors, train_data['label'])

#������������ �� ������ ������� �����
predicted = clf.predict(test_vectors)
accuracy = accuracy_score(test_data['label'], predicted)
f1 = f1_score(test_data['label'], predicted, average='weighted')

print("Accuracy: {:.3f}, F1 score: {:.3f}".format(accuracy, f1))

#������������� ����� �� ����� �����
new_data = pd.DataFrame([
['https://harchi.info/', '����������'],
['https://www.amazon.com/', '������-�������'],
['https://www.netflix.com/', '������-����']
], columns=['url', 'text'])

new_vectors = tfidf_vectorizer.transform(new_data['text'])
new_predicted = clf.predict(new_vectors)

print("Predictions for new data:")
for url, label in zip(new_data['url'], new_predicted):
        print("{}: {}".format(url, label))
