
# -*- coding: windows-1251 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
import csv

# вказуємо ім'я файлу та список заголовків стовпців
filename = "data.csv"
fields = ['url', 'text', 'label']

# список даних
rows = [
        ['http://www.ua-football.com/', 'футбол', 'Спорт'],
        ['http://dynamo.kiev.ua/', 'команда', 'Спорт'],
        ['http://sportnews.com.ua/', 'гравці', 'Спорт'],
        ['http://terrikon.com/', 'правила гри', 'Спорт'],
        ['http://1927.kiev.ua/', 'вправи', 'Спорт'],
        ['http://sportnews.com.ua/', 'рецепт', 'Їжа'],
        ['https://klopotenko.com/', 'консистенція', 'Їжа'],
        ['https://picantecooking.com/', 'смак', 'Їжа'],
        ['http://patelnya.com.ua/', 'посуд', 'Їжа'],
        ['https://www.smachno.in.ua/', 'грам', 'Їжа'],
        ['http://cookery.com.ua/', 'приготування', 'Їжа'],
        ['https://weheartit.com/', 'колір', 'Дизайн'],
        ['https://www.behance.net/', 'інтерфейс', 'Дизайн'],
        ['https://dribbble.com/', 'креатив', 'Дизайн'],
        ['https://www.awwwards.com/', 'дизайн', 'Дизайн'],
        ['https://www.smashingmagazine.com/', 'веб-дизайн', 'Дизайн'],
        ['https://designmodo.com/', 'шаблони', 'Дизайн'],
        ['https://www.nytimes.com/', 'політика', 'Політика'],
        ['https://www.bbc.com/', 'дипломатія', 'Політика'],
        ['https://www.cnn.com/', 'світ', 'Політика'],
        ['https://www.reuters.com/', 'економіка', 'Політика'],
        ['https://www.theguardian.com/', 'країна', 'Політика'],
        ['https://www.washingtonpost.com/', 'партнери', 'Політика'],
        ['https://www.themuse.com/', 'кредитні картки', 'Бізнес'],
        ['https://www.entrepreneur.com/', 'підприємництво', 'Бізнес'],
        ['https://www.forbes.com/', 'бізнес-новини', 'Бізнес'],
        ['https://www.bloomberg.com/', 'фінанси', 'Бізнес'],
        ['https://www.ft.com/', 'торгівля', 'Бізнес'],
        ['https://techcrunch.com/', 'технології', 'Технології'],
        ['https://www.theverge.com/', 'гаджети', 'Технології'],
        ['https://www.wired.com/', 'пристрої', 'Технології'],
        ['https://www.cnet.com/', 'комп\'ютери', 'Технології'],
        ['https://www.techradar.com/', 'мобільні пристрої', 'Технології'],
        ['https://www.apple.com/', 'гаджети', 'Технології'],
        ['https://www.playstation.com/', 'ігри', 'Ігри'],
        ['https://www.xbox.com/', 'геймпад', 'Ігри'],
        ['https://www.nintendo.com/', 'консолі', 'Ігри'],
        ['https://corp.xumo.com/', 'фільм', 'Відео'],
        ['https://www.apple.com/ua/apple-tv-plus/', 'серіал', 'Відео'],
        ['https://sweet.tv/', 'ролик', 'Відео'],
        ['https://www.imdb.com/', 'відео', 'Відео'],
        ['https://megogo.net/ua', 'телебачення', 'Відео'],
        ['https://www.youtube.com/?gl=UA&hl=ru', 'канали онлайн', 'Відео'],
        ['https://www.asos.com/', 'купівля', 'Товари'],
        ['https://www.ikea.com/', 'товари', 'Товари'],
        ['https://best.aliexpress.com/?lan=en&spm=a2g0o.best.1000002.1.39e72c25Dlp1Wz', 'споживачі', 'Товари'],
        ['https://www.ebay.co.uk/', 'клієнти', 'Товари'],
        ['https://www.walmart.com/?veh=aff&wmlspartner=imp_143303&sharedid=&affiliates_ad_id=565706&campaign_id=9383&irgwc=1&sourceid=imp_3dnwjjVbexyNWIQzAexm0VcCUkAQQzQ1AzebVc0&veh=aff&wmlspartner=imp_143303&clickid=3dnwjjVbexyNWIQzAexm0VcCUkAQQzQ1AzebVc0&sharedid=&affiliates_ad_id=565706&campaign_id=9383', 'платформа', 'Товари'],
        ['https://rakuten.ua/', 'магазин', 'Товари'],
]
#записуємо дані у файл
with open(filename, 'w', newline='', encoding='windows-1251') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

#завантажуємо дані
data = pd.read_csv(filename, encoding='windows-1251')

#вилучаємо невикористовувані стовпці та рядки з нульовими значеннями
data.dropna(subset=['url', 'text', 'label'], inplace=True)
data.drop(['url'], axis=1, inplace=True)

#розбиваємо дані на навчальну та тестову вибірки
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

#побудова векторів TF-IDF
tfidf_vectorizer = TfidfVectorizer()
train_vectors = tfidf_vectorizer.fit_transform(train_data['text'])
test_vectors = tfidf_vectorizer.transform(test_data['text'])

#побудова та тренування моделі
clf = MLPClassifier(hidden_layer_sizes=(70,), max_iter=700, alpha=0.0001,
solver='adam', random_state=42, learning_rate='adaptive', verbose=True)
clf.fit(train_vectors, train_data['label'])

#передбачення та оцінка точності моделі
predicted = clf.predict(test_vectors)
accuracy = accuracy_score(test_data['label'], predicted)
f1 = f1_score(test_data['label'], predicted, average='weighted')

print("Accuracy: {:.3f}, F1 score: {:.3f}".format(accuracy, f1))

#протестування моделі на нових даних
new_data = pd.DataFrame([
['https://harchi.info/', 'харчування'],
['https://www.amazon.com/', 'онлайн-магазин'],
['https://www.netflix.com/', 'онлайн-відео']
], columns=['url', 'text'])

new_vectors = tfidf_vectorizer.transform(new_data['text'])
new_predicted = clf.predict(new_vectors)

print("Predictions for new data:")
for url, label in zip(new_data['url'], new_predicted):
        print("{}: {}".format(url, label))
