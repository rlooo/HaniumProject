import pandas as pd
import nltk
data = pd.read_csv(r'train_clean2.csv', error_bad_lines=False,encoding="cp949")

print(len(data))  #1000002개의 데이터

print(data.head(5)) # 상위 5개의 데이터만 출력

text = data[['abstract']] # text 열만 별도로 저장
print(text.head(5)) # 확인

text['abstract'] = text.apply(lambda row: nltk.word_tokenize(row['abstract']), axis=1) #문제해결을 위한 토큰화
print(text.head(5))

from nltk.stem import WordNetLemmatizer
text['abstract'] = text['abstract'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])
print(text.head(5))

tokenized_doc = text['abstract'].apply(lambda x: [word for word in x if len(word) > 3])
print(tokenized_doc[:5])
#X=data.fit_transform(data['abstract'])


detokenized_doc = []
for i in range(len(text)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

text['abstract'] = detokenized_doc # 다시 text['headline_text']에 재저장

print(text['abstract'][:5])


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(text['abstract'])
print(X.shape) # TF-IDF 행렬의 크기 확인

from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=15,learning_method='online',random_state=777,max_iter=1)


lda_top=lda_model.fit_transform(X)

print(lda_model.components_)
print(lda_model.components_.shape)

terms = vectorizer.get_feature_names() # 단어 집합. 1,000개의 단어가 저장됨.

def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])
get_topics(lda_model.components_,terms)
