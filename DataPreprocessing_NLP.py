import re
import pandas as pd
import numpy as np
import json
import csv
from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

#DATA_IN_PATH='D:\한이음\\'
# TODO 데이터 경로를 변수로 설정

train_data=pd.read_csv("USPTO_PatentVariable_table_1.csv",encoding="cp949",header=0,delimiter=",",quotechar = '"',quoting=csv.QUOTE_ALL)

'''
print(train_data['abstract'][0])

첫번째 요약 데이터 출력 확인

abstract=train_data['abstract'][0]
abstract_text=abstract
abstract_text=re.sub("[^a-zA-Z]"," ",abstract_text) # TODO 영어 문자를 제외한 나머지는 모두 공백으로

print(abstract_text)


마침표 사라진 것 확인

stop_words=set(stopwords.words('english')) # TODO 영어 불용어 set을 만든다.

abstract_text=abstract_text.lower()
words=abstract_text.split() # TODO 소문자로 변환한 후 단어마다 나눠서 단어 리스트로 만든다.
words=[w for w in words if not w in stop_words] # TODO 불용어를 제거한 리스트를 만든다.

print(words)


문자열이었던 abstract이 단어 리스트로 바뀐 것을 확인
모델에 적용시키기 위해서는 다시 하나의 문자열로 합쳐야 한다.


clean_abstract=' '.join(words)
print(clean_abstract)


단어 리스트가 하나의 문자열로 바뀐 것을 확인

이제 전체 데이터에 적용

전처리 과정을 하나의 함수로 정의

'''
def preprocessing(abstract, remove_stopwords=False):
    # TODO 불용어 제거는 옵션으로 선택 가능

    abstract_text = abstract

    # TODO 1. 영어가 아닌 특수문자를 공백으로 바꾸기
    abstract_text = re.sub("[^a-zA-Z]", " ", abstract_text)

    # TODO 2. 대문자를 소문자로 바꾸고 공백 단위로 텍스트를 나눠서 리스트를 받는다.
    words = abstract_text.lower().split()

    if remove_stopwords:
        # TODO 3. 불용어 제거

        # TODO 영어 불용어 제거 불러오기
        stops=set(stopwords.words('english'))

        # TODO 불용어가 아닌 단어로 이뤄진 새로운 리스트 생성
        words = [w for w in words if not w in stops]

        # TODO 4. 단어 리스트를 공백을 넣어서 하나의 글로 합친다.
        clean_abstract = ' '.join(words)

    else: # 불용어를 하지 않을 때
        clean_abstract = ' '.join(words)

    return clean_abstract

clean_train_abstracts=[]
for abstract in train_data['abstract']:
    clean_train_abstracts.append(preprocessing(abstract,remove_stopwords=True))


clean_train_df=pd.DataFrame({'abstract':clean_train_abstracts})

tokenizer=Tokenizer()
tokenizer.fit_on_texts(clean_train_abstracts)
text_sequences=tokenizer.texts_to_sequences(clean_train_abstracts)

# TODO 각 요약이 텍스트가 아닌 인덱스의 벡터로 구성될 것이다.

word_vocab=tokenizer.word_index

print(word_vocab)

# TODO 각 인덱스가 어떤 단어를 의미하는지 알 수 있도록 단어사전을 생성

print("전체 단어 개수: ",len(word_vocab))

# TODO 단어는 총 10000여개

data_configs={}

data_configs['vocab']=word_vocab
data_configs['vocab size']=len(word_vocab)+1

# TODO 데이터에 대한 정보인 단어 사전과 전체 단어 개수는 새롭게 딕셔너리에 저장


MAX_SEQUENCE_LENGTH=119 # TODO 문장 최대 길이 -> 앞서 데이터 분석과정에서 나온 단어 개수의 중간값 사용

train_inputs=pad_sequences(text_sequences,maxlen=MAX_SEQUENCE_LENGTH,padding='post')

print('Shape of train data: ',train_inputs.shape)

# TODO 패딩처리로 모든 데이터가 119라는 길이를 가지게 됨

'''
예시를 보면 라벨, 정답을 알려주는 값을 저장하는데 예시는 리뷰 데이터라 긍정부정을 라벨로 넘겼다.
근데 우리의 데이터는 정답이라고 할 만한 라벨 값이 없다.

어떤 것을 예측하는 모델은 만들 수 없다.

그리고 평가 데이터로 사용할 만한 데이터가 현재 없다. 
사실 있어도 어떻게 사용할 수 있을지 감이 잘 안온다.

성능 평가를 할 거라면 기준이 필요한데 우리에겐 현재 그 기준이 없다.



'''


TRAIN_INPUT_DATA='train_input2.npy'
TRAIN_CLEAN_DATA='train_clean2.csv'
DATA_CONFIGS='data_configs2.json'

np.save(open(TRAIN_INPUT_DATA,'wb'),train_inputs)
clean_train_df.to_csv(TRAIN_CLEAN_DATA,index=False)

json.dump(data_configs,open(DATA_CONFIGS,'w'),ensure_ascii=False)

