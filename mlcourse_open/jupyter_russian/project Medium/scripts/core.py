# -*- coding: utf-8 -*-



import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge, LinearRegression
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, save_npz
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


from tools import write_submission_file,translate_to_english
from preprocessing import variant_1_preprocessing
from sklearn.feature_extraction.text import CountVectorizer

train_target = pd.read_csv(os.path.join('../medium data/', 'train_log1p_recommends.csv'), index_col='id')
y_train = train_target['log_recommends'].values #4.33328

# X_train, X_test = variant_1_preprocessing()




#contents_test_translated = list(pd.read_csv('../medium data/test_translated2.csv')['content'])
#contents_train_translated = list(pd.read_csv('../medium data/contents_train_translated.csv')['content'])
#
## TO LOWER
#contents_train_translated = list(map(lambda x: x.lower(), contents_train_translated))
#contents_test_translated = list(map(lambda x: x.lower(), contents_test_translated))

# STOPWORDS
from nltk.corpus import stopwords
from nltk import word_tokenize
from string import punctuation
stop_words = set(stopwords.words('english'))
stop_words.update(set([x for x in punctuation] +
                      ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'] +
                      ['“', '”', '—', '``', "''", '‘', '’', '«', '»', '●', '😂', '😛', '']))


from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()

def clearing_text(text):
    clear_text = [i for i in word_tokenize(text) if i not in stop_words]
    clear_text_split_error_words = list()
    for word in clear_text:
        if '.' in word:
            word = [porter_stemmer.stem(x) for x in word.split('.') if ((x != '') and (x not in stop_words) and (len(x) > 1))]
            clear_text_split_error_words.extend(word)
        else:
            clear_text_split_error_words.append(porter_stemmer.stem(word))

    return ' '.join(clear_text_split_error_words)






#contents_train_translated_clear = list()
#for i in range(len(contents_train_translated)):
#    contents_train_translated_clear.append(clearing_text(contents_train_translated[i]))
#    if i%1000 == 0:
#        print(i)
#
#
#contents_test_translated_clear = list()
#for i in range(len(contents_test_translated)):
#    contents_test_translated_clear.append(clearing_text(contents_test_translated[i]))
#    if i%1000 == 0:
#        print(i)
#
#pd.DataFrame(contents_train_translated_clear,columns=['content']).to_csv('contents_train_translated_clear.csv',index=False)
#pd.DataFrame(contents_test_translated_clear,columns=['content']).to_csv('contents_test_translated_clear.csv',index=False)

contents_train_translated_clear = list(pd.read_csv('contents_train_translated_clear.csv')['content'])
contents_test_translated_clear = list(pd.read_csv('contents_test_translated_clear.csv')['content'])

# Load fitches
from loaders import load_publisheds, load_titles, load_domains, load_authors, load_link_tags, load_meta_tags
help_data_train = pd.DataFrame()
help_data_test = pd.DataFrame()
#help_data_train['published'] = load_publisheds('../medium data/train.json')
#help_data_train['published'] = help_data_train['published'].apply(pd.to_datetime)
#help_data_train['year'] = help_data_train['published'].apply(lambda x: x.year)
#help_data_train['month'] = help_data_train['published'].apply(lambda x: x.month)
#help_data_train['day'] = help_data_train['published'].apply(lambda x: x.day)
#help_data_train['year_month_day'] = help_data_train['year']*100+help_data_train['month']
#help_data_train.drop(axis=1,labels = 'published', inplace=True)
#
#scaler_year = StandardScaler()
#year_scaled_train = scaler_year.fit_transform(help_data_train[['year_month_day']])

help_data_train['domain'] = load_domains('../medium data/train.json')
help_data_test['domain'] = load_domains('../medium data/test.json')

help_data_train['autor'] = load_authors('../medium data/train.json')['twitter']
help_data_train['autor'] = help_data_train['autor'].fillna('Noname')

help_data_test['autor'] = load_authors('../medium data/test.json')['twitter']
help_data_test['autor'] = help_data_test['autor'].fillna('Noname')

#CountVectorizer_description = CountVectorizer()
#help_data_train['description'] = load_meta_tags('../medium data/train.json')['description'].apply(clearing_text)
#help_data_train_description = CountVectorizer_description.fit_transform(help_data_train['description'])
#
#help_data_test['description'] = load_meta_tags('../medium data/test.json')['description'].apply(clearing_text)
#help_data_test_description = CountVectorizer_description.transform(help_data_test['description'])

#help_data_test['published'] = load_publisheds('../medium data/test.json')
#help_data_test['published'] = help_data_test['published'].apply(pd.to_datetime)
#help_data_test['year'] = help_data_test['published'].apply(lambda x: x.year)
#help_data_test['month'] = help_data_test['published'].apply(lambda x: x.month)
#help_data_test['day'] = help_data_test['published'].apply(lambda x: x.day)
#help_data_test['year_month_day'] = help_data_test['year']*100+help_data_test['month']
#help_data_test.drop(axis=1,labels = 'published', inplace=True)

#year_scaled_test = scaler_year.transform(help_data_test[['year_month_day']])

CountVectorizer_title = CountVectorizer()
help_data_train['title'] = load_titles('../medium data/train.json')
help_data_train['title'] = help_data_train['title'].apply(clearing_text)
#help_data_train['title'] = help_data_train['title'].apply(translate_to_english)
help_data_train_title = CountVectorizer_title.fit_transform(help_data_train['title'])

help_data_test['title'] = load_titles('../medium data/test.json')
help_data_test['title'] = help_data_test['title'].apply(clearing_text)
#help_data_test['title'] = help_data_test['title'].apply(translate_to_english)
help_data_test_title = CountVectorizer_title.transform(help_data_test['title'])


a = LabelEncoder().fit_transform(pd.concat([help_data_train['domain'],help_data_test['domain']]))
a = OneHotEncoder().fit_transform(a.reshape(-1, 1))
help_data_train_domain = a[:len(help_data_train['domain'])]
help_data_test_domain = a[len(help_data_train['domain']):]


a = LabelEncoder().fit_transform(pd.concat([help_data_train['autor'],help_data_test['autor']]))
a = OneHotEncoder().fit_transform(a.reshape(-1, 1))
help_data_train_autor = a[:len(help_data_train['autor'])]
help_data_test_autor = a[len(help_data_train['autor']):]


#pd.DataFrame(help_data_train_title,columns=['content']).to_csv('train_title_en.csv',index=False)
#pd.DataFrame(help_data_test_title,columns=['content']).to_csv('test_title_en.csv',index=False)

#Длина статьи
scaler_len = StandardScaler()
help_data_train['len'] = [len(i) for i in contents_train_translated_clear]
help_data_test['len'] = [len(i) for i in contents_test_translated_clear]
help_data_train_len = scaler_len.fit_transform(help_data_train['len'].reshape(-1, 1))
help_data_test_len = scaler_len.transform(help_data_test['len'].reshape(-1, 1))

#Дата публикации OHE
a = LabelEncoder().fit_transform(pd.concat([help_data_train['year'],help_data_test['year']]))
a = OneHotEncoder().fit_transform(a.reshape(-1, 1))
help_data_train_year = a[:len(help_data_train['year'])]
help_data_test_year = a[len(help_data_train['year']):]



from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=100000, ngram_range=(1, 2))
X_train = tfidf_vectorizer.fit_transform(contents_train_translated_clear)
X_test = tfidf_vectorizer.transform(contents_test_translated_clear)

data_train = csr_matrix(hstack([help_data_train_title, help_data_train_domain, help_data_train_autor, help_data_train_len, X_train]))
data_test = csr_matrix(hstack([help_data_test_title, help_data_test_domain, help_data_test_autor, help_data_test_len, X_test]))



 # SUBMISSION
ridge = Ridge(random_state=17, alpha = 0.01)
ridge.fit(data_train, y_train)
ridge_test_pred = ridge.predict(data_test)

ridge_test_pred2=ridge_test_pred+(4.33328-ridge_test_pred.mean())

write_submission_file(prediction=ridge_test_pred2,  filename='submission2.csv')



#lr = LinearRegression()
#lr.fit(data_train, y_train)
#lr_test_pred = lr.predict(data_test)
#write_submission_file(prediction=lr_test_pred,  filename='submission_lr.csv')




# подключим библиотеки keras 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor


# Опишем нашу сеть.
def baseline_model():
    model = Sequential()
    model.add(Dense(128, input_dim=Xtr.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    
def split(train,y,ratio):
    idx = ratio
    return train[:idx, :], train[idx:, :], y[:idx], y[idx:]
    
Xtr, Xval, ytr, yval = split(data_train, y_train, 45000)

estimator = KerasRegressor(build_fn=baseline_model,epochs=20, nb_epoch=20, batch_size=64,validation_data=(Xval, yval), verbose=1)


estimator.fit(Xtr, ytr)











def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
















