
# coding: utf-8

# In[230]:

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import sklearn


# In[243]:

#nltk.download_shell()
media = pd.read_csv('Desktop/Imbalanced Data set/Media/train.csv')
media_eval = pd.read_csv('Desktop/Imbalanced Data set/Media/evaluation.csv')


# In[148]:

media[0:3]


# In[45]:

print(len(media))


# In[263]:

media.describe()


# In[265]:

media_eval.info()


# In[16]:

media.groupby('label').describe()


# In[258]:

media_null = media.isnull().sum() #To check the values in each cols
media_eval_null = media_eval.isnull().sum()


# In[262]:

media_eval_null


# In[266]:

## Filling the NA Values
media['url'] = media['url'].fillna('Not Available')
media['additionalAttributes'] = media['additionalAttributes'].fillna('Not Available')
media['breadcrumbs'] = media['breadcrumbs'].fillna('Not Available')


# In[267]:

# Filling the NA Values
media_eval['url'] = media_eval['url'].fillna('Not Available')
media_eval['additionalAttributes'] = media_eval['additionalAttributes'].fillna('Not Available')
media_eval['breadcrumbs'] = media_eval['breadcrumbs'].fillna('Not Available')


# In[268]:

media[0:100]


# In[269]:

media_eval[0:100]


# In[72]:

media['length_breadcrumbs'] = media['breadcrumbs'].apply(len)
media['length_additionalAttributes'] = media['additionalAttributes'].apply(len)


# In[75]:

get_ipython().magic('matplotlib inline')


# In[78]:

media['length_breadcrumbs'].plot.hist(bins=100)


# In[87]:

media['length_additionalAttributes'].plot.hist(bins=100).axis([0,1500,0,450000])


# In[88]:

media['length_breadcrumbs'].describe()


# In[89]:

media['length_additionalAttributes'].describe()


# In[92]:

media[media['length_breadcrumbs'] == 2195]['breadcrumbs'].iloc[0]


# In[96]:

media[media['length_additionalAttributes'] == 4977]['additionalAttributes'].iloc[0]


# In[103]:

media.hist(column = 'length_breadcrumbs',by = 'label',bins = 50,figsize = (20,8))
#Checking the length of breadcrumbs and plotting


# In[105]:

media.hist(column = 'length_additionalAttributes',by = 'label',bins = 50,figsize = (20,8))
#Checking the length of additionalattributes and plotting


# In[175]:

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[251]:

def text_cleanup(med_data):
    """
    1. Remove Punctuation
    2. Remove stop words
    3. Return a list of clean text
    4. Stemming of the words performed
    """
    
    rm_punc = [char for char in med_data if char not in string.punctuation]
    rm_punc = ''.join(rm_punc)
    stemmer =  PorterStemmer()
    rm_punc = stemmer.stem(rm_punc)
    
    return [word for word in rm_punc.split() if word.lower() not in stopwords.words('english')]


# In[254]:

media['breadcrumbs'].head(5).apply(text_cleanup)


# In[255]:

media_eval['breadcrumbs'].head(5).apply(text_cleanup)


# In[256]:

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 


# In[270]:

# Generating the countVectorizer for training set
bow_trans = CountVectorizer(analyzer=text_cleanup).fit(media['breadcrumbs'])  


# In[273]:

print(len(bow_trans.vocabulary_)) ##Printing the length


# In[181]:

media_mess = media['breadcrumbs'][4]


# In[182]:

media_mess


# In[191]:

bowt = bow_trans.transform([media_mess])


# In[195]:

print(bowt)  
print(bowt.shape)  ## Printing the shape


# In[196]:

print(bow_trans.get_feature_names()[13010]) ## Finding the words which appear twice


# In[275]:

## To generate bag of words count and vocabulary has been fitted for training set
media_message = bow_trans.transform(media['breadcrumbs'])
media_message.shape


# In[300]:

## Baf of words Transformation of the evaluation set and fiited vocabulary of the training set
media_eval_message1 = bow_trans.transform(media_eval['breadcrumbs'])


# In[198]:

media_message.nnz


# In[218]:

#Sparsity of the training set
print (100 * media_message.nnz / (media_message.shape[0]*media_message.shape[1]))


# In[277]:

#Sparsity of the evaluation set
print (100 * media_eval_message.nnz / (media_eval_message.shape[0]*media_eval_message.shape[1]))


# In[278]:

### TFIDF Implementation on training set
media_tfidf = TfidfTransformer().fit(media_message)


# In[280]:

tfidf_trans = media_tfidf.transform(bowt)


# In[281]:

print(tfidf_trans)


# In[282]:

media_message_tfidf = media_tfidf.transform(media_message)


# In[304]:

## TFIDF Implemation of evaluation set
media_eval_tfidf1 = TfidfTransformer().fit(media_eval_message1)


# In[305]:

media_eval_message_tfidf1 = media_eval_tfidf.transform(media_eval_message1)


# In[322]:

print (media_eval_message_tfidf1.shape)


# In[231]:

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,f1_score,accuracy_score,confusion_matrix


# In[284]:

## Naive bayes classfier
media_classfication = MultinomialNB().fit(media_message_tfidf,media['label'])


# In[233]:

media_classfication.predict(tfidf_trans)


# In[234]:

media['label'][4]


# In[235]:

media_predict = media_classfication.predict(media_message_tfidf)  # Predicting the training set


# In[306]:

media_eval_predict = media_classfication.predict(media_eval_message_tfidf1)  #Predicting the evaluation set


# In[307]:

print(media_eval_predict)


# In[314]:

media_eval['label'] = media_eval_predict


# In[321]:

writer = pd.DataFrame(media_eval).to_csv('media_eval.csv')


# In[312]:

accuracy_score(media['label'],media_predict)  #Accuracy Score of the training set 


# In[236]:

print (media_predict)  #Train Prediction


# In[313]:

confusion_matrix(media['label'], media_predict)  # Confusion Matrix of the training set


# In[242]:

classification_report(media['label'], media_predict)  #Precision, Recall, F1 Score of Training set


# In[ ]:



