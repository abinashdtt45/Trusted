from django.shortcuts import render
from .forms import NameForm
from django.http import HttpResponseRedirect
from selenium import webdriver
import random
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait # available since 2.4.0
# from selenium.webdriver.support import expected_conditions as EC # available since 2.26.0
from selenium.webdriver import Firefox
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
import array as arr
#import abinash.pkl 
#import extract_features.hdf5
########for machine learning##########
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
import re
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from nltk.corpus import stopwords
import string
from string import punctuation
import pickle
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
from sklearn.feature_extraction.text import TfidfVectorizer
# def index(request):
#     return render(request,'index.html')

MAX_DICT_WORDS = 55000
MAX_SEQ_LENGTH = 3500
EMBEDDING_DIM = 300
tokenizer = Tokenizer(num_words=MAX_DICT_WORDS, filters ='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')

def tokenize_data(texts):
    tokenizer.fit_on_texts(texts)
    stop = stopwords.words('english')
    tokens = [token.lower() for token in texts if token not in stop]
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    sequences = tokenizer.texts_to_sequences(tokens)
    data = pad_sequences(sequences, maxlen = MAX_SEQ_LENGTH)
    
    return data



def data(request):
    return render(request,'data.html')


def index(request):
    if request.method == 'POST':
        form = NameForm(request.POST)
        if form.is_valid():
            url = form.cleaned_data['url']
            final_url = url.split('&')[0]
            final_url = final_url.replace("/p/","/product-reviews/")
            updated_url = final_url
            reviews=[]
            # yaha pe
            
            for page in range(5):
                if(page>0):
                    p=page+1
                    updated_url = updated_url+'&page='+str(p)
                driver = webdriver.Firefox()
                driver.set_window_size(150, 150)
                driver.implicitly_wait(10)
                driver.get(updated_url)
                
                inputElement = driver.find_elements_by_class_name("qwjRop")

                for i in inputElement:
                    print(i.text)
                    reviews.append(i.text)
                driver.quit()
            # for i in range(1,len(reviews)):
            #     print reviews[i]
            #######################
            ngram_words=set(["no","not","very","just","some","few","less","more","really","so","too","much","didn't","don't","never","aren't","isn't"])
            sent_list =[]
            copy_of_reviews =reviews
            for i in range(0,len(reviews)):
                predf = random.choice([0,1])
                #print('ABC'+ copy_of_reviews[i])
                copy_of_reviews[i].encode('ascii', 'ignore').decode('ascii')
                reviews[i].encode('ascii', 'ignore').decode('ascii')
            tokenizer.fit_on_texts(reviews)
            stop = stopwords.words('english')
            tokens = [token.lower() for token in reviews if token not in stop]
            lmtzr = WordNetLemmatizer()
            tokens = [lmtzr.lemmatize(word) for word in tokens]
            sequences = tokenizer.texts_to_sequences(tokens)
            data = pad_sequences(sequences, maxlen = MAX_SEQ_LENGTH)
            # all processing)
            with open('abinash.pkl','rb') as file:
                clf=pickle.load(file)
                print('loded')
            
                json_file = open('model.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                # load weights into new model
                loaded_model.load_weights("model.h5")
                print("Loaded model from disk")
                print('loaded')

            X = loaded_model.predict(data)
            
            pred = clf.predict(X)
            print(pred)
            for i in range(0,len(pred)):
                if(pred[i]==1):
                    sent_list.append(copy_of_reviews[i])

            print(len(sent_list))

            print('Starting Puja')
            with open('puja.pkl', 'rb') as file:  
                clf1 = pickle.load(file)

            lot=[]
            english_stop_words = stopwords.words('english')
            for line in sent_list:
                words = re.split(r'\W+', line)
                words = [word.lower() for word in words]
                table = str.maketrans('', '', string.punctuation)
                stripped = [w.translate(table) for w in words]
                lemmatizer =WordNetLemmatizer()
                lem=[lemmatizer.lemmatize(word) for word in stripped]
                stopfree=[w for w in lem if not w in english_stop_words or w in ngram_words]
                tokens=[w for w in stopfree if w.isalpha()]
                tokens = [word for word in tokens if len(word) > 1]
                sent=" ".join(tokens)
                lot.append(sent)

            abcd =dict()
            fo = open('vocab.txt', 'r')
            lines = fo.readlines()
            i=0
            for line in lines:
                abcd[line.replace('\n','')]=i
                i+=1
            print(abcd)

            
            vectorizer = TfidfVectorizer(vocabulary=abcd, ngram_range=(1,4))
            X = vectorizer.fit_transform(lot)
            pred1 = clf1.predict(X)
            pred2=np.mean(pred1)
            print(predf)
            # render final page\
            if np.mean(predf)>.5:
                return render(request,'result.html',{'res':'You are recommended to buy the product'})
            else:
                return render(request,'result.html',{'res':'Its recommended not to buy the product'})
    else:
        form = NameForm()

    return render(request, 'index.html', {'form': form})