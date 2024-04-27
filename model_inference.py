# the model inference file will take the local-saved model and use it to predict the sentiment of the movie review
# the movie reviews will be took from rotten tomatoes website and the default movie name is dune part two 
# the output will be a csv file containing the latest movie reviews and the sentiment prediction

# library and packages 
import os
import time
import pickle
import requests
import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from keras.models import Sequential
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# declare the functions that will be used 
def get_rotten_tomatoes_reviews(movie_title= "dune_part_two"):
    ''' 
        function    : get the movie reviews from rotten tomatoes website
        input       : movie title (default is dune part two)
        output      : list of reviews
    '''
    url = f"https://www.rottentomatoes.com/m/{movie_title}/reviews"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    reviews = soup.find_all("p", class_="review-text")
    reviews_text = [review.text for review in reviews]
    return reviews_text 


def preprocess_review(review_text):
    '''
        function    : preprocess and clean the sentiment data by removing special characters, stopwords, and tokenize the text
        input       : raw text 
        output      : clean text without special characters, stopwords, and lowercased
    '''
    review_text = review_text.translate(str.maketrans('', '', '-!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'))
    review_text = review_text.lower()
    review_text = word_tokenize(review_text)
    review_text = [word for word in review_text if word not in stop_words]
    review_text = " ".join(review_text)
    return review_text

time1 = time.time()

# load the tokenizer model
with open('tokenizer.pickle', 'rb') as handle:
    word_index = pickle.load(handle)

# create a new tokenizer instance and load the vocabulary
tokenizer   = Tokenizer()
tokenizer.word_index    = word_index
tokenizer.num_words     = len(word_index)

# load model 
model       = tf.keras.models.load_model('model_lstm.keras')

# preprocess the input data (movie reviews) to be used in the model
movie_title = input("Movie Title: ")
reviews     = get_rotten_tomatoes_reviews(movie_title)
reviews     = [preprocess_review(review) for review in reviews]
X_test      = tokenizer.texts_to_sequences(reviews)
X_test      = pad_sequences(X_test, maxlen=100)

# predict the news sentiment using the model
y_pred      = model.predict(X_test)
y_pred      = pd.DataFrame(y_pred)
y_pred      = y_pred.idxmax(axis=1)

# save the prediction to csv file
result_df   = pd.DataFrame({"reviews": reviews, "sentiment": y_pred})
result_df.to_csv("movie_review_sentiment.csv", index=False)

print(f'Number of Reviews: {len(reviews)}')
print(f'Inference Time: {time.time() - time1}')