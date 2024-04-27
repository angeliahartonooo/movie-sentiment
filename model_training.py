# this file is used to train the movie review sentiment model 
# the dataset is imdb and rotten tomatoes review that contains the movie reviews and the sentiment (positive / negative)

# importing library and packages needed 
import os
import re
import nltk
import pickle
import itertools
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from keras.models import Sequential
from nltk.tokenize import word_tokenize
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words  = stopwords.words('english')



# declare the functions that will be used 
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

def train_model(X_train, y_train):
    ''' 
        function    : train the model using training samples and training labels 
        input       : training feature (X_train) and training labels (y_train)
        output      : trained model
    '''

    embed_dim = 256
    lstm_out = 256
    batch_size = 512
    max_features = 2000

    y_train = pd.get_dummies(y_train).values

    model = Sequential()
    model.add(Embedding(max_features, embed_dim,input_length = X_train.shape[1]))
    model.add(SpatialDropout1D(0.1))
    model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(2,activation='softmax'))
    opt = Adam(learning_rate=0.01)
    model.compile(loss = 'categorical_crossentropy', optimizer=opt,metrics = ['accuracy'])
    model.fit(X_train, y_train, epochs = 25, batch_size=batch_size, verbose = 1)
    return model

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
        function    : plots a confusion matrix with labels and normalization options.
        input       : confusion matrix, classes, normalize, title, cmap 
        output      : showing the confusion matrix plot and saving the confusion matrix to image
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'  # Display format for normalized values (e.g., "0.23")
    else:
        fmt = 'd'  # Display format for raw counts (e.g., "23")
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()


# read imdb and rotten tomatoes sentiment analysis dataset from local directory
imdb_dataset                    = pd.read_csv('imdb_dataset.csv')
rotten_tomatoes_dataset         = pd.read_csv('rotten_tomatoes_dataset.csv')[['reviewText', 'scoreSentiment']]
rotten_tomatoes_dataset.columns = ['review', 'sentiment']


# map the label positive to 1 and negative to 0 
# for the sake of simplicity, I just take 10000 samples from each dataset
imdb_dataset['sentiment']               = imdb_dataset['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
rotten_tomatoes_dataset['sentiment']    = rotten_tomatoes_dataset['sentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

dataset = pd.concat([imdb_dataset.iloc[:25000], rotten_tomatoes_dataset.iloc[:25000]], axis=0)
dataset = dataset.drop_duplicates(subset='review')
dataset = dataset.dropna(subset=['review', 'sentiment'])


# preprocess and clean the review data by removing special characters, stopwords, and tokenize the text
dataset['review'] = dataset['review'].apply(lambda x: str(preprocess_review(str(x))))


# split the dataset into train and test data 
train_df, test_df = train_test_split(dataset, test_size=0.25, random_state=42)

X_train = train_df['review'].values
y_train = train_df['sentiment'].values

X_test  = test_df['review'].values
y_test  = test_df['sentiment'].values


# limit the number of features and tokenize the words 
max_features    = 2000
keras_tokenizer = Tokenizer(num_words=max_features, split=' ')
keras_tokenizer.fit_on_texts(X_train)


# saving the keras tokenizer data to tokenize new text that wants to be classify
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(keras_tokenizer.word_index, handle)


# tokenize the test and train set to be used in the model
X_train     = keras_tokenizer.texts_to_sequences(X_train)
X_train     = pad_sequences(X_train, maxlen=100)
X_test      = keras_tokenizer.texts_to_sequences(X_test)
X_test      = pad_sequences(X_test, maxlen=100)


# train the model and save 
model_result = train_model(X_train, y_train)

# test to see the model accuracy and the classification report
y_pred  = model_result.predict(X_test)
y_pred  = pd.DataFrame(y_pred)
y_pred  = y_pred.idxmax(axis=1)
cm      = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, ['Negative', 'Positive'])
print(classification_report(y_test, y_pred))


# save the model
model_result.save('model_lstm.keras')