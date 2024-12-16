# Import Libraries
import pandas as pd
import os
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')

# Import Datasets - https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset
dataset = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/../datasets/emails.csv')


# Clean the texts
def clean_text(text):
    # Remove "Subject:" or any other unwanted patterns
    text = re.sub(r'Subject:', '', text)

    # Remove numbers and punctuation
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Stemmer
    ps = PorterStemmer()

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [ps.stem(word) for word in words if word.lower() not in stop_words]

    # Rejoin words into a single string
    text = ' '.join(words)
    return text


corpus = dataset['text'].apply(clean_text)

# Bag of words model
cv = CountVectorizer(max_features=25000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Naive Bayes model on the training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
