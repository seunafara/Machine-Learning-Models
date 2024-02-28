import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/../Datasets/india_2018-2010_import.csv')

# Clean the texts
def clean_text(text):

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


encoder = OneHotEncoder(handle_unknown='ignore')

corpus = dataset['Commodity'].apply(clean_text)
cv = CountVectorizer(max_features=25000)
first_frame = cv.fit_transform(corpus).toarray()

# print(first_frame)

country_year_encoded = pd.DataFrame(dataset, columns=['country', 'year'])
encoder.fit(country_year_encoded)
second_frame = encoder.transform(country_year_encoded).toarray()

# print(second_frame)

X = np.concatenate((first_frame, second_frame), axis=1)

df = pd.DataFrame(dataset, columns=['value'])
y = df.to_numpy()
y = df.values.ravel()
y = np.where(np.isnan(y), 0, y)

# Assuming X is your feature matrix and y is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the SVR model on the training set
# regressor = SVR(kernel='rbf')
# regressor.fit(X_train, y_train)

# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print(r2_score(y_test, y_pred))
