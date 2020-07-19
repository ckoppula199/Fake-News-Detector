"""
This is a program that uses TfidfVectorizer and a PassiveAggressiveClassifier to
detect fake news articles.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

"""
---------------Data Pre-processing---------------
"""

dataset = pd.read_csv("Data/news.csv")
print(dataset.head(), end="\n\n") # view first 5 rows of data to see its structure.

# List of all the news articles
news_text = dataset.text.values
# y is the vector of labels
y = dataset.loc[:, 'label'].values

# Find out how many of each label type there is
print(f"There are {y[y == 'FAKE'].shape[0]} FAKE labels")
print(f"There are {y[y == 'REAL'].shape[0]} REAL labels", end = "\n\n")

X_train, X_test, y_train, y_test = train_test_split(news_text, y, test_size=0.2, random_state=42)


"""
---------------Using TfidfVectorizer---------------
"""

# Collection of raw documents converted into matric of TF-IDF features.
# Initialised to exclude stop words (and, the, to etc) and words with a greater
# document frequency than 0.7
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Apply TfidfVectorizer to training and test set.
tfidf_vectorizer.fit_transform(X_train)
tfidf_vectorizer.transform(X_test)
