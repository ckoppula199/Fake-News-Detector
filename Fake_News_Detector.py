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

# Collection of raw documents converted into matrix of TF-IDF features.
# Initialised to exclude stop words (and, the, to etc) and words with a greater
# document frequency than 0.7
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Apply TfidfVectorizer to training and test set.
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Convert to numpy arrays
tfidf_train = tfidf_train.toarray()
tfidf_test = tfidf_test.toarray()


"""
---------------Training PassiveAggressiveClassifier---------------
"""

# Training the classifier
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfidf_train, y_train)


"""
---------------Predicting Single Result---------------
"""

# Input has to be a 2D array
y_pred_single = classifier.predict([tfidf_test[0]])
print("Predicting Single Result\n")
# Below line prints the entire article
# print(f"Input is {X_test[0]}")
print(f"Actual label is {y_test[0]}")
print(f"Predicted label is {y_pred_single}\n\n")


"""
---------------Predicitng Test Set Result---------------
"""

y_pred = classifier.predict(tfidf_test)
print("Prediction, Actual")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1), end='\n\n')

"""
--------------Making the Confusion Matrix and Evaluating the Model---------------
"""

# Gives confusion matrix C such that Cij is equal to the number of observations known to be in group i and predicted to be in group j.
cm = confusion_matrix(y_test, y_pred)
true_positive = cm[1][1]
true_negative = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]

print("\nModel Evaluation")
print(f"Accuracy is {accuracy_score(y_test, y_pred)}")
print(f"Precision is {true_positive / (true_positive + false_positive)}")
print(f"Recall is {true_positive / (true_positive + false_negative)}")
print(f"F1 score is {f1_score(y_test, y_pred, labels=['REAL', 'FAKE'], pos_label='REAL')}")
