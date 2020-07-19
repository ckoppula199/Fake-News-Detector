# Fake-News-Detector  
  
This program is capable of detecting whether or not a news article is fake or not.  
  
Using sklearn I first build a TfidfVectorizer using the dataset and then I use a PassiveAggressiveClassifier to fit the model.  
tfidf is made up of 2 terms:  
  
tf stands for term frequencey and is how often a particular word appears in a document. A higher term frequency means that term appears more often than others and so the document would be a good match when the term is part of the search terms.  
  
idf stands for inverse document frequency and is a measure of how significant a word is in the entire corpus. It is the total number of documents divided by the number of documents that contain the term (documents where the term frequency != 0).  
  
tfidf = tf x idf  
  
The Passive Aggressive Classifier is an online learning algorithm. If it makes a correct classification then the weights are not updated but if it makes an incorrect classification then the weights are updated. Unlike other learning algorithms, it does not converge and instead makes updates to the weights that correct losses.  
  
Finally, a confusion matrix is generated and precision, recall and accuracy were then worked out.
