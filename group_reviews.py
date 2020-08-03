#
# Grouping reviews
#
# This is an example about how to group text data using
# hierarchical agglomerative clustering.
#
# Requirements: nltk, numpy, scipy, sklearn

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.cluster.hierarchy as hac

# Download these if you have not done so:
# nltk.download('stopwords') 
# nltk.download('punkt')

# Define the documents to analyse.
documents = ["Definitely recommend their breakfast.",
    "Good staff and pizza.",
    "No signal in the terrace.",
    "Had a delicious English breakfast.",
    "Not enough seating in the terrace." ]

# Define a tokenizing routine that does some preprocessing.
def preprocess (text):
    # Tokenize
    text = nltk.word_tokenize(text)
    # Perform simple normalization (case) and remove punctuation and stop words
    stopwords = nltk.corpus.stopwords.words("english")
    text = [w.lower() for w in text if w.isalpha() and w not in stopwords]
    # Perform stemming
    stemmer = nltk.stem.porter.PorterStemmer()
    text = [stemmer.stem(w) for w in text]
    
    return text

# Convert the documents into vectors by using term frequency.
tf = TfidfVectorizer(
    tokenizer=preprocess,use_idf=False)
X = tf.fit_transform(documents)
vocabulary = tf.get_feature_names()

# Perform agglomerative clustering on the word vectors.
Z = hac.linkage(X.toarray(), method='average')
distances = Z[:,2]
# Use an heuristic to find a good number of clusters:
# stop clustering when the gap between subsequent combination
# distances is the largest.
nclusters = 2 + np.argmax(np.diff(distances))
clusterIdx = hac.fcluster(Z,nclusters,criterion='maxclust')

# Find the centroid of each cluster.
clusters = np.unique(clusterIdx)
centroids = np.vstack(
    [np.mean(X[clusterIdx == cluster,:],0) 
    for cluster in clusters])
# Define labels for each cluster by taking the most
# important term in each centroid.
order = np.argsort(centroids)
labels = np.asarray(vocabulary)[order[:,-1]]

# Pair each document with its label.
table = np.hstack((
    np.asarray(documents).reshape(-1,1),
    labels[clusterIdx-1]))
# Order by label.
table = table[np.argsort(clusterIdx)]

print(table)
