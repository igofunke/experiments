#
# Analyse postings with entropy
#
# This is an example on how to use entropy on a job postings dataset
# to pinpoint the most relevant terms and choose preferences on the fly.
#
# Requirements: 
#   nltk, numpy, scipy, sklearn
#
#   Additionally download the nltk resources punkt and stopwords:
#   >>> import nltk
#   >>> nltk.download('punkt')
#   >>> nltk.download('stopwords')
#  
#   The dataset nyc-jobs.csv contains job postings from the official
#   website of the City of New York. You can download it from any of
#   the following locations:
#   (*) https://data.cityofnewyork.us/City-Government/NYC-Jobs/kpav-sd4t
#   (*) https://data.world/city-of-ny/kpav-sd4t


import csv
import random
import numpy as np
import nltk
import scipy
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from itertools import compress

with open('nyc-jobs.csv') as csvfile:
    jobs = list(csv.reader(csvfile))

# Some constants
jobTitleField = 4
jobDescritpionField = 15

# Retrieve a sample of the jobs
N = 1000
random.seed(0) # for reproducibility
jobs = random.sample(jobs,N)

def preprocess (job):
    # Only keep the relevant fields for the analysis, such as the job description.
    return job[jobDescritpionField]

# Define a tokenizing routine that does some preprocessing.
def tokenize (text):
    # Tokenize
    text = nltk.word_tokenize(text)
    # Perform simple normalization (case) and remove punctuation and stop words
    stopwords = nltk.corpus.stopwords.words("english")
    text = [w.lower() for w in text if w.isalpha() and w not in stopwords]
    
    return text

# Convert the documents into vectors by using term frequency.
tf = CountVectorizer(
    preprocessor=preprocess,
    tokenizer=tokenize,
    binary=True,
    ngram_range=(1,2))

X = tf.fit_transform(jobs).tolil()
vocabulary = tf.get_feature_names()

# Keep track of the preferences
yes = set()
no = set()
maybe = set()

ones = csr_matrix(np.ones(N))
selector = ones.copy() # Selects rows in X
remaining = N

while remaining > 0:
    # Calculate entropy
    p = (selector*X / N).todense()
    entropy = scipy.stats.entropy(np.vstack((p,1-p)))

    # Choose the feature with the highest entropy
    featureIdx = np.argmax(entropy)
    feature = vocabulary[featureIdx]

    option = input(" (N = " + str(remaining) + ") Does '" + feature + "' have to appear? "
        "[(y)es, (n)o, (m)aybe, (e)xit]: ")

    if option == "y":
        newSelector = X[:,featureIdx].transpose()
        selector = selector.multiply(newSelector)
        yes.add(feature)
    elif option == "n":
        newSelector = ones - X[:,featureIdx].transpose()
        selector = selector.multiply(newSelector)
        no.add(feature)
    elif option == "m":
        maybe.add(feature)
    elif option == "e":
        print("Exiting...")
        break
    else:
        print("Unrecognised option")
        continue

    # Disable the feature
    X[:,featureIdx] = 0
    remaining = selector.sum()

isSelected = (selector > 0).todense().flat
selectedJobs = compress(jobs,isSelected)

getTitleAndDesc = lambda job: job[jobTitleField] + " - " + job[jobDescritpionField][:100]
selectedJobs = map(getTitleAndDesc,selectedJobs)

print("[Remaining entries]")
print("\n".join(selectedJobs))

print()
print("[Preference of words]")
print("Yes: ", end="")
print(yes)
print("No: ", end="")
print(no)
print("Maybe: ",end="")
print(maybe)
