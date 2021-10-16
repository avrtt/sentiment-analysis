from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from joblib import dump, load
import random

nltk.download('movie_reviews')

texts = [' '.join(nltk.corpus.movie_reviews.words(fileids=[f])) for f in nltk.corpus.movie_reviews.fileids()]
labels = [0 if nltk.corpus.movie_reviews.categories(fileids=[f])==['neg'] else 1 for f in nltk.corpus.movie_reviews.fileids()]

X = []
y = []
indices = list(range(len(texts)))
random.shuffle(indices)
for i in indices:
  X.append(texts[i])
  y.append(labels[i])

vectorizer = TfidfVectorizer()
clf = LogisticRegression()
X = vectorizer.fit_transform(X)
clf.fit(X, y)

dump(vectorizer, 'vectorizer.joblib')
dump(clf, 'clf.joblib')

