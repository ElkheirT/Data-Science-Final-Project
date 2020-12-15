import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn import svm
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from SpellCheck import get_correction
import json

with open('./english_contractions.json') as f:
    contractions = json.load(f)

amazon_data = pd.read_csv('data/amazon_labelled.txt', sep="\t")

tokenizer = RegexpTokenizer(r"[a-zA-Z']+")
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] in contractions.keys():
            tokens[i] = contractions[tokens[i]]
        else:
            tokens[i] = get_correction(tokens[i])
        tokens[i] = lemmatizer.lemmatize(tokens[i], pos='v')
    return " ".join(tokens)

X = amazon_data.iloc[:, 0]
X = X.apply(preprocess_text)
y = amazon_data.iloc[:, 1]


model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_df=.7), svm.LinearSVC())
scores = cross_val_score(model, X, y, cv=5)
print(np.mean(scores))
