from os import listdir
from os.path import isfile, join

import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.externals import joblib


################################################################
# Read training raw data
################################################################

pos_files_path = '/home/ngkpg/Downloads/aclImdb/train/pos/'
pos_files_list = [f for f in listdir(pos_files_path) if isfile(join(pos_files_path, f))]

neg_files_path = '/home/ngkpg/Downloads/aclImdb/train/neg/'
neg_files_list = [f for f in listdir(neg_files_path) if isfile(join(neg_files_path, f))]

# print(pos_files_list)
# print(neg_files_list)

positive_reviews = []
negative_reviews = []

for file_name in pos_files_list:   # reading every file from positive reviews folder
    with open(pos_files_path+file_name, 'r') as file:
        positive_reviews.append(file.read().replace('\n', ''))

for file_name in neg_files_list:   # reading every file from negative reviews folder
    with open(neg_files_path+file_name, 'r') as file:
        negative_reviews.append(file.read().replace('\n', ''))

# print(positive_reviews[:2])
# print(negative_reviews[:2])


################################################################
# Tokenization, Stop-words Removal and Lemmatization
################################################################

stop_words = stopwords.words('english') + list(string.punctuation)

pos_tokens = []
for doc in positive_reviews:
    pos_tokens.append(word_tokenize(doc))

neg_tokens = []
for doc in negative_reviews:
    neg_tokens.append(word_tokenize(doc))

for ti in range(len(pos_tokens)):
    pos_tokens[ti] = [i for i in pos_tokens[ti] if i.lower() not in stop_words]

for ti in range(len(neg_tokens)):
    neg_tokens[ti] = [i for i in neg_tokens[ti] if i.lower() not in stop_words]

spacyProcessor = spacy.load('en', disable=['parser', 'ner'])

pos_lemmatized_tokens = []
for doc_tokens in pos_tokens:
    sp_doc = spacyProcessor(" ".join([token for token in doc_tokens]))
    pos_lemmatized_tokens.append([token.lemma_ for token in sp_doc])

neg_lemmatized_tokens = []
for doc_tokens in neg_tokens:
    sp_doc = spacyProcessor(" ".join([token for token in doc_tokens]))
    neg_lemmatized_tokens.append([token.lemma_ for token in sp_doc])

preprocessed_positive_reviews = []
for doc_tokens in pos_lemmatized_tokens:
    preprocessed_positive_reviews.append(" ".join([tokens for tokens in doc_tokens]))

preprocessed_negative_reviews = []
for doc_tokens in neg_lemmatized_tokens:
    preprocessed_negative_reviews.append(" ".join([tokens for tokens in doc_tokens]))

# print(preprocessed_positive_reviews[:2])
# print(preprocessed_negative_reviews[:2])


################################################################
# Featurization
################################################################

pos_review_labels = [1] * len(preprocessed_positive_reviews)
neg_review_labels = [0] * len(preprocessed_negative_reviews)
review_labels = pos_review_labels + neg_review_labels

preprocessed_reviews = preprocessed_positive_reviews + preprocessed_negative_reviews

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(preprocessed_reviews)

print(len(tfidf_vectorizer.vocabulary_))

joblib.dump(tfidf_vectorizer, 'mini_project_tfidf_vectorizer.pkl')

review_features = tfidf_vectorizer.transform(preprocessed_reviews)


################################################################
# Training
################################################################

X_train, X_test, y_train, y_test = train_test_split(review_features, review_labels, test_size=0.33, random_state=42)

clf_nb = GaussianNB()
clf_lr = LogisticRegression(random_state=42)
clf_sv = SVC(random_state=42)

clf_nb.fit(X_train.toarray(), y_train)
clf_lr.fit(X_train.toarray(), y_train)
clf_sv.fit(X_train.toarray(), y_train)

joblib.dump(clf_nb, 'mini_project_clf_nb.pkl')
joblib.dump(clf_lr, 'mini_project_clf_lr.pkl')
joblib.dump(clf_sv, 'mini_project_clf_sv.pkl')


################################################################
# Testing
################################################################

clf_nb_results = clf_nb.predict(X_test.toarray())
clf_lr_results = clf_lr.predict(X_test.toarray())
clf_sv_results = clf_sv.predict(X_test.toarray())

clf_nb_acc = accuracy_score(y_test, clf_nb_results)
clf_lr_acc = accuracy_score(y_test, clf_lr_results)
clf_sv_acc = accuracy_score(y_test, clf_sv_results)

print('\n\nTest results:')
print('\tNB: ', clf_nb_acc)     # 0.6704242424242425   =>  67.04%
print('\tLR: ', clf_lr_acc)     # 0.8876363636363637   =>  88.76%
print('\tSV: ', clf_sv_acc)     # 0.49903030303030305  =>  49.90%

