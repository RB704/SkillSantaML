from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
from nltk.corpus import wordnet     # constants wordnet.NOUN, wordnet.VERB

from nltk import pos_tag

from nltk import PorterStemmer
from nltk import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer

import spacy


# Document Corpus

dc = ['The quick brown fox jumped over the lazy dog!',
      'I hope you are enjoying this class!',
      'An apple fell on Rishabh\'s head.',
      'This is quite an enjoyable course, hopefully it is also useful.',
      'The striped bats are hanging on their feet for best and ate best fishes.',
      'Following mice attacks, caring farmers were marching to Delhi for better living conditions.']
# Tokenization

tokens = []
for doc in dc:
    tokens.append(word_tokenize(doc))

tokens_all = tokens.copy()

# Stop-words removal

stop_words = stopwords.words('english') + list(string.punctuation)
for ti in range(len(tokens)):
    tokens[ti] = [i for i in tokens[ti] if i.lower() not in stop_words]

# Stemming, POS and Lemmatization

porterStemmer = PorterStemmer()
sbStemmer = SnowballStemmer('english')
wnLemmatizer = WordNetLemmatizer()
spacyProcessor = spacy.load('en', disable=['parser', 'ner'])

porter_stemmed_tokens = []
sb_stemmed_tokens = []
wn_lemmatized_tokens = []
sp_lemmatized_tokens = []
tokens_pos = []
row = 0

tag_dict = {
    "J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "R": wordnet.ADV
}

for doc_tokens in tokens:

    ps = []
    ss = []
    wl = []

    tokens_pos.append(pos_tag(doc_tokens))

    sp_doc = spacyProcessor(" ".join([token for token in doc_tokens]))
    sp_lemmatized_tokens.append([token.lemma_ for token in sp_doc])

    for i in range(len(doc_tokens)):

        ps.append(porterStemmer.stem(doc_tokens[i]))
        ss.append(sbStemmer.stem(doc_tokens[i]))
        
        wl.append(wnLemmatizer.lemmatize(doc_tokens[i], tag_dict.get(tokens_pos[row][i][1][0], wordnet.NOUN)))

    porter_stemmed_tokens.append(ps)
    sb_stemmed_tokens.append(ss)

    wn_lemmatized_tokens.append(wl)

    row = row + 1

print(tokens_pos)

print('\nAll tokens:\n\t', tokens_all)
print('Stopwords removed tokens:\n\t', tokens)
print('\nPorter Stemmed tokens:\n\t', porter_stemmed_tokens)
print('Snowball Stemmed tokens:\n\t', sb_stemmed_tokens)
print('\nWordnet Lemmatized tokens:\n\t', wn_lemmatized_tokens)
print('Spacy Lemmatized tokens:\n\t', sp_lemmatized_tokens)


# Create a new document corpus with the filtered and stemmed / lemmatized tokens
# We'll select Spacy's lemmatized tokens as it seems to be better

ndc = []
for doc_tokens in sp_lemmatized_tokens:
    ndc.append(" ".join([tokens for tokens in doc_tokens]))

# Text Vectorization (Featurization) using Counts and TF-IDF

count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

count_vectorizer.fit(ndc)
tfidf_vectorizer.fit(ndc)

print(count_vectorizer.vocabulary_)
print(tfidf_vectorizer.vocabulary_)
print(tfidf_vectorizer.idf_)

count_vector = count_vectorizer.transform(ndc)
tfidf_vector = tfidf_vectorizer.transform(ndc)

print(count_vector.shape)
print(count_vector.toarray())
print(tfidf_vector.shape)
print(tfidf_vector.toarray())


# Create a truth vector for classification training

doc_class = {
    'nursery': 0,
    'junior': 1,
    'senior': 2
}

dc_truth = ['nursery', 'junior', 'junior', 'junior', 'senior', 'senior']

