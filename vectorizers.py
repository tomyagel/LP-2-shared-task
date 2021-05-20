'''
All vectorizer features should be functions of the following template:

def transform(texts):
    return matrix

For most uses a vectorizer.fit will suffice, however, not for PoS vectorizer,
because it doesn't transform the text directly, first it has to get PoS tags.
The custom vectorizer features have to accept a list of strings (texts),
not a singular text.
'''

from collections import Counter
import nltk
from nltk.corpus import stopwords
import numpy as np
import os.path
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
from string import punctuation
from tqdm import tqdm
from utils import flatten


def get_pos_tags(texts):
    pos = []
    # for text in tqdm(texts):
    for text in texts:
        sents = nltk.sent_tokenize(text)
        pos_sent = ''
        for sent in sents:
            pos_sent += ' '.join([p for word,
                                  p in nltk.pos_tag(nltk.word_tokenize(sent))])
        pos.append(pos_sent)
    return pos


def walk_pos_tree(tree):
    # Terminal node
    if isinstance(tree, tuple):
        return None
    vals = [walk_pos_tree(child) for child in tree]
    if all([v is None for v in vals]):
        return tree.label()
    if None in vals:
        new_vals = []
        for i in range(len(tree)):
            if vals[i] is None:
                if isinstance(tree[i], tuple):
                    new_vals.append(tree[i][1])
                else:
                    new_vals.append(tree[i].label())
            else:
                new_vals.append(vals[i])
        vals = new_vals
    return vals


def walk_pos_tree_flat(tree):
    return flatten(walk_pos_tree(tree))


def bigrams_per_line(text):
    for ln in text.split('.'):
        terms = re.findall(r'\w{1,}', ln)
        for bigram in zip(terms, terms[1:]):
            yield '%s %s' % bigram


grammar = r"""
  NP:
      {<DT|WDT|PP\$|PRP\$>?<\#|CD>*(<JJ|JJS|JJR><VBG|VBN>?)*(<NN|NNS|NNP|NNPS>(<''><POS>)?)+}
      {<DT|WDT|PP\$|PRP\$><JJ|JJS|JJR>*<CD>}
      {<DT|WDT|PP\$|PRP\$>?<CD>?(<JJ|JJS|JJR><VBG>?)}
      {<DT>?<PRP|PRP\$>}
      {<WP|WP\$>}
      {<DT|WDT>}
      {<JJR>}
      {<EX>}
      {<CD>+}
  VP: {<VBZ><VBG>}
      {(<MD|TO|RB.*|VB|VBD|VBN|VBP|VBZ>)+}
    """

Reg_parser = nltk.RegexpParser(grammar)


class PosVectorizer():
    def __init__(self, vec_type, n=1):
        self.vectorizer = vec_type(ngram_range=(n, n))

    def transform(self, texts):
        return self.vectorizer.transform(get_pos_tags(texts))


class PosSubtreeVectorizer():
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer=bigrams_per_line, ngram_range=(2, 2))

    def fit(self, texts):
        corpus = self.preprocess(texts)
        self.vectorizer.fit(corpus)

    def transform(self, texts):
        corpus = self.preprocess(texts)
        return self.vectorizer.transform(corpus)

    def preprocess(self, texts):
        corpus = []
        for text in texts:
            sentences = nltk.sent_tokenize(text)
            sent_tokens = []
            for sent in sentences:
                pos_tags = nltk.pos_tag(nltk.word_tokenize(sent))
                pos_tags_without_punct = [
                    tag for tag in pos_tags if tag[0] not in string.punctuation]
                if len(pos_tags_without_punct) > 1:
                    reg_tags = Reg_parser.parse(pos_tags_without_punct)
                    pos_tree = walk_pos_tree_flat(reg_tags)
                    tokens = ' '.join(pos_tree)
                    if len(pos_tree) > 1:
                        sent_tokens.append(tokens)
            corpus.append('.'.join(sent_tokens))
        return corpus


class PunctuationFreqVectorizer():
    def __init__(self):
        pass

    def fit(self, texts):
        pass

    def transform(self, texts):
        m = []
        for text in texts:
            str_punc = "".join(m for m in text if m in punctuation)
            tokens = nltk.word_tokenize(text)
            n = len(tokens) + len(set(tokens))
            counts = Counter(str_punc)
            v = np.array([((counts[punc] + 1) / n) for punc in punctuation])
            m.append(v)
        return np.vstack(m)


def finegrainstopword(texts, most_common_words):
    m = []
    for text in texts:
        text.lower()
        words = text.split()
        counts = Counter(words)
        stopwordz = stopwords.words('english')
        f = {k: v for k, v in counts.items() if k in stopwordz}
        tokes = len(nltk.word_tokenize(text))
        v = np.array([f.get(term, 0) / tokes for term in most_common_words])
        m.append(v)
    return np.vstack(m)


class PairIter:
    def __init__(self, ls, start=0):
        self.max = len(ls)
        self.ls = ls
        self.num = start
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self):
        num = self.num
        p = self.p
        if num == self.max and p == 0:
            raise StopIteration
        self.num += self.p == 1
        self.p = (self.p + 1) % 2
        return self.ls[num][p]


def get_vectorizers(train_model, most_common_words, text_pairs=None):
    if train_model and text_pairs is None:
        raise Exception('Text pairs are required for training')

    nltk.download('stopwords')

    # Part of speech tags
    all_pos = None

    if train_model:
        path = 'model' + os.sep + 'pos.pickle'
        if os.path.isfile(path):
            print('Loading pos tags')
            with open(path, 'rb') as handle:
                all_pos = pickle.load(handle)
        else:
            print('Extracting pos tags')
            all_pos = get_pos_tags(PairIter(text_pairs))
            with open(path, 'wb') as handle:
                pickle.dump(all_pos, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def finegrainstopword_most_common(texts):
        return finegrainstopword(most_common_words)

    # Train/load vectorizers
    print('Constructing vectorizers')

    # vec_word_ngram = CountVectorizer(ngram_range=(2, 2), max_features=2000)
    vec_word_tfidf = TfidfVectorizer(ngram_range=(2, 2), max_features=2000)
    vec_char_ngram = CountVectorizer(
        analyzer='char', ngram_range=(3, 3), max_features=2000)
    vec_char_tfidf = TfidfVectorizer(
        analyzer='char', ngram_range=(3, 3), max_features=2000)
    # vec_pos_count = PosVectorizer(CountVectorizer)
    vec_pos_tfidf = PosVectorizer(TfidfVectorizer)
    vec_pos_tfidf_bigrams = PosVectorizer(TfidfVectorizer, n=2)
    vec_pos_subtree = PosSubtreeVectorizer()
    vec_punc_freq = PunctuationFreqVectorizer()
    vec_stopwords = TfidfVectorizer(vocabulary=stopwords.words('english'))

    def fit_pos_vec(pos_vec, _):
        pos_vec.vectorizer.fit(tqdm(all_pos))

    PosVectorizer.fit = fit_pos_vec

    vectorizer_function_names = get_all_vectorizer_names()
    vectorizers = [
        vec_word_tfidf,
        vec_char_ngram,
        vec_char_tfidf,
        vec_pos_tfidf,
        vec_pos_tfidf_bigrams,
        vec_pos_subtree,
        vec_punc_freq,
        vec_stopwords,
    ]

    for i in range(len(vectorizers)):
        print('{}/{}'.format(i + 1, len(vectorizers)))
        path = 'model/{}.pickle'.format(vectorizer_function_names[i])

        # if train_model:
        if not os.path.isfile(path):
            vectorizers[i].fit(
                tqdm(PairIter(text_pairs), total=len(text_pairs) * 2))
            with open(path, 'wb') as handle:
                pickle.dump(vectorizers[i], handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        # Loads the model
        else:
            if not os.path.isfile(path):
                raise Exception('{} is not found'.format(path))
            with open(path, 'rb') as handle:
                vectorizers[i] = pickle.load(handle)

    return vectorizers


def get_all_vectorizer_names():
    return [
        'vec_word_tfidf',
        'vec_char_ngram',
        'vec_char_tfidf',
        'vec_pos_tfidf',
        'vec_pos_tfidf_bigrams',
        'vec_pos_subtree',
        'vec_punc_freq',
        'vec_stopwords',
    ]
