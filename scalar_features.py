'''
All non vectorizer features should be functions of the following template:

def feature(text):
    return scalar

The text parameter is a single string (text).
The return value is a single scalar for that text.
'''

from collections import Counter
from itertools import groupby
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
import syllables
import string
from string import punctuation
import re
import regex


def get_words(entry):
    return filter(lambda w: len(w) > 0,
                  [w.strip("0123456789!:,.?(){}[]") for w in entry.split()])


def richness(text):
    dict1 = {}
    stemmer1 = PorterStemmer()
    for w in get_words(text):
        w = stemmer1.stem(w).lower()
        try:
            dict1[w] += 1
        except KeyError:
            dict1[w] = 1

    return len(dict1) / sum(dict1.values())


def yule(entry):
    # yule's I measure (the inverse of yule's K measure)
    # higher number is higher diversity - richer vocabulary
    d = {}
    stemmer = PorterStemmer()
    for w in get_words(entry):
        w = stemmer.stem(w).lower()
        try:
            d[w] += 1
        except KeyError:
            d[w] = 1
    M1 = float(len(d))
    M2 = sum([len(list(g)) * (freq**2)
              for freq, g in groupby(sorted(d.values()))])

    try:
        return (M1 * M1) / (M2 - M1)

    except ZeroDivisionError:
        return 0


# number of sentences per text
def nr_sents(text):
    sentences = sent_tokenize(text)
    return len(sentences)


# average length of a sentence
def avg_words(text):
    sentences = sent_tokenize(text)

    n_words = 0
    for sentence in sentences:
        for word in sentence.split():
            n_words = n_words + 1

    average = n_words / (len(sentences))
    return average

# average number of characters per word


def avg_chars(text):
    sentences = sent_tokenize(text)

    n_words = 0
    n_chars = 0
    for sentence in sentences:
        for word in sentence.split():
            word = word.translate(str.maketrans('', '', string.punctuation))
            n_words = n_words + 1
            n_chars = n_chars + len(word)

    average = n_chars / n_words
    return average


def fleschReadingEaseScore(text):
    sentences = sent_tokenize(text)
    n_words = 0
    for sentence in sentences:
        for word in sentence.split():
            n_words = n_words + 1

    sentences = nr_sents(text)

    nr_syllables = 0
    words = word_tokenize(text)

    for word in words:
        nr_syllables = nr_syllables + syllables.estimate(word)

    return 206.835 - 1.015 * (n_words / sentences) - 84.6 * (nr_syllables / n_words)


def num_emo(text):
    return len(regex.findall(r':' + "\S", text)) / len(text)


def get_tokens(entry):  # get_words and nltk.word_tokenize exculde !/?
    return filter(lambda w: len(w) > 0,
                  [w.strip("0123456789:,.(){}[]") for w in entry.split()])


def three_cons_chars(text):
    count = 0
    for word in get_tokens(text):
        for i in range(0, len(word) - 2):
            if (word[i]) == (word[i + 1]) == (word[i + 2]) != ".":
                count += 1
    return count / len(text)


def punctuation_freq(text):
    str2 = "".join(m for m in text if m in punctuation)
    return len(str2) / len(nltk.word_tokenize(text))


def stopword_freq(text):
    str2 = text.split()
    counts = Counter(str2)
    stopwordz = stopwords.words('english')
    freqs = {k: v for k, v in counts.items() if k in stopwordz}
    # print(freqs)
    return (sum(freqs.values())) / len(nltk.word_tokenize(text))


def stopword_freq_new(text, most_common):
    str2 = text.split()
    counts = Counter(str2)
    stopwordz = most_common
    freqs = {k: v for k, v in counts.items() if k in stopwordz}
    return (sum(freqs.values())) / len(nltk.word_tokenize(text))


def cons_caps(text):
    cons_caps = re.findall(r"(\b(?:[A-Z]+[A-Z]+)\b)", text)
    return len(cons_caps) / len(nltk.word_tokenize(text))


def curses(text):
    file = open('facebook-bad-words.txt')
    curses = file.read()
    curses = curses.replace(' ', '')
    curse_list = curses.split(',')
    tokens = wordpunct_tokenize(text)
    cursecount = 0

    for token in tokens:
        token = token.translate(str.maketrans('', '', string.punctuation))
        if token in curse_list:
            cursecount = cursecount + 1
        else:
            pass
    return cursecount / len(tokens)


def get_all_features():
    scalar_feature_functions = [
        richness,
        avg_words,
        fleschReadingEaseScore,
    ]

    return scalar_feature_functions


def get_all_feature_names():
    return [
        'richness',
        'avg_words',
        'fleschReadingEaseScore',
    ]
