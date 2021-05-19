from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


def flatten(items, seqtypes=(list, tuple)):
    for i, x in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i + 1] = items[i]
    return items


def get_most_common_words(texts):
    vec = CountVectorizer(
        stop_words=None, vocabulary=stopwords.words('english')).fit(texts)
    bag_of_words = vec.transform(texts)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return [k for k, v in words_freq[:10]]
