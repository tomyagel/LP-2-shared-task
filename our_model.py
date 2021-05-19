import argparse
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import scalar_features
import vectorizers
import os
import training
import pickle
import nltk


def transform(pair, vecs, scalars):
    X = []

    # Iterate by columns
    # Vectorizer features
    for vec_fn in vecs:
        v = vec_fn.transform(pair)
        X.append(1 - cosine_similarity(
            v[0].reshape(1, -1), v[1].reshape(1, -1)))

    # Scalar features
    for feature_fn in scalars:
        f1 = feature_fn(pair[0])
        f2 = feature_fn(pair[1])
        X.append(np.abs(f1 - f2))

    return np.array(X, dtype=object)


def main():
    parser = argparse.ArgumentParser(
        description='Application for training or test a PAN20 model')

    # data settings:
    parser.add_argument('-i', type=str, required=True,
                        help='Path to evaluation directory where pairs.jsonl is found')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to output directory where answers.jsonl will be written')
    parser.add_argument('-t', type=str,
                        help='Path to directory containing jsonl-files with the input pairs and truth for training')
    parser.add_argument('-cache', type=str,
                        help='Path to directory for caching transformed data, creates and loads cache only if set')

    args = parser.parse_args()

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    train_model = args.t is not None

    # Model
    if train_model:
        print('Training model')
        input_truth = args.t + os.sep + 'truth.jsonl'
        input_pairs = args.t + os.sep + 'pairs.jsonl'
        model, vecs, scalars = training.train_model(
            input_truth, input_pairs, args.cache)
        with open('model' + os.sep + 'model.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Loading model')
        with open('model' + os.sep + 'model.pickle', 'rb') as handle:
            model = pickle.load(handle)
        with open('model' + os.sep + 'most_common_words.pickle', 'rb') as handle:
            most_common_words = pickle.load(handle)
        vecs = vectorizers.get_vectorizers(False, most_common_words)
        scalars = scalar_features.get_all_features(most_common_words)

    # Evaluate
    test_pairs = args.i + os.sep + 'pairs.jsonl'

    print('Calculating test similarities')
    with open(args.o + os.sep + 'answers.jsonl', 'w') as outp:
        with open(test_pairs) as inp:
            for line in tqdm(inp):
                d = json.loads(line.strip())
                problem_id = d['id']
                X = transform(d['pair'], vecs, scalars)
                similarity = model.predict(X.reshape(1, -1))
                r = {'id': problem_id, 'value': float(similarity[0])}
                outp.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
