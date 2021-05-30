import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scalar_features
import vectorizers
from utils import get_most_common_words
import os.path
import pickle
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, accuracy_score


# Data transformation
def transform_data(vecs, scalars, text_pairs, cache_dir):
    N = len(text_pairs)
    K = len(vecs) + len(scalars)
    X = np.empty((N, K))

    vec_names = vectorizers.get_all_vectorizer_names()
    feat_names = scalar_features.get_all_feature_names()

    # Iterate by columns
    # Vectorizer features
    j = 0
    for vec_fn, name in zip(vecs, vec_names):
        print('{}/{}'.format(j + 1, K))
        loaded = False
        if cache_dir:
            path = cache_dir + os.sep + name + '.pickle'
            if os.path.isfile(path):
                with open(path, 'rb') as handle:
                    X[:, j] = pickle.load(handle)
                loaded = True
        if not loaded:
            for i in tqdm(range(N)):
                v = vec_fn.transform(text_pairs[i])
                X[i, j] = 1 - cosine_similarity(
                    v[0].reshape(1, -1), v[1].reshape(1, -1))
            if cache_dir:
                with open(path, 'wb') as handle:
                    pickle.dump(X[:, j], handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
        j += 1

    # Scalar features
    for feature_fn, name in zip(scalars, feat_names):
        print('{}/{}'.format(j + 1, K))
        loaded = False
        if cache_dir:
            path = cache_dir + os.sep + name + '.pickle'
            if os.path.isfile(path):
                with open(path, 'rb') as handle:
                    X[:, j] = pickle.load(handle)
                loaded = True
        if not loaded:
            for i in tqdm(range(N)):
                f1 = feature_fn(text_pairs[i][0])
                f2 = feature_fn(text_pairs[i][1])
                X[i, j] = np.abs(f1 - f2)
            if cache_dir:
                with open(path, 'wb') as handle:
                    pickle.dump(X[:, j], handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
        j += 1

    return X


class GmClassifier():
    def __init__(self, n_components=4, random_state=0):
        self.gm_neg = GaussianMixture(
            n_components=n_components, random_state=random_state)
        self.gm_pos = GaussianMixture(
            n_components=n_components, random_state=random_state)

    def fit(self, X, y):
        X_neg = X[y == 0]
        X_pos = X[y == 1]
        self.gm_neg.fit(X_neg)  # Different authors
        self.gm_pos.fit(X_pos)  # Same author

    def predict(self, X):
        y_neg = self.gm_neg.score_samples(X)
        y_pos = self.gm_pos.score_samples(X)
        pred = np.where(y_pos > y_neg, 1, 0)
        return pred

    def set_params(self, **params):
        self.gm_neg.set_params(**params)
        self.gm_pos.set_params(**params)


def params_LR():
    def creator():
        return LogisticRegression(max_iter=10000)

    params = []
    for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        params.append({'C': C})
    return creator, params


def params_SVM():
    def creator():
        return make_pipeline(StandardScaler(), SVC(
            gamma='auto', probability=True))

    params = []
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = ['scale', 0.001, 0.01, 0.1, 1]

    for C in Cs:
        for gamma in gammas:
            params.append({'svc__C': C, 'svc__gamma': gamma})
    return creator, params


def params_RF():
    def creator():
        return RandomForestClassifier(random_state=42)

    n_estimators = [50, 100, 150, 200]
    max_depth = [5, 10, 15, 20]

    params = []
    for n in n_estimators:
        for d in max_depth:
            params.append({'n_estimators': n, 'max_depth': d})
    return creator, params


def params_GMM():
    def creator():
        return make_pipeline(StandardScaler(), GmClassifier())

    params = []
    for n in range(4):
        params.append({'gmclassifier__n_components': (n + 1) * 4})
    return creator, params


def select_best_params(model_creator, name, params, X, y):
    print('Selecting best {}'.format(name))
    skf = StratifiedKFold(n_splits=5)

    avg_scores = []

    for param in tqdm(params):
        scores = []
        for train_index, test_index in skf.split(X, y):
            clf = model_creator()
            clf.set_params(**param)
            clf.fit(X[train_index], y[train_index])
            y_pred = clf.predict_proba(X[test_index])[:, 1]
            scores.append(mean_squared_error(y[test_index], y_pred))
        avg_scores.append(np.mean(scores))
        print('Average MSE {} with parameters {}'.format(
            avg_scores[-1], param))

    best_idx = np.argmin(avg_scores)
    print('Best average MSE {} with parameters {}'.format(
        avg_scores[best_idx], params[best_idx]))
    clf = model_creator()
    clf.set_params(**params[best_idx])
    return clf


def get_all_model_trainers():
    models = [params_LR(), params_SVM(), params_RF()]
    names = ['LogisticRegression', 'SVM', 'RandomForest']
    return models, names


def binarize(y, threshold=0.5):
    y = np.array(y)
    y = np.ma.fix_invalid(y, fill_value=threshold)
    y[y >= threshold] = 1
    y[y < threshold] = 0
    return y


def select_best_model(X, y):
    # Split into train and test to measure the best model
    # Use cross validation for each model to select best hyperparameters
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models, names = get_all_model_trainers()

    created_models = []
    scores = []

    for model, name in zip(models, names):
        creator, params = model
        best_m = select_best_params(creator, name, params, X, y)

        best_m.fit(train_X, train_y)
        y_pred = best_m.predict_proba(test_X)[:, 1]
        score = mean_squared_error(test_y, y_pred)
        scores.append(score)
        created_models.append(best_m)
        print("%s: %0.5f mean squared error" % (name, score))

    best_idx = np.argmin(scores)
    print('Best model {}'.format(names[best_idx]))

    print('Test scores:')
    y_pred = created_models[best_idx].predict_proba(test_X)[:, 1]
    print('MSE: {}'.format(mean_squared_error(test_y, y_pred)))
    print('AUC: {}'.format(roc_auc_score(test_y, y_pred)))
    print('Accuracy: {}'.format(accuracy_score(test_y, binarize(y_pred))))
    print('F1: {}'.format(f1_score(test_y, binarize(y_pred))))

    return created_models[best_idx]


def train_model(input_truth, input_pairs, cache_dir):
    # Training input
    print('-> Loading input')

    gold = {}
    for line in open(input_truth):
        d = json.loads(line.strip())
        gold[d['id']] = int(d['same'])

    labels = []
    text_pairs = []

    for line in tqdm(open(input_pairs)):
        d = json.loads(line.strip())
        if d['id'] in gold:
            text_pairs.append(d['pair'])
            labels.append(gold[d['id']])

    # Data transformation
    vecs = vectorizers.get_vectorizers(True, text_pairs)
    scalars = scalar_features.get_all_features()
    print('-> Transforming data')

    # Transform ALL data, not just train
    loaded_X = False

    if cache_dir:
        path = cache_dir + os.sep + 'transformed.pickle'
        if os.path.isfile(path):
            print('-> Loading transformed data')
            with open(path, 'rb') as handle:
                X = pickle.load(handle)
                loaded_X = True

    if not loaded_X:
        X = transform_data(vecs, scalars, text_pairs, cache_dir)
        if cache_dir:
            with open(path, 'wb') as handle:
                pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Model training
    y = np.array(labels)
    print('-> Training models')

    model = select_best_model(X, y)

    # Retrain on the entire dataset
    model.fit(X, y)

    y_pred = model.predict(X)
    print('Train scores:')
    print('MSE: {}'.format(mean_squared_error(y, y_pred)))
    print('AUC: {}'.format(roc_auc_score(y, y_pred)))
    print('Accuracy: {}'.format(accuracy_score(y, binarize(y_pred))))
    print('F1: {}'.format(f1_score(y, binarize(y_pred))))

    return model, vecs, scalars
