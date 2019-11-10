from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
from joblib import dump, load
from os import path


class SkModels:
    def __init__(self):
        self.models = {
            'SVM': NuSVC(gamma='scale'),
            'Bayes': GaussianNB(),
            'Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
        }
        self.scoring = {
            'Accuracy': 'accuracy',
            'Precision': 'precision_macro',
            'Recall': 'recall_macro',
            'F-score': 'f1_macro',
            'AUC': 'roc_auc'
        }

    def validate(self, model, data, labels):
        return cross_validate(model, data, labels, cv=4, scoring=self.scoring,
                              return_train_score=False, n_jobs=-1, return_estimator=False)

    def run_models(self, data, labels):
        scores = {}
        for name, model in self.models:
            scores[name] = self.validate(model, data, labels)

        avg_scores = {}
        for name, validation in scores.items():
            avg_scores[name] = []
            for score in validation:
                avg_scores[name].append(np.mean(validation[score]))

        return pd.DataFrame.from_dict(avg_scores, orient='index', columns=[
            'fit_time',
            'score_time',
            'accuracy',
            'precision',
            'recall',
            'f-score',
            'roc_auc'
        ])

    def predict(self, data):
        results = {}
        for name, model in self.models:
            results[name] = model.predict(data)
        return results

    def fit_and_save_models(self, filepath, data, labels):
        for name, model in self.models:
            model.fit(data, labels)
            dump(model, '{}-{}.gz'.format(filepath, name), compress=3)

    def load_models(self, filepath):
        for name, model in self.models:
            if path.isfile('{}-{}.joblib'.format(filepath, name)):
                self.models[name] = load('{}-{}.gz'.format(filepath, name))
