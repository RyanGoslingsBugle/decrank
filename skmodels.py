from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
import numpy as np
import pandas as pd
from joblib import dump, load
from os import path


class SkModels:
    def __init__(self):
        self.models = {
            'SGD': SGDClassifier(random_state=1, max_iter=10_000, verbose=1, early_stopping=True, n_jobs=-1),
            'Bayes': GaussianNB(),
            'Forest': RandomForestClassifier(verbose=1, n_estimators=100, n_jobs=-1),
        }
        self.scoring = {
            'Accuracy': 'accuracy',
            'Precision': 'precision_macro',
            'Recall': 'recall_macro',
            'F-score': 'f1_macro',
            'AUC': 'roc_auc'
        }

    def validate(self, model, data, labels):
        splitter = StratifiedKFold(n_splits=10, random_state=1)
        return cross_validate(model, data, labels, cv=splitter, scoring=self.scoring,
                              return_train_score=False, n_jobs=-1,
                              return_estimator=False, error_score=np.nan)

    def run_models(self, data, labels):
        scores = {}
        for name, model in self.models.items():
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
        for name, model in self.models.items():
            results[name] = model.predict(data)
        return results

    def save_models(self, filepath, data, labels):
        for name, model in self.models.items():
            model.fit(data, labels)
            dump(model, '{}-{}.gz'.format(filepath, name), compress=3)

    def load_models(self, filepath):
        for name, model in self.models.items():
            if path.isfile('{}-{}.gz'.format(filepath, name)):
                self.models[name] = load('{}-{}.gz'.format(filepath, name))
