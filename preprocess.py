from imblearn.combine import SMOTEENN
from sklearn.base import TransformerMixin
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk import corpus
from sklearn.preprocessing import label_binarize
import numpy as np
import spacy
import torch
from spacy.compat import cupy


class EmbedTransformer(TransformerMixin):
    """
    Generate embeddings with spacy implementation of DistilBERT model (https://arxiv.org/abs/1910.01108)
    Use take average of word embeddings as document embedding
    """
    def __init__(self):
        is_using_gpu = spacy.require_gpu()
        if is_using_gpu:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.nlp = spacy.load('en_trf_distilbertbaseuncased_lg')

    def fit_transform(self, data, **kwargs):
        new_x = []
        for doc in self.nlp.pipe(data):
            new_x.append(cupy.asnumpy(doc.vector))
        return np.array(new_x)


class PreProcessor:
    """
    Perform pre-processing
    Import dataframe and apply sklearn transformations
    Vectorize (non full text) strings with TF/IDF
    Normalise int values, remove mean and scale to unit variance
    Instantiate an embedding transformer for message text feature
    Returns a sparse matrix of features
    """
    def __init__(self):
        stopwords = set(corpus.stopwords.words('english'))
        self.mapper = DataFrameMapper([
            (['created_at'], StandardScaler()),
            (['user_created_at'], StandardScaler()),
            (['favorite_count'], StandardScaler()),
            (['retweet_count'], StandardScaler()),
            (['user_followers_count'], StandardScaler()),
            (['user_following_count'], StandardScaler()),
            ('hashtags', TfidfVectorizer(stop_words=stopwords, max_features=1_000)),
            ('urls', TfidfVectorizer(stop_words=stopwords, max_features=1_000)),
            ('user_description', TfidfVectorizer(stop_words=stopwords, max_features=10_000)),
            ('user_location', TfidfVectorizer(stop_words=stopwords, max_features=1_000)),
            ('user_name', TfidfVectorizer(stop_words=stopwords, max_features=1_000)),
            ('user_screen_name', TfidfVectorizer(stop_words=stopwords, max_features=1_000)),
            ('user_profile_urls', TfidfVectorizer(stop_words=stopwords, max_features=1_000)),
            ('full_text', EmbedTransformer())
        ], sparse=True)
        self.svd = TruncatedSVD(algorithm='randomized')
        self.balancer = SMOTEENN(n_jobs=12)

    def transform(self, df):
        labels = label_binarize(df.pop('label'), classes=['none', 'astroturf'])
        return self.mapper.fit_transform(df), labels

    def truncate(self, data_array, components=1000):
        """
        Run feature dimensionality reduction process
        In this case LSA using a randomised sampling methodology (https://arxiv.org/abs/0909.4061)
        Returns a dense array
        """
        self.svd.n_components = components
        return self.svd.fit_transform(data_array)

    def balance(self, data_array, labels):
        """
        Balance classes for training
        Re-sample with SMOTE oversampling (https://arxiv.org/abs/1106.1813)
        & edited nearest-neighbours cleaning of the synthetic data points
        (http://www.inf.ufrgs.br/maslab/pergamus/pubs/balancing-training-data-for.pdf)
        """
        return self.balancer.fit_resample(data_array, labels.ravel())
