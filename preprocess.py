from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from imblearn.combine import SMOTEENN
from nltk import corpus
from scipy import sparse
from sklearn.preprocessing import label_binarize


class EmbedTransformer:
    """
    Generate document embedding for column using Flair (https://www.aclweb.org/anthology/C18-1139/)
    using FLAIR framework (https://www.aclweb.org/anthology/N19-4010/)
    """
    def __init__(self):
        glove_embedding = WordEmbeddings('glove')
        flair_forward_embedding = FlairEmbeddings('news-forward-fast')
        flair_backward_embedding = FlairEmbeddings('news-backward-fast')
        self.document_embeddings = DocumentPoolEmbeddings([glove_embedding,
                                                           flair_forward_embedding,
                                                           flair_backward_embedding])

    def generate_embeds(self, text):
        try:
            sentence = Sentence(text)
            self.document_embeddings.embed(sentence)
            embed = sentence.get_embedding().detach().cpu().numpy()
        except Exception as e:
            print("Exception for text value: {}".format(text))
            print(e)
        return embed


class PreProcessor:
    """
    Perform pre-processing
    Import dataframe and apply sklearn transformations
    Vectorize (non full text) strings with TF/IDF
    Normalise int values, remove mean and scale to unit variance
    Instantiate a Flair embedding transformer for text features
    Returns a sparse matrix of features
    """
    def __init__(self, components=100, jobs=8):
        stopwords = set(corpus.stopwords.words('english'))
        self.mapper = DataFrameMapper([
            (['created_at'], StandardScaler()),
            (['user_created_at'], StandardScaler()),
            (['favorite_count'], StandardScaler()),
            (['retweet_count'], StandardScaler()),
            (['user_followers_count'], StandardScaler()),
            (['user_following_count'], StandardScaler()),
            ('hashtags', TfidfVectorizer(stop_words=stopwords)),
            ('urls', TfidfVectorizer(stop_words=stopwords)),
            ('user_description', TfidfVectorizer(stop_words=stopwords)),
            ('user_location', TfidfVectorizer(stop_words=stopwords)),
            ('user_name', TfidfVectorizer(stop_words=stopwords)),
            ('user_screen_name', TfidfVectorizer(stop_words=stopwords)),
            ('user_profile_urls', TfidfVectorizer(stop_words=stopwords)),
        ], sparse=True)
        self.svd = TruncatedSVD(n_components=components, algorithm='randomized')
        self.balancer = SMOTEENN(n_jobs=jobs)

    def transform(self, df):
        new_data = df.copy()
        labels = label_binarize(new_data.pop('label'), classes=['none', 'astroturf'])
        full_text = new_data.pop('full_text')
        embeds = []
        embedder = EmbedTransformer()
        texts = full_text.values.tolist()
        for text in texts:
            embeds.append(embedder.generate_embeds(text))
        sp_embeds = sparse.csr_matrix(embeds)
        vectors = self.mapper.fit_transform(new_data)
        return sparse.hstack([vectors, sp_embeds], format='csr'), labels

    def truncate(self, data_array):
        """
        Run feature dimensionality reduction process
        In this case LSA using a randomised sampling methodology (https://arxiv.org/abs/0909.4061)
        Returns a dense array
        """
        return self.svd.fit_transform(data_array)

    def balance(self, data_array, labels):
        """
        Balance classes for training
        Re-sample with SMOTE oversampling (https://arxiv.org/abs/1106.1813)
        & edited nearest-neighbours cleaning of the synthetic data points
        (http://www.inf.ufrgs.br/maslab/pergamus/pubs/balancing-training-data-for.pdf)
        """
        return self.balancer.fit_resample(data_array, labels.ravel())
