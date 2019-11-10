from os import listdir
from pprint import pprint
import datetime

import joblib
import pandas as pd
from scipy import sparse
from sklearn.metrics import classification_report

import dbsetup
import frameset
import preprocess
import skmodels


def import_files():
    """
    Import data from files into mongodb and create merged collections
    """
    importer = dbsetup.Importer('astroturf')

    print("Importing Twitter training set...")
    importer.clear(col_name='ira_tweets')
    importer.import_tweets_from_file(col_name='ira_tweets', filename='data/ira_tweets_csv_unhashed.csv')
    importer.clear(col_name='event_tweets')
    importer.import_tweets_from_file(col_name='event_tweets', filename='data/2012_events_twitter.json',
                                     json_type=True, borked_json=False)
    importer.clear(col_name='merged_tweets')
    importer.merge_tweets(merge_col='merged_tweets', col_1='ira_tweets', col_2='event_tweets')

    print("Importing Twitter test set...")
    importer.clear(col_name='recent_tweets')
    for file in listdir('data/recent_clean_data'):
        importer.import_tweets_from_file(col_name='recent_tweets', filename='data/recent_clean_data/' + file,
                                         json_type=True, borked_json=True)
    importer.clear(col_name='recent_ast_tweets')
    for file in listdir('data/recent_ast_data'):
        importer.import_tweets_from_file(col_name='recent_ast_tweets', filename='data/recent_ast_data/' + file)
    importer.clear(col_name='recent_merged_tweets')
    importer.merge_tweets(merge_col='recent_merged_tweets', col_1='recent_ast_tweets', col_2='recent_tweets')

    print("Importing Reddit test set...")
    importer.clear(col_name='ira_reddit')
    importer.import_reddit_from_file(col_name='ira_reddit', filename='data/comments.csv')
    importer.clear(col_name='old_reddit')
    for file in listdir('data/reddit'):
        importer.import_reddit_from_file(col_name='old_reddit', filename='data/reddit/' + file, json_type=True)
    importer.clear(col_name='merged_reddit')
    importer.merge_reddit(merge_col='merged_reddit', col_1='ira_reddit', col_2='old_reddit')


def load(collection):
    """
    Load and format tweet data as Pandas dataframe
    :param collection: Collection name to read from
    :return: Pandas Dataframe
    """
    print("Reading to Dataframe...")
    framer = frameset.Framer('astroturf')
    df = framer.get_frame(collection)
    return df


def do_transforms(df, type_str, preprocessor):
    print("Applying transformations...")
    data_arr, labels = preprocessor.transform(df)
    print("Transformation produced a: {}".format(type(data_arr)))
    print("With shape: {}".format(data_arr.shape))
    print("Saving array of transformed values...")
    sparse.save_npz('data/{}-data-tf.npz'.format(type_str), data_arr)
    joblib.dump(labels, 'data/{}-label-tf.gz'.format(type_str), compress=3)
    del df

    return data_arr, labels


def process(df, type_str, balance=True):
    """
    Apply data transformations/dimensionality reduction
    :param balance: Boolean apply class re-sampling to resolve imbalance
    :type df: Pandas dataframe
    :type type_str: training or test data label
    """
    preprocessor = preprocess.PreProcessor()
    data_arr, labels = do_transforms(df, type_str, preprocessor)

    for dimensions in [100, 1_000]:
        print("Applying dimensionality reduction...")
        data_lsa = preprocessor.truncate(data_arr, dimensions)
        print("Reduction produced a: {}".format(type(data_lsa)))
        print("With shape: {}".format(data_lsa.shape))

        print("Saving reduced data...")
        joblib.dump(data_lsa, 'data/{}-data-dm-{}.gz'.format(type_str, dimensions), compress=3)
        del data_lsa

    print("Saving full set...")
    joblib.dump(data_arr.toarray(), 'data/{}-data-dm-full.gz'.format(type_str), compress=3)
    del data_arr

    if balance:
        for dimensions in [100, 1_000, 'full']:
            print("Applying class re-sampling to array with dimensions: {}...".format(dimensions))
            data_arr = joblib.load('data/{}-data-dm-{}.gz'.format(type_str, dimensions))
            data_rs, label_rs = preprocessor.balance(data_arr, labels)
            print("Re-sampling produced a: {}".format(type(data_rs)))
            print("With shape: {}".format(data_rs.shape))
            print("Label shape: {}".format(label_rs.shape))

            print("Saving prepared data...")
            joblib.dump(data_rs, 'data/{}-data-rs-{}.gz'.format(type_str, dimensions), compress=3)
            joblib.dump(label_rs, 'data/{}-label-rs-{}}.gz'.format(type_str, dimensions), compress=3)
            del data_rs, label_rs


def train_sk(data, dimension, labels):
    """
    Train and validate sklearn model set
    :param dimension: string representation of x-axis length of input vector
    :param data: np array of data input vectors
    :param labels: np array of label values
    :return: Pandas Dataframe of model scores
    """
    simple_models = skmodels.SkModels()
    scores = simple_models.run_models(data, labels)
    simple_models.fit_and_save_models('models/sk-models-{}'.format(dimension), data, labels)
    print("Training scores for data with dimensions: {}".format(data.shape))
    pprint(scores)
    now = datetime.datetime.now()
    with open('results/{}-train-{}-results.txt'.format(now.strftime("%Y-%m-%d"), dimension),
              mode='a', encoding='utf-8') as f:
        f.write(scores)
    return scores


def predict_sk(data, dimension, labels):
    """
    Perform predictions with sklearn models
    :param labels: np array of true label values
    :param dimension: string representation of x-axis dimension of input vectors
    :param data: numpy array of model input vectors
    :return: dictionary of predicted labels for test data (keys = model names, values = list of label values)
    """
    simple_models = skmodels.SkModels()
    simple_models.load_models('models/sk-models-{}'.format(dimension))
    predictions = simple_models.predict(data)
    pprint(classification_report(labels, predictions, target_names=['none', 'astroturf']))
    now = datetime.datetime.now()
    with open('results/{}-test-{}-results.txt'.format(now.strftime("%Y-%m-%d"), dimension),
              mode='a', encoding='utf-8') as f:
        f.write(classification_report(labels, predictions, target_names=['none', 'astroturf']))
    return predictions


def main():
    print("Loading data...")
    # import_files()
    # df = load('merged_tweets')
    # df.to_pickle('train-data.zip')
    df = pd.read_pickle('train-data.zip')
    df = df.sample(frac=0.01)
    print("Start pre-processing...")
    process(df, 'train')
    tdf = load('recent_merged_tweets')
    tdf.to_pickle('test-data.zip')
    process(tdf, 'test', False)
    print("Pre-processing complete.")

    print("Running model training...")
    for dimension in [100, 1_000, 'full']:
        with joblib.load('data/train-data-rs-{}.gz'.format(dimension)) as data:
            with joblib.load('data/train-label-rs-{}.gz'.format(dimension)) as labels:
                train_sk(data, dimension, labels)

    print("Making predictions...")
    for dimension in [100, 1_000, 'full']:
        with joblib.load('data/test-data-dm-{}.gz'.format(dimension)) as data:
            with joblib.load('data/test-label-tf.gz'.format(dimension)) as labels:
                predict_sk(data, dimension, labels)


if __name__ == '__main__':
    main()
