from os import listdir
from pprint import pprint

import numpy as np
import pandas as pd
from scipy import sparse

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
    importer.import_tweets_from_file(col_name='ira_tweets', filename='data/ira_tweets_csv_unhashed.csv')
    importer.import_tweets_from_file(col_name='event_tweets', filename='data/2012_events_twitter.json',
                                     json_type=True, borked_json=False)
    importer.merge_tweets(merge_col='merged_tweets', col_1='ira_tweets', col_2='event_tweets')

    print("Importing Twitter test set...")
    for file in listdir('data/recent_clean_data'):
        importer.import_tweets_from_file(col_name='recent_tweets', filename='data/recent_clean_data/' + file,
                                         json_type=True, borked_json=True)
    for file in listdir('data/recent_ast_data'):
        importer.import_tweets_from_file(col_name='recent_ast_tweets', filename='data/recent_ast_data/' + file)
    importer.merge_tweets(merge_col='recent_merged_tweets', col_1='recent_ast_tweets', col_2='recent_tweets')

    print("Importing Reddit test set...")
    importer.import_reddit_from_file(col_name='ira_reddit', filename='data/comments.csv')
    for file in listdir('data/reddit'):
        importer.import_reddit_from_file(col_name='old_reddit', filename='data/reddit/' + file, json_type=True)
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
    parts = 0
    if len(df.index) > 1000000:
        parts = (len(df.index) // 1000000) + 1
        print("Splitting into {} parts...".format(parts))
        dfs = np.array_split(df, parts)
        del df
        i = 0
        for df in dfs:
            print("Transforming part {}".format(i))
            data_arr, labels = preprocessor.transform(df)
            print("Transformation produced a: {}".format(type(data_arr)))
            print("With shape: {}".format(data_arr.shape))

            print("Saving array of transformed values...")
            sparse.save_npz('{}-data-tf-part{}.npz'.format(type_str, i), data_arr)
            np.savez('{}-label-tf-part{}.npz'.format(type_str, i), labels=labels)
            i += 1
        del dfs
    else:
        data_arr, labels = preprocessor.transform(df)
        print("Transformation produced a: {}".format(type(data_arr)))
        print("With shape: {}".format(data_arr.shape))

        print("Saving array of transformed values...")
        sparse.save_npz('{}-data-tf.npz'.format(type_str), data_arr)
        np.savez('{}-label-tf.npz'.format(type_str), labels=labels)

    return parts, data_arr, labels


def process(df, type_str):
    """
    Apply data transformations/dimensionality reduction
    :type df: Pandas dataframe
    :type type_str: training or test data label
    """
    preprocessor = preprocess.PreProcessor()
    parts, data_arr, labels = do_transforms(df, type_str, preprocessor)

    if parts > 0:
        for part in range(0, parts):
            new_matrix = sparse.load_npz('{}-data-tf-part{}.npz'.format(type_str, part))
            with np.load('{}-label-tf-part{}.npz'.format(type_str, part)) as arr_list:
                new_labels = arr_list['labels']
            if part == 0:
                data_arr = new_matrix
                labels = new_labels
            else:
                data_arr = sparse.vstack([data_arr, new_matrix])
                labels = np.vstack((labels, new_labels))

    datasets = {}
    for dimensions in [100, 1000]:
        print("Applying dimensionality reduction...")
        data_lsa = preprocessor.truncate(data_arr, dimensions)
        print("Reduction produced a: {}".format(type(data_lsa)))
        print("With shape: {}".format(data_lsa.shape))

        print("Applying class re-sampling...")
        data_rs, label_rs = preprocessor.balance(data_lsa, labels)
        print("Re-sampling produced a: {}".format(type(data_rs)))
        print("With shape: {}".format(data_rs.shape))
        print("Label shape: {}".format(label_rs.shape))

        print("Saving prepared data...")
        np.savez('{}-data-rs-{}.npz'.format(type_str, dimensions), data=data_rs, labels=label_rs)

        datasets[str(dimensions)] = (data_rs, label_rs)

    return datasets


def train_sk(data, labels):
    """
    Train model set on data
    :param data: np array of data vectors
    :param labels: np array of label values
    :return: Pandas Dataframe of model scores
    """
    simple_models = skmodels.SkModels()
    scores = simple_models.run_models(data, labels)
    simple_models.fit_and_save_models('models/sk-models', data, labels)
    return scores


def main():
    print("Loading data...")
    # import_files()
    # df = load('merged_tweets')
    # df.to_pickle('train-data.zip')
    df = pd.read_pickle('train-data.zip')

    print("Start pre-processing...")
    process(df, 'train')
    print("Pre-processing complete.")

    print("Running model training...")
    for dimension in [100, 1000]:
        with np.load('train-data-rs-{}.npz'.format(dimension)) as arr_list:
            data = arr_list['data']
            labels = arr_list['labels']
        training_scores = train_sk(data, labels)
        print("Training scores for data with dimensions: {}".format(data.shape))
        pprint(training_scores)


if __name__ == '__main__':
    main()
