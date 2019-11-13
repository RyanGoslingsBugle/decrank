from os import listdir
import datetime

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support, \
    accuracy_score

import dbsetup
import frameset
import preprocess
import skmodels
import tfmodels


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
    print("Applying transformations to {} set...".format(type_str))
    data_arr, labels = preprocessor.transform(df)
    print("Transformation produced a: {}".format(type(data_arr)))
    print("With shape: {}".format(data_arr.shape))
    print("Saving array of transformed values...")
    sparse.save_npz('data/{}-data-tf.npz'.format(type_str), data_arr)
    joblib.dump(labels, 'data/{}-label-tf.gz'.format(type_str), compress=3)
    del df

    return data_arr, labels


def process(df, type_str, dimensions, balance=True):
    """
    Apply data transformations/dimensionality reduction
    :param dimensions: List of int values to reduce dimensions to
    :param balance: Boolean apply class re-sampling to resolve imbalance
    :type df: Pandas dataframe
    :type type_str: training or test data label
    """
    preprocessor = preprocess.PreProcessor()
    data_arr, labels = do_transforms(df, type_str, preprocessor)

    for dimension in dimensions:
        print("Applying dimensionality reduction...")
        data_lsa = preprocessor.truncate(data_arr, dimension)
        print("Reduction produced a: {}".format(type(data_lsa)))
        print("With shape: {}".format(data_lsa.shape))

        print("Saving reduced data...")
        joblib.dump(data_lsa, 'data/{}-data-dm-{}.gz'.format(type_str, dimension), compress=3)
        del data_lsa

    del data_arr

    if balance:
        for dimension in dimensions:
            print("Applying class re-sampling to array with dimensions: {}...".format(dimension))
            data_arr = joblib.load('data/{}-data-dm-{}.gz'.format(type_str, dimension))
            data_rs, label_rs = preprocessor.balance(data_arr, labels)
            print("Re-sampling produced a: {}".format(type(data_rs)))
            print("With shape: {}".format(data_rs.shape))
            print("Label shape: {}".format(label_rs.shape))

            print("Saving prepared data...")
            joblib.dump(data_rs, 'data/{}-data-rs-{}.gz'.format(type_str, dimension), compress=3)
            joblib.dump(label_rs, 'data/{}-label-rs-{}.gz'.format(type_str, dimension), compress=3)
            del data_rs, label_rs


def predict(data, dimension, labels, model_type='sk-models'):
    """
    Perform predictions with sklearn models
    :param model_type:
    :param labels: np array of true label values
    :param dimension: string representation of x-axis dimension of input vectors
    :param data: numpy array of model input vectors
    :return: dictionary of predicted labels for test data (keys = model names, values = list of label values)
    """
    if model_type == 'sk-models':
        models = skmodels.SkModels()
    elif model_type == 'tf-models':
        models = tfmodels.TFModels(dimension)
    else:
        raise KeyError('Model type not found.')
    models.load_models('models/{}-{}'.format(model_type, dimension))
    predictions = models.predict(data)
    report = {}
    for name, scores in predictions.items():
        print("Test scores for {} model, data dimensions: {}".format(name, data.shape))
        accuracy = accuracy_score(labels, scores)
        precision, recall, f_score, support = precision_recall_fscore_support(labels, scores, average='macro')
        roc_auc = roc_auc_score(labels, scores, average='macro')
        report[name] = [accuracy, precision, f_score, roc_auc]

        c_matrix = confusion_matrix(labels, scores)
        new_ax = plt.subplot(label='{}-{}'.format(name, dimension))
        cm_plt = sns.heatmap(c_matrix, annot=True, fmt='d', ax=new_ax, cbar=None, cmap='Greens',
                             xticklabels=['none', 'astroturf'], yticklabels=['none', 'astroturf'])
        new_ax.set_xlabel('Predicted labels')
        new_ax.set_ylabel('True labels')
        new_ax.set_title('Confusion matrix for {} model at {} dimensions'.format(name, dimension))
        now = datetime.datetime.now()
        cm_plt.get_figure().savefig('results/{}-test-{}-{}-{}-cmatrix.png'
                                   .format(now.strftime("%Y-%m-%d"), model_type, name, dimension))

    pd_report = pd.DataFrame.from_dict(report, orient='index',
                                       columns=['accuracy', 'precision', 'f1-score', 'roc_auc'])
    now = datetime.datetime.now()
    pd_report.to_csv('results/{}-test-{}-{}-results.csv'
                     .format(now.strftime("%Y-%m-%d"), model_type, dimension), index=True)
    return predictions


def train(data, dimension, labels, model_type='sk-models'):
    """
    Train and validate model set
    :param model_type:
    :param dimension: string representation of x-axis length of input vector
    :param data: np array of data input vectors
    :param labels: np array of label values
    :return: Pandas Dataframe of model scores
    """
    if model_type == 'sk-models':
        models = skmodels.SkModels()
        scores = models.run_models(data, labels)
        models.save_models('models/{}-{}'.format(model_type, dimension), data, labels)
        print("Training scores for data with dimensions: {}".format(data.shape))
        print(scores)
        now = datetime.datetime.now()
        scores.to_csv('results/{}-train-{}-{}-results.csv'
                      .format(now.strftime("%Y-%m-%d"), model_type, dimension), index=True)
    elif model_type == 'tf-models':
        models = tfmodels.TFModels(dimension)
        scores = models.run_models(data, labels)
        models.save_models('models/{}-{}'.format(model_type, dimension))
        print("Training scores for data with dimensions: {}".format(data.shape))
        for name, frame in scores.items():
            print("Model type: {}".format(name))
            print(frame)

            plt.figure()
            plt.plot(frame['accuracy'], label='Accuracy')
            plt.plot(frame['recall'], label='Recall')
            plt.plot(frame['precision'], label='Precision')
            plt.plot(frame['f1-score'], label='F1 Score')
            plt.plot(frame['roc_auc'], label='ROC AUC')
            plt.legend(loc='best')

            now = datetime.datetime.now()
            plt.savefig('results/{}-train-{}-{}-{}-history.png'
                        .format(now.strftime("%Y-%m-%d"), model_type, name, dimension))
            frame.to_csv('results/{}-train-{}-{}-{}-results.csv'
                         .format(now.strftime("%Y-%m-%d"), model_type, name, dimension), index=False)
    else:
        raise KeyError('Model type not found.')
    return scores


def main():
    dimensions = [100, 500, 1_000]

    print("Loading data...")
    import_files()

    df = load('merged_tweets')
    df.to_pickle('train-data.zip')
    df = pd.read_pickle('train-data.zip')
    df = df.sample(frac=0.1)

    tdf = load('recent_merged_tweets')
    tdf.to_pickle('test-data.zip')
    tdf = pd.read_pickle('test-data.zip')
    tdf = tdf.sample(frac=0.1)

    print("Start pre-processing...")
    process(df, 'train', dimensions)
    process(tdf, 'test', dimensions, False)
    print("Pre-processing complete.")

    print("Running model training...")
    for dimension in dimensions:
        data = joblib.load('data/train-data-rs-{}.gz'.format(dimension))
        labels = joblib.load('data/train-label-rs-{}.gz'.format(dimension))
        train(data, dimension, labels)
        train(data, dimension, labels, 'tf-models')

    print("Making predictions...")
    for dimension in dimensions:
        data = joblib.load('data/test-data-dm-{}.gz'.format(dimension))
        labels = joblib.load('data/test-label-tf.gz'.format(dimension))
        predict(data, dimension, labels)
        predict(data, dimension, labels, 'tf-models')


if __name__ == '__main__':
    main()
