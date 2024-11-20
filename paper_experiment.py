import pickle
import time
import pandas as pd
from os.path import join, exists

from ITMO_FS import RFS, reliefF_measure, pearson_corr
from ITMO_FS.filters.multivariate import MultivariateFilter
from ITMO_FS.filters.univariate import spearman_corr, f_ratio_measure
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from arpa.shannon_information import shannon
from arpa.transformer.FeatureSelectionTransformer import ARPA
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from evaluetor.model import Model
from evaluetor.simpledb import SimpleDB

num_workers = 10
n_repeats = 10
n_splits = 10
processes = 10
verbose = 0
metrics = [accuracy_score]
# metrics = [accuracy_score, f1_score, recall_score, precision_score]

internal_gss = StratifiedKFold(n_splits=2)
gss = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

train_size = 0.8
folder = "./classification/"
features_range_percentile = [0.2, 0.4, 0.6, 0.8]
n = 5
max_iter = [1000]
version = 0
solver = ["saga"]

db_name = "./databases/{}x{}.mat_new".format(n_repeats, n_splits)


def save(path_to_file, obj):
    with open(path_to_file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(path_to_file):
    try:
        with open(path_to_file, 'rb') as handle:
            return pickle.load(handle)
    except EOFError:
        return None

def get_pipeline(cls, selector=None, memory="./cache", n_bins=5):
    if selector is not None:
        return Pipeline([
            ('selector', selector),
            ('classifier', cls)],
            memory=memory)
    else:
        return Pipeline([
            ('classifier', cls)],
            memory=memory)


def get_models(features_range, X_discrete, target, path_matrix=None, pre_dispatch=10, C=[1]):
    models = []
    cls = LinearSVC(random_state=42, class_weight="balanced", C=1)
    ####################################ARPA##############################################

    for m in [shannon.process_call_corrcoef]:
        params = {
            'selector__k': features_range,
            'classifier__C': C,
        }

        selector = ARPA(k=10,
                                                   function=m,
                                                   verbose=verbose,
                                                   processes=processes,
                                                   )

        pipe = get_pipeline(cls, selector=selector)

        models.append(Model(cls=pipe, metrics=metrics, params=params,
                           n_jobs=num_workers, gss=internal_gss,
                           name='arpa', pre_dispatch=pre_dispatch))


####################################chi2##############################################
    selector = SelectKBest(chi2, k=10)

    params = {
        'selector__k': features_range,
        'classifier__C': C
    }

    pipe = get_pipeline(cls, selector)
    models.append(Model(cls=pipe, metrics=metrics, params=params, n_jobs=num_workers,
                        gss=internal_gss, name="chi2"))
    ####################################Person##############################################
    selector = SelectKBest(pearson_corr, k=10)

    params = {
        'selector__k': features_range,
        'classifier__C': C
    }

    pipe = get_pipeline(cls, selector)
    models.append(Model(cls=pipe, metrics=metrics, params=params, n_jobs=num_workers,
                        gss=internal_gss, name="pearson"))
    ####################################F Score##############################################
    selector = SelectKBest(f_ratio_measure, k=10)

    params = {
        'selector__k': features_range,
        'classifier__C': C
    }

    pipe = get_pipeline(cls, selector)
    models.append(Model(cls=pipe, metrics=metrics, params=params, n_jobs=num_workers,
                        gss=internal_gss, name="f_score"))

    ####################################reliefF##############################################
    selector = SelectKBest(reliefF_measure, k=10)

    params = {
        'selector__k': features_range,
        'classifier__C': C
    }

    pipe = get_pipeline(cls, selector)
    models.append(Model(cls=pipe, metrics=metrics, params=params, n_jobs=num_workers,
                        gss=internal_gss, name="relief"))

    ####################################LogisticRegression##############################################
    selector = SelectKBest(mutual_info_classif, k=10)

    params = {
        'selector__k': features_range,
        'classifier__C': C
    }

    pipe = get_pipeline(cls, selector)

    models.append(Model(cls=pipe, metrics=metrics, params=params, n_jobs=num_workers,
                        gss=internal_gss, name="mic"))

    ####################################MRMR##############################################

    for measure in ["MRMR", "JMI"]:
        selector = MultivariateFilter(measure=measure, n_features=features_range[0])
        params = {
            'classifier__C': C
        }
        pipe = Pipeline([
            ('pre_selector', SelectKBest(mutual_info_classif, k=1000)),
            ('selector', selector),
            ('classifier', cls)],
            memory="./cache")

        models.append(Model(cls=pipe, metrics=metrics, params=params, n_jobs=num_workers,
                            gss=internal_gss, name=measure.lower(), pre_dispatch=pre_dispatch))
    return models


rows = []
it = 0

datasets = pd.read_csv('csv/data_description.csv')
datasets.sort_values(by='attributes', inplace=True)

columns = {'metric': 'text', 'value': 'text', 'classifier': 'text', 'n_features': 'text',
           'elapsed_time': 'text', 'it': 'text', 'fold': 'text', 'n1': 'text', 'n2': 'text',
           'dataset': 'text', 'filename': 'text'}

db = SimpleDB(columns=columns, audit_db_name=db_name)
db.init_db()

folder = "./csv/"

if __name__ == '__main__':

    for idx, row in datasets.iterrows():
        dataset_name = row["dataset_name"]
        df = pd.read_csv(join(folder, dataset_name + ".csv")).values
        X, Y = df[:, :-1], df[:, -1]

        print(dataset_name)

        n_features = row["attributes"]
        n_classes = row["classes"]
        n_samples = row["size"]
        feature_range = [10, 20, 30, 40, 50, 60]


        for n_feature in feature_range:
            for it, (train_index, test_index) in enumerate(gss.split(X, Y)):
                path = './costs/dataset_name:{}-n_repeats:{}-n_splits:{}-it:{}'.format(dataset_name, n_repeats, n_splits, it)
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                # Discretize X and remove constant variables

                pipe = Pipeline([
                                ('scaler', preprocessing.StandardScaler()),
                                ('discretize',
                                 preprocessing.KBinsDiscretizer(n_bins=5,
                                                                encode='ordinal',
                                                                strategy='uniform')),
                    ]
                )

                X_train = pipe.fit_transform(X_train, Y_train).astype(int)
                X_test = pipe.transform(X_test).astype(int)

                classifiers = get_models(features_range=[n_feature], X_discrete=X_train, target=Y_train,
                                         path_matrix=path)

                for index, model in enumerate(classifiers):

                    filename = "{}-{}-{}-{}".format(dataset_name, model.name, n_feature, it)
                    audit = db.get_file_process_status(filename + "_" + metrics[0].__name__ + "_test")

                    if audit is None or (audit is not None and audit[-1] != 'SUCCESS'):
                        try:
                            print(f"Now processing text for {filename}")
                            start = time.time()
                            result, predict, search = model.eval(X_train, X_test, Y_train, Y_test)
                            end = time.time()
                            result['n_features'] = n_feature
                            result['elapsed_time'] = end - start
                            result['it'] = it
                            result['fold'] = (it + 1) % n_splits
                            result['n1'] = len(Y_train)
                            result['n2'] = len(Y_test)
                            result['dataset'] = dataset_name
                            result['filename'] = filename
                            result.columns = ['metric', 'value', 'classifier', 'n_features', 'elapsed_time',
                                              'it', 'fold', 'n1', 'n2', 'dataset', 'filename']
                            for m in result['metric']:
                                data = result.loc[result['metric'] == m].to_dict('records')[0]
                                db.start_file_process(filename + "_" + m, data)
                                db.finalize_file_process(filename + "_" + m)
                            print("Inserted Done")
                        except Exception as inst:
                            print(inst)
                            print(inst.args)
                            db.finalize_file_process(filename + "_" + metrics[0].__name__, 'ERROR')
                            print("ERROR")


