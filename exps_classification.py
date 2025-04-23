from sklearn.datasets import load_iris, load_diabetes, load_digits, load_linnerud, load_wine, load_breast_cancer
import umap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes, load_digits, load_wine, load_breast_cancer
from sklearn.datasets import load_linnerud
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import time
from sklearn.utils import Bunch
from sklearn.preprocessing import normalize
import numpy as np
from sklearn import manifold, datasets
import random


DS_FOLDER = './datasets/'
MAT_DS_FOLDER = DS_FOLDER + 'mat_ds/'
HILL_VALLEY = MAT_DS_FOLDER + 'hillvalley.mat'
LOW_SPEC = MAT_DS_FOLDER + 'lowresspectR.mat'
ISOLET = MAT_DS_FOLDER + 'isolet.mat'
ARCENE = MAT_DS_FOLDER + 'arcene.mat'


def get_df_from_path(path_df):
    df = pd.read_csv(path_df)
    df = df.iloc[: , 1:]
    data = df.iloc[:,:-1]
    labels = df.iloc[:,-1:]
    target = np.array(labels)
    target = np.squeeze(target)
    ds_bunch = Bunch()
    ds_bunch.data = np.asarray(data)
    ds_bunch.target = target
    return ds_bunch


def get_test_datasets_exps():
    # load datasets
    from sklearn import datasets
    breast_cancer = datasets.load_breast_cancer()
    wine = datasets.load_wine()
    path_df = DS_FOLDER +'low_spectra.csv'
    low_spectra = get_df_from_path(path_df)
    path_df = DS_FOLDER +'isolet.csv'
    isolet = get_df_from_path(path_df)
    path_df = DS_FOLDER + 'musk.csv'
    musk = get_df_from_path(path_df)
    path_df = DS_FOLDER + 'madelon.csv'
    madelon = get_df_from_path(path_df)
    digits = load_digits()
    test_datasets = [wine, digits, breast_cancer, low_spectra, musk, isolet, madelon]
    ds_names = ['wine', 'digits', 'breast_cancer', 'low_spectra', 'musk', 'isolet', 'madelon']
    test_datasets = [low_spectra, musk, isolet, madelon]
    ds_names = ['low_spectra', 'musk', 'isolet', 'madelon']
    test_datasets = [wine, digits, breast_cancer, low_spectra]
    ds_names = ['wine', 'digits', 'breast_cancer', 'low_spectra']
    return test_datasets, ds_names


def reduce_dim_umap(dataset_name, data, labels):
    df_results = pd.DataFrame(columns=['dataset', 'distance', 'n_neighbors', 'min_dist', 'n_components', 'accuracy', 'f1_score', 'umap_time', 'runtime'])
    results = []
    metrics = ['euclidean', 'manhattan', 'cosine', 'canberra', 'braycurtis']
    neighbors = [15, 30, 50, 100]
    min_dists = [0.01, 0.1, 1]
    data_features = data.shape[1]
    n_comps = [int(data_features*0.20), int(data_features*(0.30)), int(data_features*0.50)]
    # Normalize data
    data = normalize(data)
    # Baseline: no umap
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)
    svm = SVC()
    init_time = time.time()
    svm.fit(X_train, y_train)
    total_time = round(time.time() - init_time, 4)
    y_preds = svm.predict(X_test)
    acc = round(accuracy_score(y_test, y_preds), 4)
    f1_s = round(f1_score(y_test, y_preds, average='micro'), 4)
    df_results.loc[df_results.shape[0]] = [dataset_name, 'raw_data', 0, 0, 0, acc, f1_s, 0, total_time]
    # UMAP results...
    for m in metrics:
        for n in neighbors:
            for dist in min_dists:
                for n_c in n_comps:
                    cum_acc, cum_f1, cum_runtime, cum_embtime = 0, 0, 0, 0
                    num_reps = 10
                    for rep in range(0,num_reps):
                        init_embtime = time.time()
                        emb_umap_train = umap.UMAP(n_neighbors=n, metric=m, min_dist=dist, n_components=n_c).fit_transform(X_train)
                        emb_time = round(time.time() - init_embtime, 6)
                        svm = SVC()
                        init_runtime = time.time()
                        svm.fit(emb_umap_train, y_train)
                        run_time = round(time.time() - init_runtime, 6)
                        emb_umap_test = umap.UMAP(n_neighbors=n, metric=m, min_dist=dist, n_components=n_c).fit_transform(X_test)
                        y_preds = svm.predict(emb_umap_test)
                        acc = round(accuracy_score(y_test, y_preds), 4)
                        f1_s = round(f1_score(y_test, y_preds, average='micro'), 4)
                        cum_acc = cum_acc + acc
                        cum_f1 = cum_f1 + f1_s
                        cum_embtime = cum_embtime + emb_time
                        cum_runtime = cum_runtime + run_time
                    cum_acc = round(cum_acc/num_reps, 4)
                    cum_f1 = round(cum_f1/num_reps, 4)
                    cum_embtime = round(cum_embtime/num_reps, 6)
                    cum_runtime = round(cum_runtime/num_reps, 6)
                    df_results.loc[df_results.shape[0]] = [dataset_name, m, n, dist, n_c, cum_acc, cum_f1, cum_embtime, cum_runtime]
    return df_results



def generate_line(n_samples, noise=10, offset=0):
  x_lines=list(range(n_samples))
  y_lines=list(range(n_samples))
  for x in x_lines:
    y_lines[x]=x_lines[x]+random.random()*noise+offset
  return x_lines, y_lines

def generate_lines_dataset(n_samples, num_lines):
  x=[]
  y=[]
  for i in range(num_lines):
    x_coord, y_coord=generate_line(n_samples,offset=40*i)

    for index_c in range(n_samples):
      x.append([x_coord[index_c],y_coord[index_c]])
      y.append(i)
  return np.array(x), np.array(y)


def load_toy_datasets():
    n_samples = 500
    n_components = 2
    # Dataset circles
    X_circles, y_circles = datasets.make_circles(n_samples=n_samples, factor=0.2)
    # Dataset moons
    x_moons, y_moons = datasets.make_moons(n_samples=n_samples)
    # Dataset lines
    x_lines, y_lines = generate_lines_dataset(150, 2)
    all_datasets = {}
    all_datasets['circles'] = (X_circles, y_circles)
    all_datasets['moons'] = (x_moons, y_moons)
    all_datasets["lines"] = (x_lines, y_lines)
    return all_datasets


def load_datasets():
    linnerrud = load_linnerud()
    X_linnerrud = linnerrud.data
    y_linnerrud = linnerrud.target
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target
    wine = load_wine()
    X_wine = wine.data
    y_wine = wine.target
    cancer = load_breast_cancer()
    X_cancer = cancer.data
    y_cancer = cancer.target
    all_datasets = {}
    #all_datasets['digits'] = (X_digits, y_digits)
    #all_datasets["linnerrud"] = (X_linnerrud, y_linnerrud)
    all_datasets['wine'] = (X_wine, y_wine)
    all_datasets["cancer"] = (X_cancer, y_cancer)
    return all_datasets



def start_exps_classification():
    all_datasets = {}
    test_ds, names_ds = get_test_datasets_exps()
    for td, dn in zip(test_ds,names_ds):
        print('* Dataset ', dn)
        print('data shape = ', td.data.shape)
        print('target shape = ', td.target.shape)
        print('')
        all_datasets[dn] = (td.data, td.target)

    #all_datasets = load_datasets()
    #all_datasets = load_toy_datasets()
    for key_dataset in all_datasets:
        df_results = pd.DataFrame(columns=['dataset', 'distance', 'n_neighbors', 'min_dist', 'n_components', 'accuracy', 'f1_score', 'time'])
        print(key_dataset)
        input_dataset = all_datasets[key_dataset][0]
        target_dataset = all_datasets[key_dataset][1]
        results = reduce_dim_umap(key_dataset, input_dataset, target_dataset)
        #df_results = pd.concat([df_results, results])
        results.to_csv(key_dataset + '_' + 'classification_results_norm_10reps.csv')

