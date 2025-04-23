import gensim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


def plot_similar_words(title, labels, embedding_clusters, word_clusters, saving_path):
    plt.figure(figsize=(16, 9))
    cmap = cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=0.8, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.6, xy=(x[i], y[i]), xytext=(0, 0),
                         textcoords='offset points', ha='right', va='bottom', size=10)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.savefig(saving_path)


def start_exps_word_embeddings():
    # model = api.load("glove-twitter-25")  # load glove vectors, 25-dimensional vector
    corpus = api.load('text8')
    model = Word2Vec(corpus)

    key_words = ['dream', 'paris', 'car', 'computer']

    similar_words = [] # Para almacenar la lista de palabras similares de cada palabra clave
    similar_embeddings = [] # Para almacenar la lista de embeddings de las palabras similares de cada palabra clave

    for word in key_words:
        embeddings = []
        words = []
        for similar_word, _ in model.wv.most_similar(word, topn=20):
            words.append(similar_word)
            embeddings.append(model.wv[similar_word])
        similar_words.append(words)
        similar_embeddings.append(embeddings)


    # Redimensionamos para unir los embeddings de cada palabra clave en un único conjunto
    similar_embeddings = np.array(similar_embeddings)
    n, m, k = similar_embeddings.shape
    embs_list = similar_embeddings.reshape(n * m, k)

    metrics = ['euclidean', 'manhattan', 'canberra', 'cosine', 'braycurtis']
    neighbors = [2, 15, 30, 50, 100]
    min_dist = [0.01, 0.05, 0.1, 0.5, 1]
    n_neigh = 100
    for min_d in min_dist:
        for metric in metrics:
            # Obtenemos los embeddings generados con UMAP
            embs_umap = umap.UMAP(n_neighbors=n_neigh, min_dist=min_d, n_components=2,
                                  metric=metric, random_state=123).fit_transform(embs_list)

            # Volvemos redimensionar para obtener los embeddings que corresponden a cada palabra clave
            embs_umap = embs_umap.reshape(n, m, 2)
            saving_path = './results/word_emb_' + metric + '_n' + str(n_neigh) + '_' + str(min_d) + '.png'
            plot_similar_words('d = ' + metric + ', n = ' + str(n_neigh) + ', min_dist = ' + str(min_d), key_words, embs_umap, similar_words, saving_path)
