#!/usr/bin/env python
# SIM-CITY client
#
# Copyright 2015 Netherlands eScience Center <info@esciencecenter.nl>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from matplotlib.pyplot import subplot, figure, imshow, plot, axis, title
from sklearn.externals import joblib
import os
import zlib
import numpy
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from scipy import sparse
import math
from sklearn import manifold
from sklearn.manifold import MDS 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class ScikitLda(object):

    def __init__(self, corpus=None, lda=None, n_topics=10,
                 max_iter=5, learning_method='online', learning_offset=50.,
                 **kwargs):
        if lda is None:
            self.lda = LatentDirichletAllocation(
                n_topics=n_topics, max_iter=max_iter,
                learning_method=learning_method,
                learning_offset=learning_offset, **kwargs)
        else:
            self.lda = lda

        self._corpus = corpus
        self._weights = None

    def fit(self):
        self.lda.fit(self.corpus.sparse_matrix())

    def partial_fit(self, corpus):
        self.lda.partial_fit(corpus.sparse_matrix())
        self._weights = None

    @property
    def topics(self):
        return self.lda.components_

    @property
    def n_topics(self):
        return self.lda.n_topics

    @property
    def corpus(self):
        return self._corpus

    @property
    def weights(self):
        if self._weights is None:
            self._weights = self.partial_weights(self.corpus)
        return self._weights

    def partial_weights(self, corpus):
        weights = self.transform(corpus)
        return (weights.T / weights.sum(axis=1)).T

    def transform(self, corpus):
        return self.lda.transform(corpus.sparse_matrix())

    def topic_words(self, n_words=10):
        topicWords = []
        topicWeightedWords = []

        for topic_idx, topic in enumerate(self.topics):
            weightedWordIdx = topic.argsort()[::-1]
            wordsInTopic = [self.corpus.word(i)
                            for i in weightedWordIdx[:n_words]]

            weights = topic / topic.sum()
            topicWeights = [(weights[i], self.corpus.word(i))
                            for i in weightedWordIdx[:n_words]]

            topicWords.append(wordsInTopic)
            topicWeightedWords.append(topicWeights)

        return (topicWords, topicWeightedWords)

    def save(self, filename):
        joblib.dump(self.lda, filename)

    @classmethod
    def load(cls, filename, corpus=None):
        lda = joblib.load(filename)
        return cls(lda=lda, corpus=corpus)


def topics_by_discrete_property(lda, all_property_values):
    values = np.unique(all_property_values)
    topicsByProperty = np.empty((len(values), lda.n_topics))

    for i, prop_value in enumerate(values):
        mask = np.asarray(prop_value == all_property_values)
        prop_weights = lda.weights[mask]
        topicsByProperty[i] = np.average(prop_weights, axis=0)

    return topicsByProperty, values


def topics_by_integer_property(lda, all_property_values, delta=5):
    all_property_values = np.array(all_property_values)
    size = int(np.nanmax(all_property_values) + 1)
    topicsByProperty = np.zeros((size, lda.n_topics))

    lower = all_property_values - delta
    upper = all_property_values + delta
    for prop_value in np.arange(size):
        mask = (prop_value >= lower) & (prop_value <= upper)
        prop_weights = lda.weights[mask]
        if len(prop_weights) > 0:
            topicsByProperty[prop_value] = np.average(prop_weights, axis=0)

    return topicsByProperty


def plot_wordcloud_with_property(topicWeightedWords, topicsByProperty):
    figure(figsize=(16, 40))
    for idx, topic in enumerate(topicWeightedWords):
        wc = WordCloud(background_color="white")
        img = wc.generate_from_frequencies(
            [(word, weight) for weight, word in topic])
        subplot(len(topicWeightedWords), 2, 2 * idx + 1)
        imshow(img)
        axis('off')

        subplot(len(topicWeightedWords), 2, 2 * idx + 2)
        plot(topicsByProperty[:, idx])
        axis([10, 100, 0, 1.0])
        title('Topic #%2d' % (idx))

def ComputeAngle(x):
    result = math.fabs(2 * math.acos(1-math.fabs(x))/math.pi)
    return result

def Angularize(a):
    for i in range(len(a)): 
        for j in range(len(a[0])):
            v = ComputeAngle(a[i][j])
            a[i][j] = v
    


def main():
    # alldata = ScikitLda('lda_3.tar.bz2')
    #alldata = ScikitLda()
    #alldata.load('lda_3.tar\lda_3.pkl')
    #lda = ScikitLda.load('lda_3.tar\lda_3\lda_3.pkl')    
    #print(lda.topics)

    #print(lda.topics)
    dirname = "F:/enron_out_0.1/"
    topics = []
    for subdir in [x[0] for x in os.walk(dirname)][1:]:
        for filename in os.listdir(subdir):
         if filename.endswith('pkl'):
                print("attempting... ", filename)
                lda = ScikitLda.load(subdir+"/"+filename)
                for topic in lda.topics:
                    topics.append(topic / topic.sum())
    
    cos_distance = pairwise_distances(topics, metric='cosine')
    Angularize(cos_distance)
    # pick number of components=2 so that we can plot on 2-space., random_state is chosen so that we can re-produce. 
    mds = TSNE(n_components=2)
    #mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1) 
    pos = mds.fit_transform(cos_distance) # shape (n_components, n_samples) 
    xs, ys = pos[:, 0], pos[:, 1]


    from sklearn.cluster import KMeans
    k_fit = KMeans(n_clusters=25).fit_predict(cos_distance)
    figure(figsize=(15,15))
    x = np.arange(10)
    yys = [i+x+(i*x)**2 for i in range(25)]
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, 25))
    #print colors[24]

    #plt.scatter(xs,ys, c=k_fit, s=100)
    for idx in range(0,25):
        plt.scatter(xs[numpy.where(k_fit==idx)], ys[numpy.where(k_fit==idx)], s=100, label=str(idx), c=colors[idx])
        plt.legend()   
    plt.show()      
    


    print("d")


main()
