#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Sudheer Kolachina"
__copyright__ = "Copyright 2017, The Bullshit project"
__credits__ = ["mufula tufula inspirators"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Klassic Koala"
__email__ = "koala@mit.edu"
__status__ = "Pregnancy"

import argparse
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
import re
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis, NMF, LatentDirichletAllocation, TruncatedSVD, FastICA
from sklearn.metrics import silhouette_samples, silhouette_score
from time import time
from extract_ud_verbs import extract_verbs 
from mpl_toolkits.mplot3d import Axes3D

def component_analysis(verbs):
    verb_data = verbs.iloc[:,1:]
    verb_data.dropna(inplace=True)
    fig = plt.figure(1)
    plt.clf()
    ax = Axes3D(fig, elev=-150, azim=110)
    #comp = PCA(n_components=3,random_state=10).fit_transform(verb_data)
    #comp = FactorAnalysis(n_components=3,svd_method='lapack',random_state=10).fit_transform(verb_data)
    #comp = PCA(n_components=3,svd_solver='randomized',random_state=10).fit_transform(verb_data)
    #comp = NMF(n_components=3, init='nndsvda',random_state=10).fit_transform(verb_data)
    #comp = TruncatedSVD(n_components=3,random_state=10).fit_transform(verb_data)
    #comp = FastICA(n_components=3,random_state=10).fit_transform(verb_data)
    comp = LatentDirichletAllocation(n_components=3,learning_method='online',random_state=10).fit_transform(verb_data) 
    ax.scatter(comp[:, 0], comp[:, 1], comp[:, 2],
               cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First three components")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.show()

def silhouette_analysis(verbs):
    range_n_clusters = [2, 3, 4, 5, 6]
    data = verbs.iloc[:,1:]
    data.dropna(inplace=True)
    #verb_data = FactorAnalysis(n_components=3,svd_method='lapack',random_state=10).fit_transform(data)
    #verb_data = NMF(n_components=2, init='nndsvda',random_state=10).fit_transform(data)
    #verb_data = LatentDirichletAllocation(n_components=2,learning_method='online',random_state=10).fit_transform(data) 
    #verb_data = FastICA(n_components=2,random_state=10).fit_transform(data) 
    #verb_data = PCA(n_components=2).fit_transform(data)
    verb_data = PCA(n_components=2,svd_solver='randomized',random_state=10).fit_transform(data)
    for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(verb_data) + (n_clusters + 1) * 10])

        # Initialize the kmeans with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        kmeans = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = kmeans.fit_predict(verb_data)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(verb_data, cluster_labels)
        print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(verb_data, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("Silhouette plot for different clusters.")
        ax1.set_xlabel("Silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(verb_data[:, 0], verb_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = kmeans.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("Visualization of clustered data")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering"
                      "with %d clusters" % n_clusters),
                    fontsize=12, fontweight='bold')

    plt.show()

def cluster_verbs(verbs):
    verb_data = verbs.iloc[:,1:]
    verb_data.dropna(inplace=True)
    data = PCA(n_components=2).fit_transform(verb_data)
    kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
    kmeans.fit(data)
    print 'Inertia: %.3f, Silhouette score: %.3f'%(kmeans.inertia_, silhouette_score(data, 
                                kmeans.labels_,metric='euclidean'))


    h = 0.02
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    
    plt.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on Verbs')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
                 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help="Categories to use", default="true")
    args = parser.parse_args()
    verbs = pd.DataFrame(extract_verbs(args.infile))
    #component_analysis(verbs)
    #cluster_verbs(verbs)
    silhouette_analysis(verbs)
    
if __name__ == "__main__":
    main()
