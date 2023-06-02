from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
import pandas as pd
import matplotlib.pyplot as plt
import os

import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('agg')


def upload_file(request):
    if request.method == 'POST':
        file = request.FILES['file']
        data = pd.read_csv(file)
        data.loc[(data['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].mean()
        data.loc[(data['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].mean()
        data.drop('CUST_ID', axis=1, inplace=True)
        plt.figure(figsize=(10, 50))
        for i in range(len(data.columns)):
            plt.subplot(17, 1, i + 1)
            sns.histplot(data=data, x=data.columns[i], kde=True, color="g")
            plt.title(data.columns[i])
        plt.tight_layout()
        plt.savefig(os.path.join('static', 'histogram.png'))
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        scores_1 = []
        range_values = range(1, 20)

        for i in range_values:
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(data_scaled)
            scores_1.append(kmeans.inertia_)
        plt.plot(scores_1, 'bx-')
        plt.savefig(os.path.join('static', 'kmeans.png'))
        kmeans = KMeans(4)
        kmeans.fit(data_scaled)
        labels = kmeans.labels_
        cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=[data.columns])
        cluster_centers = scaler.inverse_transform(cluster_centers)
        cluster_centers = pd.DataFrame(data=cluster_centers, columns=[data.columns])
        y_kmeans = kmeans.fit_predict(data_scaled)
        data_cluster = pd.concat([data, pd.DataFrame({'cluster': labels})], axis=1)
        for i in data.columns:
            plt.figure(figsize=(35, 5))
            for j in range(4):
                plt.subplot(1, 4, j + 1)
                cluster = data_cluster[data_cluster['cluster'] == j]
                cluster[i].hist(bins=20)
                plt.title('{}     \nCluster  {}  '.format(i, j))
            plt.savefig(os.path.join('static', 'cluster_{}.png'.format(i)))
        return render(request, 'result.html')
    else:
        return render(request, 'upload.html')
