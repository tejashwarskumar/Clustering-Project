import pandas as pd
airline = pd.read_csv("C:/My Files/Excelr/06 - Clustering/Assignment/EastWestAirlines.csv")
airline.describe()
airline.columns

def normDatafnc(i):
    x = (i - i.min())/i.max() - i.min();
    return x;

airline.iloc[:,1:]

normData = normDatafnc(airline.iloc[:,1:])

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage
import matplotlib.pylab as plt

z = linkage(normData,method="complete",metric='euclidean')
plt.figure(figsize=(10,5));plt.title("airline Dentogram");plt.xlabel("index");plt.ylabel('Distance')
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=5)

from sklearn.cluster import AgglomerativeClustering
airline_complete = AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="complete").fit(normData)
airline_complete.labels_
airline['clust'] = airline_complete.labels_
airline.head()
airline_clustData = airline.iloc[:,[5,0,1,2,3,4]]
airline_clustData.iloc[:,1:].groupby(airline.clust).median()

def normFunc(i):
    x = (i-i.min()/i.max()-i.min())
    return x;

normData = normFunc(airline.iloc[:,1:])

k = list(range(2,15))
len(k)

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
TWSS = [];
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(normData)
    WSS=[];
    for j in range(i):
        WSS.append(sum(cdist(normData.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,normData.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
        
len(WSS)
len(TWSS)

#screen plot
plt.plot(k,TWSS,'ro-');plt.xlabel("K Cluster Number");plt.ylabel("Total Within Square");

model = KMeans(6)
model.fit(normData)
model.labels_
labels = pd.Series(model.labels_)
airline['clust'] = labels
airline = airline.iloc[:,[5,0,1,2,3,4]]
airline.iloc[:,1:5].groupby(airline['clust']).mean()
airline.iloc[:,1].value_counts().index[0]
