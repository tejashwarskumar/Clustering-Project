import pandas as pd
crime = pd.read_csv("C:/My Files/Excelr/06 - Clustering/Assignment/crime_data.csv")
crime.describe()
crime.columns

def normDatafnc(i):
    x = (i - i.min())/i.max() - i.min();
    return x;

crime.iloc[:,1:]

normData = normDatafnc(crime.iloc[:,1:])

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage
import matplotlib.pylab as plt

z = linkage(normData,method="complete",metric='euclidean')
plt.figure(figsize=(10,5));plt.title("Crime Dentogram");plt.xlabel("index");plt.ylabel('Distance')
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=5)

from sklearn.cluster import AgglomerativeClustering
crime_complete = AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="complete").fit(normData)
crime_complete.labels_
crime['clust'] = crime_complete.labels_
crime.head()
crime_clustData = crime.iloc[:,[5,0,1,2,3,4]]
crime_clustData.iloc[:,1:].groupby(crime.clust).median()

def normFunc(i):
    x = (i-i.min()/i.max()-i.min())
    return x;

normData = normFunc(crime.iloc[:,1:])

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

model = KMeans(4)
model.fit(normData)
model.labels_
labels = pd.Series(model.labels_)
crime['clust'] = labels
crime = crime.iloc[:,[5,0,1,2,3,4]]
crime.iloc[:,1:5].groupby(crime['clust']).mean()
crime.iloc[:,1].value_counts().index[0]
