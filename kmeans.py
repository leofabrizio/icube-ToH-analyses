# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:53:23 2020

This is the code I used to clusterize subjects using kmeans. The output of this cluster analysis is
already included in the dataset in the column 'cluster'

@author: FLeo
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#carico dataframe coi dati                                    
file_name = 'data.csv'
df = pd.read_csv(file_name)
# df.drop(df.columns[0], axis=1, inplace=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#drop useless columns

df.drop(['initials','trial','order','pin_facet_1',
         'pin_facet_2','pin_facet_3','pin_facet_4','pin_facet_5','pin_facet_6',
         'correct','response','n_tocchi_trial','n_tocchi_facet_1','n_tocchi_facet_2',
         'n_tocchi_facet_3','n_tocchi_facet_4','n_tocchi_facet_5','n_tocchi_facet_6',
         'touch_density_trial','filt_touch_density_trial','touch_density_facet_1','touch_density_facet_2',
         'touch_density_facet_3','touch_density_facet_4','touch_density_facet_5',
         'touch_density_facet_6','filt_touch_density_facet_1','filt_touch_density_facet_2',
         'filt_touch_density_facet_3','filt_touch_density_facet_4','filt_touch_density_facet_5',
         'filt_touch_density_facet_6','dur_facet_1','dur_facet_2','dur_facet_3',
         'dur_facet_4','dur_facet_5','dur_facet_6','raw_mean_rot_velocity'], axis=1, inplace=True) 

# df_sub.reset_index(drop=True,inplace=True)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
codes = np.arange(1,21)
lista = []

for code in codes:
        
        df_sub = df[df['code']==code]
        lista.append([df_sub['accuracy'].mean(),df_sub['MRT'].mean()])
        
#convert to dataframe
dati = pd.DataFrame(lista)
# dati.rename(columns={0: "accuracy", 1: "n°_tocchi_trial", 2: 'duration_trial', 3: "ratio_touch", 4: 'ratio_duration', 
#                              5: 'max_rotation', 6: 'MRT'},inplace=True)
dati.rename(columns={0: "accuracy", 1: 'MRT'},inplace=True)

#normalize
X = dati.values
clus_dataset = StandardScaler().fit_transform(X)

#check for optimal k 
max_clusters = 20
inertia_values = []
for n in range(2,max_clusters):
    k_means = KMeans(init='k-means++',n_clusters=n, n_init=12)
    k_means.fit(clus_dataset)
    inertia_values.append(k_means.inertia_)
    
plt.plot(range(2,max_clusters),inertia_values)
plt.xlabel('Number of Clusters',fontsize=14)
plt.ylabel('Inertia',fontsize=14)
plt.show()

#model
clusterNum = 2
k_means = KMeans(init = 'k-means++', n_clusters = clusterNum, n_init = 12)
k_means.fit(clus_dataset)
labels = k_means.labels_
print(labels)

dati['Clus_km']=labels
dati.head(5)
centroidi = dati.groupby('Clus_km').mean()
dati.insert(0,'code',codes)
dati.to_csv('kmeans_acc_plus_mrt.csv',index=False)
