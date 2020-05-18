# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:58:23 2020

@author: pchu1207
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:51:02 2020

@author: pchu1207
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd
#%%
os.chdir("//ucdenver.pvt/som/NEUROSURG/N_ LAB - Alexander/Data/calcium imaging/calcium imaging sample movie")
data = pd.read_csv('deconv.csv')
topo_centroids = pd.read_csv('topo_centroids.csv')
os.chdir("C:/Users/pchu1207/Google Drive/Data/Alexander/Code")

topo_centroids2 = topo_centroids.iloc[one_index2,:] #keep the cell coordinates from cells with activity
topo_centroids2 = topo_centroids2.reset_index(drop=True) #drop stupid indices
topo_centroids3 = topo_centroids2
topo_centroids3['cluster'] = k_labels
topo_centroids3.columns = ['x','y','cluster']
#
data2 = np.transpose(data)
data2 = data2.reset_index(drop=True) # drop stupid indices
one_index = np.all(data2==0, axis=0) #get bin ary for any cells with activity
one_index2 = one_index.index[one_index == False].tolist() #get index for cells with no activity 
topo_centroids2 = topo_centroids.iloc[one_index2,:] #keep the cell coordinates from cells with activity
topo_centroids2 = topo_centroids2.reset_index(drop=True) #drop stupid indices
data2= data2.loc[:, (data != 0).any(axis=1)] #delete any cells that have no activity 
data2 = pd.DataFrame(data2)        

data_binary=[]
data_binary[:]=np.where(data2>0,1,0) #convert movie to binary, any spiking = 1
data_binary_df = pd.DataFrame(data_binary)
data_binary_df_trans = transpose(data_binary_df)

k_labels_list = k_labels[0].values.tolist()
k_labels_shapes = ['circle' if x== 1 else 'diamond' for x in k_labels_list]
        
participation = []
for j in np.arange(0,len(data_binary_df.iloc[0,:]),1):
    part2 = popevent_binary + data_binary_df[j] 
    participation.append(part2)
participation = pd.DataFrame(participation)
participation_binary = np.where(participation>1,1,0) #this is the participation "population vector", 
        #if a cell contributes to the sychronized event, it is counted (1), if not, it is not counted (0)
        #output = participation_binary
participation_binary = pd.DataFrame(participation_binary)
   
participation_events = []
for k in np.arange(0,len(participation_binary.columns),1):
    temp = participation_binary.iloc[:,k].sum() / len(participation_binary)
    participation_events.append(temp)
part_array=  np.array(participation_events)
participation_events_perc = part_array[np.nonzero(part_array)].mean()

data2_p = participation_binary.transpose()
data2_p = pd.DataFrame(data2_p)
data_corrm = np.corrcoef(data2_p, rowvar=False)  #normalized covariance matrix = correlation matrix


data_corrm2 = pd.DataFrame(data_corrm)
data_corrm2 = data_corrm2.stack().reset_index()
data_corrm2.columns = ['var1', 'var2','value']
data_corrm2_filtered=data_corrm2.loc[ (data_corrm2['value'] > corr_threshold) & (data_corrm2['var1'] != data_corrm2['var2']) ] #sets threshold for correlation value and removes self correlations
data_corrm2_filteredxx = data_corrm2_filtered.reset_index(drop=True)
    

activity_frame = []
fig = go.Figure()

for k in range(0,len(data2_p),1):
    activity_frame.append(go.Scatter(x= topo_centroids3['x'], y=topo_centroids3['y'], mode ='markers',
                                marker = dict(color = data_binary_df.iloc[k,:], size = 15, symbol = k_labels_shapes)))
for i in range(0,len(activity_frame),1):
    fig.add_trace(activity_frame[i])

# Make 10th trace visible
fig.data[0].visible = True

# Create and add slider
steps = []
for i in range(0,len(activity_frame),2):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data) },
              {"title": "Frame " + str(k)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Corr coefficient: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

plot(fig)
