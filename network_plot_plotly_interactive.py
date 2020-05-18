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



#%%
data2 = np.transpose(data)
data2 = data2.reset_index(drop=True) # drop stupid indices
one_index = np.all(data2==0, axis=0) #get binary for any cells with activity
one_index2 = one_index.index[one_index == False].tolist() #get index for cells with no activity 
topo_centroids2 = topo_centroids.iloc[one_index2,:] #keep the cell coordinates from cells with activity
topo_centroids2 = topo_centroids2.reset_index(drop=True) #drop stupid indices
data2= data2.loc[:, (data != 0).any(axis=1)] #delete any cells that have no activity 
data2 = pd.DataFrame(data2)        

data_binary=[]
data_binary[:]=np.where(data2>0,1,0) #convert movie to binary, any spiking = 1
data_binary_df = pd.DataFrame(data_binary)
data_binary_df_trans = transpose(data_binary_df)
        
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

node_trace_corr = []
edge_trace_corr= []
node_adjacencies = []
node_text = []

# Add traces, one for each slider step
for corr_threshold in np.arange(0, 1, 0.1):

    data_corrm2 = pd.DataFrame(data_corrm)
    data_corrm2 = data_corrm2.stack().reset_index()
    data_corrm2.columns = ['var1', 'var2','value']
    data_corrm2_filtered=data_corrm2.loc[ (data_corrm2['value'] > corr_threshold) & (data_corrm2['var1'] != data_corrm2['var2']) ] #sets threshold for correlation value and removes self correlations
    data_corrm2_filteredxx = data_corrm2_filtered.reset_index(drop=True)
    
    G = nx.Graph()
    for k, rows in topo_centroids3.iterrows():
        G.add_node(k, pos=(float(topo_centroids3.loc[k,'x']),float(topo_centroids3.loc[k,'y'])))  

    
    for j, rows in data_corrm2_filteredxx.iterrows():
        G.add_edge(data_corrm2_filteredxx.iloc[j,0], data_corrm2_filteredxx.iloc[j,1], weight = data_corrm2_filteredxx.iloc[j,2])  
    
    labels = nx.get_edge_attributes(G,'weight')
    labels2 = list(labels.values())
    labels3 = np.array(labels2)
    labels4 = labels3*5
    node_pos = nx.get_node_attributes(G,'pos')
    degree_dict = dict(G.degree(G.nodes()))
    degree = list(degree_dict.values())
    degree_mean1 = sum(degree)/len(degree)
    degree1 = [(x/degree_mean1)*2 for x in degree]
    degree2 = [x*50 for x in degree]
    k_labels_list = k_labels[0].values.tolist()
    k_labels_shapes = ['circle' if x== 1 else 'square' for x in k_labels_list]

    
    edge_x = []
    edge_y = []
    color_weight = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace_corr.append(go.Scatter(
            x=edge_x, y=edge_y,
            hoverinfo='text',
            mode='lines'))



    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        
    node_trace_corr.append(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo="text",
            marker=dict(
                    showscale=True,
                    # colorscale options
                    #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                    #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                    #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                    colorscale='RdBu',
                    reversescale=True,
                    color=[],
                    size=10,
                    colorbar=dict(
                            thickness=15,
                            title='Node Connections',
                            xanchor='left',
                            titleside='right'
                            ),
                            line_width=2)))
  
 
    node_adjac= []
    node_text2 = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjac.append(len(adjacencies[1]))
        node_text2.append('# of connections: '+str(len(adjacencies[1])))
    node_adjacencies.append(node_adjac)
    node_text.append(node_text2)


    for k in range(0,len(node_trace_corr),1):
        node_trace_corr[k].marker.color = k_labels_list
        node_trace_corr[k].marker.symbol = k_labels_shapes
        node_trace_corr[k].marker.size= node_adjacencies[k]
        node_trace_corr[k].text = node_text[k]
        edge_trace_corr[k].marker.color = labels3



fig= go.Figure()
for step in range(0,10,1):
    fig.add_trace(edge_trace_corr[step])
    fig.add_trace(node_trace_corr[step])
    
  
# Make 10th trace visible
fig.data[0].visible = True

#fig=go.Figure()
#fig.add_trace(edge_trace_corr[0]) # true ,true, false, false
#fig.add_trace(node_trace_corr[0]) # 
#fig.add_trace(edge_trace_corr[8]) # false, false, true, true
#fig.add_trace(node_trace_corr[8]) # 

# Create and add slider
steps = []
k= -0.1
for i in range(0,20,2):
    k= round(k+0.1,2)
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data) },
              {"title": "Correlation Coefficient " + '>' + str(k)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
    step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
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

#%%

