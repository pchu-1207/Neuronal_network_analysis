# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:30:44 2020

@author: pchu1207
"""
bltin_sum = sum

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy import signal
import random
from statsmodels.distributions.empirical_distribution import ECDF
import sklearn.cluster
from sklearn.manifold import TSNE
import seaborn as sb
from sklearn.decomposition import PCA
import time
from plotnine import *
import matplotlib.style
import matplotlib as mpl
import networkx as nx
from networkx.algorithms import community
from operator import itemgetter
import statistics
from matplotlib import animation


class analyze_ca():
    def __init__(self,file_directory, shuffle_num,scale_factor_time,scale_factor_spatial,activity_data_file, spatial_data_file,optimization_choice):
        self.file_directory = file_directory
        self.shuffle_num= shuffle_num #how many shuffles to estimate statistical significance
        self.scale_factor_time = scale_factor_time # 1 frame =how many ms
        self.scale_factor_spatial= scale_factor_spatial #1 pixel = how many microns
        self.activity_data_file = activity_data_file
        self.spatial_data_file = spatial_data_file
        self.optimization_choice = optimization_choice

    def load_data(self):
        global data2
        global topo_centroids2
        os.chdir(self.file_directory)
        data = pd.read_csv(self.activity_data_file)
        topo_centroids = pd.read_csv(self.spatial_data_file)
        data2 = np.transpose(data)
        data2 = data2.reset_index(drop=True) # drop stupid indices
        one_index = np.all(data2==0, axis=0) #get boolean for any cells with activity
        one_index2 = one_index.index[one_index == False].tolist() #get index for cells with no activity 
        topo_centroids2 = topo_centroids.iloc[one_index2,:] #keep the cell coordinates from cells with activity
        topo_centroids2 = topo_centroids2.reset_index(drop=True) #drop stupid indices
        data2= data2.loc[:, (data != 0).any(axis=1)] #delete any cells that have no activity 
        data2 = pd.DataFrame(data2)

############################################analyze spike frequency################################################
    def deconv_freq(self):
        global spike_freq_time
        global spike_freq_time_mean
        spike_freqs = data2.mean(axis=0)
        spike_freq_time = spike_freqs/len(data2)*self.scale_factor_time     #average number of spikes across all frames
        spike_freq_time = pd.DataFrame(spike_freq_time)
        spike_freq_time_mean = float(spike_freq_time.sum()/len(spike_freq_time))
        print("the average spike frequency is: " + str(spike_freq_time_mean))

######################################################analyze sychronized calcium events##########################################
    def synch_network_events(self):
        global shuffled_array3
        global popevent_prob
        global popevent_binary
        global hist_array
        global sum_o_spikes_array
        #make a sum of spikes array for original movie
        sum_o_spikes_array = []
        sum_o_spikes_array.append(data2.sum(axis = 1)) #sum across rows, data is inside list 0
       
        #this creates surrogate movies with shuffled spikes
        shuffled_array3 = [] #create empty list to populate later
        for j in np.arange(0,self.shuffle_num,1): # #of reshuffles to do, reshuffles each cells events across time
            shuffled_array2= []
            for i in np.arange(0,len(data2.columns),1):
                dic= {} #create empty dictionary
                dic = np.random.choice(data2.iloc[:,i],size= len(data2), replace=False) #shuffle spikes j number of times
                shuffled_array2.append(dic) #appends dictionary to list
            shuffled_array3.append(shuffled_array2)
        #read as [shuffle #][cell #][timepointx:timepointy]
        
        #probably need to rewrite this section using list comprehension to make more efficient
        #e.g. [sum(i) for i in zip(*l)]
        #this loops through each shuffled array, by each timepoint across cells and appends to new array
        #point is to make statistical comparisons using these distributions for
        #each timepoint to determine whether it is a sig population event
        dist_array = []
        for j in np.arange(0,self.shuffle_num,1): #1000
            inter_array = []
            for m in np.arange(0,len(shuffled_array3[1][1]),1): #200 
                dic2 = {}
                dic2 = [item[m] for item in shuffled_array3[j]] #selects each sublist item in sequence using list comphrension
                inter_array.append(bltin_sum(dic2))#uses built in sum rather than numpy sum to create sum of spikes   
            dist_array.append(inter_array)      
            #output of this is the sum of spikes which is shuffle# x movie frames 
        hist_array = []
        inter_array2 = []
        for m in np.arange(0,len(shuffled_array3[1][1]),1): #200 
            dic3 = {}
            dic3 = [item[m] for item in dist_array]
            inter_array2.append(dic3)
        hist_array.append(inter_array2) #this creates an list array of arbitrary supralist 0 --> movie frames x # shuffles
            #each sublist conists of all reshuffled spikes for one movie frame

        # statistical probability of observed event compared to 1000 reshuffled events, empirical cumulative density function
        popevent_prob = []
        for j in np.arange(0,len(data2),1): # 200
            ecdf = ECDF(hist_array[0][j]) # creates empirical cumulative density function based off shuffles for each frame
            popevent_prob.append(ecdf(sum_o_spikes_array[0][j])) # probability of observed event in each frame

        # count the number of synchronous calcium events
        popevent_binary=[]
        popevent_binary = np.where(np.array(popevent_prob)>0.95000000,1,0) #convert to binary
        print("the number of frames with significant synchronized events is: " + str(popevent_binary.sum()))


    def event_spacing(self):
        popevent_count = []
        #create some spacing between offset and onset of events, 2 frames
        # need to fix this to determine proper spacing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for j in np.arange(0,len(data2),1):
            k = j-2
            #l = j+2
            #if popevent_prob[j] + popevent_prob[j+1] >= 1 and popevent_prob[j-1] - popevent_prob[j] =/ 0:
            if popevent_prob[j] == 1 and popevent_prob[j] != popevent_prob[k]:
                popevent_count.append(1)
            else:
                popevent_count.append(0)
       
###########################calculate % that each cell contributes to synch events ##################################################################   
        
    def network_contributions(self):
    # analyze whether individual cells contributed to the synchronous calcium events
    # 1 = an event occured and this cell did not contribute OR
    # this cell fired outside of a network event, 2 = this cell contributed to an event, 0 = no event        
        global participation_binary    
        global data2_p
        global data_binary_df
        global shuffled_array_participation
        global reordered_cell_contribute2
        global cell_contribute_perc_df
        global cell_contribute_perc
        data_binary=[]
        data_binary[:]=np.where(data2>0,1,0) #convert movie to binary, any spiking = 1
        data_binary_df = pd.DataFrame(data_binary)
        data_binary_df_trans = data_binary_df.transpose()
    ##    
        participation = []
        for j in np.arange(0,len(data_binary_df.iloc[0,:]),1):
            part2 = popevent_binary + data_binary_df[j] 
            participation.append(part2)
        participation = pd.DataFrame(participation)
        participation_binary = np.where(participation>1,1,0) #this is the participation "population vector", 
        #if a cell contributes to the sychronized event, it is counted (1), if not, it is not counted (0)
        #output = participation_binary

        participation_binary = pd.DataFrame(participation_binary)

        #participation_binary= participation_binary.drop('cluster', axis=1)


    # reshuffle the participation population vector
        shuffled_array_participation = [] #create empty list to populate later
        for j in np.arange(0,self.shuffle_num,1): # #of reshuffles to do, reshuffles each cells events across time
            shuffled_array2= []
            for i in np.arange(0,len(data_binary_df.columns),1):
                dic= {} #create empty dictionary
                dic = np.random.choice(participation_binary.iloc[i,:],size= len(participation_binary.iloc[i,:]), replace=False) #shuffle spikes j number of times
                shuffled_array2.append(dic) #appends dictionary to list
            shuffled_array_participation.append(shuffled_array2)

    #output= shuffled_array_participation

        # % cell contribution to each significant event
        cell_contribute = []
        for j in np.arange(0,len(data2.columns),1):
            dic4 = {}
            dic4 = [item[j] for item in data_binary] + popevent_binary    #compare each cell firing to sig. pop activity
            cell_contribute.append(dic4)

        cell_contribute_perc = []
        for j in np.arange(0,len(data2.columns),1):
            dic5 = {} 
            dic5 = np.count_nonzero(cell_contribute[j]==2)/np.count_nonzero(popevent_binary)
            cell_contribute_perc.append(dic5) 
        cell_contribute_perc_df = pd.DataFrame(cell_contribute_perc)
        cell_contribute_perc_df_sort = pd.DataFrame(cell_contribute_perc_df.sort_values(0,ascending=False)) #extract values sorted
        cell_contribute_perc_df_sort_index= pd.DataFrame(list(cell_contribute_perc_df_sort.index.values)) #extract indexes sorted
        #sorted_cell_contribute_perc
        #cell_contribute_perc_df # % that each cell participates in all synchronized events

        print(' the top 10 cells and contribution % to sync network events are: ' + str(cell_contribute_perc_df_sort[0:10]))

        # average % of cells that contribute to sychronized network events
        participation_events = []
        for k in np.arange(0,len(participation_binary.columns),1):
            temp = participation_binary.iloc[:,k].sum() / len(participation_binary)
            participation_events.append(temp)
        part_array=  np.array(participation_events)
        participation_events_perc = part_array[np.nonzero(part_array)].mean()

        print('the average % of cell participating in all synchronized network events is :' + str(participation_events_perc))

        data2_p = participation_binary.transpose()
        data2_p = pd.DataFrame(data2_p)
        
        
        

        
        

        

  
    def kmeans_clustering(self):  
        global reordered_cluster_amp
        global data2_tp2
        global data2_tp
        global shuffle_sil_value
        global optimal_cluster
        global sil_value
        global k_labels
        global k_labels_shuffle
        global sil_score_bycluster
        global topo_centroids3
        global topo_centroids4
        global data_corrm
        global dist_sq_euc
        global clusters
        ################################run kmeans clustering to get cell assemblies################################
        data_corrm = np.corrcoef(data2_p, rowvar=False)  #normalized covariance matrix = correlation matrix
        #squared euclidean distance matrix
        dist_sq_euc = sp.spatial.distance.cdist(data_corrm, data_corrm, metric='sqeuclidean') 

    # try differnet cluster sizes and measure silhouette score for each
        range_n_clusters = list (range(2,21))
        sil_score= []
        for n_clusters in range_n_clusters:
            clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init =10, max_iter=100)
            preds = clusterer.fit_predict(dist_sq_euc)
            centers = clusterer.cluster_centers_
    
            score = sklearn.metrics.silhouette_score(dist_sq_euc, preds)
            sil_score.append(score)

            print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))


        optimal_cluster  = sil_score.index(max(sil_score))+2
        max_sil = max(sil_score)
        print("The optimal # of clusters is " + str(optimal_cluster) + " with a silhouette score of " + str(max_sil))

        #outputs = optimal_cluster, max_sil 
    
        #########run this kmeans separately using optimal cluster # to get kmeans labels for each cell and all sil values
        #the sklearn implementation automatically minimizes the "inertia" 
        #i.e. the sum of squared distance of sample to closest cluster center
        kmeans = sklearn.cluster.KMeans(n_clusters=optimal_cluster, n_init=10, max_iter=100).fit(dist_sq_euc) 
        predict = kmeans.fit_predict(dist_sq_euc) #predicts cluster
        #len(kmeans.cluster_centers_)
        #kmeans.fit_predict(NEW_DATA) #can enter a new set of data for predict
        #this shows which cell is in which cluster
        k_labels = pd.DataFrame(kmeans.labels_)
        #output = k_labels


        sil_value= sklearn.metrics.silhouette_samples(dist_sq_euc,predict) #this gets the silhouette value for each
        #cell, whereas the silhouette_score version takes the average of these

        # binary. separate based on cluster class
        data_binary_df_tp= data_binary_df.transpose()
        data_binary_df_tp['cluster'] = k_labels
        #
        topo_centroids3 = topo_centroids2
        topo_centroids3['cluster'] = k_labels
        topo_centroids3.columns = ['x','y','cluster']

    #########################Run on binary "participation" vector#########################
        data2_tp = data2_p.transpose()
        data2_tp = pd.DataFrame(data2_tp)
        data2_tp['cluster'] = k_labels #attach the kmeans labels to the original data
        data2_tp['sil_value'] = sil_value #attach individual silhouette values to original data
        data2_tp2 = data2_p.transpose()
        data2_tp2 = pd.DataFrame(data2_tp2)
        #This creates a list with sublists for each kmeans cluster with indexes
        cluster_index_amp = []
        for j in np.arange(0,optimal_cluster,1):
            inter = []
            inter = data2_tp.index[data2_tp['cluster'] == j].tolist() #create clustersize # of index lists
            cluster_index_amp.append(inter) #creates sublist indexes for each cluster 

        #groups the clusters in the dataframe
        #creates dataframe with raw data + cluster group + sil score attached as columns
        cluster_index_amp2 = []
        cluster_index_amp3 = []
        topo_centroids4 = []
        todf2 = pd.DataFrame()  
        for j in np.arange(0,optimal_cluster,1): 
            inter = []
            inter2=[]
            inter3 =  []
            inter = data2_tp.reindex(cluster_index_amp[j]) #uses indexes to subset original data with sil values and cluster#
            inter2 = data2_tp2.reindex(cluster_index_amp[j]) #uses indexes to subset original data, raw data
            inter3 = topo_centroids3.reindex(cluster_index_amp[j]) #uses indexes to subset the spatial data to corresponding  clusters
            cluster_index_amp2.append(inter)
            cluster_index_amp3.append(inter2)
            topo_centroids4.append(inter3)
            #todf = pd.DataFrame(cluster_index_amp2[j]) #create list of dataframes, each with cells from 1 cluster
        reordered_cluster_amp = pd.concat(cluster_index_amp2) #combine all clusters back to dataframe with cluster code
        #and sil values 
    #output = reordered_cluster_amp
        for k in np.arange(0,optimal_cluster,1):
            print('the size of cluster ' + str(k) + ' is: ' + str(len(reordered_cluster_amp[reordered_cluster_amp['cluster']==k])))    


        # get mean sil score for each cluster group in df
        sil_score_bycluster = reordered_cluster_amp.groupby('cluster')['sil_value'].mean()
        # output = sil_score_bycluster

        #transform data to get corr matrix and distance matrix for all reshuffled data
        shuffled_array4 = []
        shuffled_array4.append(shuffled_array3)
        shuffle_dist = []
        for i in np.arange(0,self.shuffle_num,1): # of reshuffles  
            shuffle_sil2 = []
            data_corrm_k = np.corrcoef(shuffled_array4[0][i], rowvar=True)  #normalized covariance matrix = correlation matrix
            dist_sq_euc2 = sp.spatial.distance.cdist(data_corrm_k, data_corrm_k, metric='sqeuclidean') 
            shuffle_sil2.append(dist_sq_euc2)
            shuffle_dist.append(shuffle_sil2)

        # outputs= shuffle_dist


        #get sil values for each cell in each reshuffle
        shuffle_dist2= []
        shuffle_dist2.append(shuffle_dist)
        start = time.time()
        shuffle_sil_value = []
        k_labels_shuffle = []
        for j in np.arange(0,self.shuffle_num,1): #runs through all reshuffled cells and calculates k means + sil value for each cell
            shuffle_sil_pred = []
            kmeans_shuffle = sklearn.cluster.KMeans(n_clusters=optimal_cluster, n_init =10, max_iter=100).fit(shuffle_dist2[0][j][0]) 
            preds = kmeans_shuffle.fit_predict(shuffle_dist2[0][j][0])
            shuffle_sil_pred = sklearn.metrics.silhouette_samples(shuffle_dist2[0][j][0], preds) #here preds serves as the "labels" and shuffle_dist serves as the dissimilarity metrics
            k_labels_shuffle.append(preds)
            shuffle_sil_value.append(shuffle_sil_pred) #sil value for each shuffle for each cell
        end = time.time()
        print(end-start)
        # outputs = k_labels_shuffle, shuffle_sil_value
             
        

        
                #####################for plotting
        cell_contribute_perc_dfindex = cell_contribute_perc_df.reset_index() #turn index into column
        cell_contribute_perc_dfindex['cluster'] = k_labels
        #this gives the actual index numbers for each cluster
        cell_contribute_perc_dfindex_sort = cell_contribute_perc_dfindex.index[cell_contribute_perc_dfindex['cluster'] == 0].tolist() 
        cell_contribute_perc_dfindex_sort2 = cell_contribute_perc_dfindex.index[cell_contribute_perc_dfindex['cluster'] == 1].tolist() 
            
        cell_contribute_perc_dfindex_sort = pd.DataFrame(cell_contribute_perc_dfindex_sort)
        cell_contribute_perc_dfindex_sort2 = pd.DataFrame(cell_contribute_perc_dfindex_sort2)
            
        reordered_cell_contribute = [cell_contribute_perc_dfindex_sort, cell_contribute_perc_dfindex_sort2]
        reordered_cell_contribute = pd.concat(reordered_cell_contribute) #this is the indexes in the right order
            
        reordered_cell_contribute2 = cell_contribute_perc_df.reindex(reordered_cell_contribute.iloc[:,0])
        reordered_cell_contribute2['index'] = reordered_cell_contribute2.index #not sure why i cant reindex here, do it manually

        
        #plot cell contribution
        plt.figure()
        plt.bar(x=reordered_cell_contribute2.loc[:,'index'], height= reordered_cell_contribute2.iloc[:,0], color='blue')
        plt.ylabel('% Contribution to All Events')
        plt.xlabel('Cell #')    
        plt.axvline(x=np.count_nonzero(data2_tp['cluster']==0), color='red', linewidth=1)  #this will plot red line based on size of cluster 1, because indexes are screwed in bar plot or something
        #plt.set_xticklabels(cell_contribute_perc_df_sort_index)
        plt.title('Contribution of Each Cell to Synchronous Calcium Events')
        plt.show()

    def cluster_optimization(self):
        global data_corrm_optimized
        global sil_value_mask_array
        global data2_tp_optimized
        global clusters2
                
        #transform structure to all shuffled sil values per cell. use this to test sig of each cells sil value
        shuffle_sil_value_percell = []
        shuffle_sil_value_percell_prob = []
        for m in np.arange(0,len(shuffle_sil_value[0]),1):
            mid = [item[m] for item in shuffle_sil_value]
            shuffle_sil_value_percell.append(mid)
            ecdf = ECDF(shuffle_sil_value_percell[m])
            shuffle_sil_value_percell_prob.append(ecdf(sil_value[m]))
        #get binary of sig sil values to use as mask i.e. index to remove cell with low sil values from respective assemblies
        sil_value_sig_mask = np.where(np.array(shuffle_sil_value_percell_prob)>0.95000000,1,0)


        #get the indexes for cells that fail the threshold test for sil value aka shitty clusters
        sil_value_mask_array = np.array(sil_value_sig_mask)
        shuffle_sil_value_percell_array = pd.DataFrame(shuffle_sil_value_percell)
        shuffle_sil_value_percell_array['cluster'] = k_labels
        shuffle_sil_value_optimized = shuffle_sil_value_percell_array[sil_value_mask_array>0]
        data2_tp_optimized = data2_tp[sil_value_mask_array>0]
        for k in np.arange(0,len(shuffle_sil_value[0]),1):
            print('the cumulative probability of the sil value of cell ' + str(k) + ' is ' + str(shuffle_sil_value_percell_prob[k]))    
        print('the number of cells that fail the 95% probability test is: ' + str((sil_value_mask_array==0).sum()))
                #output = shuffle_sil_value, each cell with 1000 shuffles worth of sil values


        print(' the number of remaining cells is: '  + str(len(data2_tp_optimized)))
        for k in np.arange(0,optimal_cluster,1):
            print(' the number of cells in cluster: ' + str(k) + ' is ' + str(np.count_nonzero(data2_tp_optimized['cluster']==k)))    


        #output = shuffle_sil_value_optimized. cells that are tightly clustered to their respective assemblies based on sil values

        #distribution of shuffled sil values excluding cells that were loosely clustered
        shuffle_sil_value_opt_cluster = shuffle_sil_value_optimized.groupby('cluster').mean()
        data2_tp_opt_cluster = data2_tp_optimized.groupby('cluster').mean()
    
        #recalculate the sig value for each cluster after removing low sil value cells
        clusters_ecdf_optimized = []    
        for j in np.arange(0,optimal_cluster,1): # # of clusters
            ecdf = ECDF(shuffle_sil_value_opt_cluster[j])
            clusters_ecdf_optimized.append(ecdf(data2_tp_opt_cluster.loc[j,'sil_value']))    

        #test sil score of each cluster against reshuffled sil scores
        shuffle_sil_value2 = []
        shuffle_sil_value2.append(shuffle_sil_value)
        shuffle_sil_means = []
        start = time.time()
        for j in np.arange(0,self.shuffle_num,1): # # of clusters
            shuffle_sil_means2 = []
            shuffle_sil_means3 = []
            shuffle_sil_means2.append(shuffle_sil_value2[0][j]) # this and next line associate sil values with k mean cluster code
            shuffle_sil_means2.append(k_labels_shuffle[j])
            shuffle_sil_means3 = pd.DataFrame(shuffle_sil_means2)
            #this may be confusing later, explanation: iloc[boolean array,array to group by boolean]
            #get means for each shuffle and cluster = sil score
            sil_score_bycluster_shuffle = shuffle_sil_means3.iloc[shuffle_sil_means3.iloc[0],shuffle_sil_means3[1]].mean()
            shuffle_sil_means.append(sil_score_bycluster_shuffle)
        end = time.time()
        print(end-start)



        clusters_ecdf = []    
        for j in np.arange(0,optimal_cluster,1): # # of clusters
            clusters2= [list(x) for x in zip(*shuffle_sil_means)] #separate clusters into individual lists
            ecdf = ECDF(clusters2[j])
            clusters_ecdf.append(ecdf(sil_score_bycluster[j]))
    

        for k in np.arange(0,optimal_cluster,1):
            print('the cumulative probability of cluster ' + str(k) + ' is ' + str(clusters_ecdf[k]))    

        for k in np.arange(0,optimal_cluster,1):
            print('the cumulative probability of cluster ' + str(k) + ' after optimization is ' + str(clusters_ecdf_optimized[k]))    

    def optimization_mod(self):
        global k_labels_shuffle_optimized
        global shuffled_array_participation
        global data2_tp
        global popevent_contr_optimized
        global spike_freq_time
        global k_labels
        global topo_centroids3
        global k_labels_shuffle
        global reordered_cluster_amp2
        global data_corrm
        global popevent_contr
        global participation_binary_assembly_optimized
        #run this next block only if you want to use optimized cluster i.e. removing cells with low
        #kmean sil values
        if self.optimization_choice == 'yes':
            ################################# choose between optimized and non optimized ###########################
            #reassigns original objects with optimized assemblies
            spike_freq_time_optimized = spike_freq_time[sil_value_mask_array>0]
            spike_freq_time = spike_freq_time_optimized
            topo_centroids3_optimized = topo_centroids3[sil_value_mask_array>0]
            k_labels_optimized= k_labels[sil_value_mask_array>0]
            k_labels= k_labels_optimized
            topo_centroids3 = topo_centroids3_optimized
            reordered_cluster_amp2 = []
            for m in range(0,len(shuffle_sil_value[0]),1):
                ecdf = ECDF(shuffle_sil_value_percell[m])
                reordered_cluster_amp2.append(ecdf(reordered_cluster_amp.loc[m,'sil_value']))
                #get binary of sig sil values to use as mask i.e. index to remove cell with low sil values from respective assemblies
            
            reordered_cluster_amp_mask = np.where(np.array(reordered_cluster_amp2)>0.95000000,1,0)
            reordered_cluster_amp_optimized = reordered_cluster_amp[reordered_cluster_amp_mask>0]        
            
            data_corrm_optimized = np.corrcoef(data2_tp2_optimized, rowvar=True)  #normalized covariance matrix = correlation matrix
            data_corrm = data_corrm_optimized
            
            data2_tp = data2_tp_optimized
            
            participation_binary['cluster'] = k_labels
            participation_binary_assembly= pd.DataFrame(participation_binary)
            popevent_contr = participation_binary_assembly.groupby('cluster').sum()
            
            participation_binary_assembly['cluster'] = k_labels
            participation_binary_assembly_optimized = participation_binary_assembly[sil_value_mask_array>0] #optimized cluster based on sil values
            

            popevent_contr_optimized = participation_binary_assembly_optimized.groupby('cluster').sum() #optimized cluster based on sil values


            #optimize the shuffled participation vector and turn it into the same structure as original 
            shuffled_array_participation_optimized = []
            for k in np.arange(0,self.shuffle_num,1): #of shuffles
                shuffled_array_participation_optimized2 = []
                shuffled_array_participation_optimized3 = np.array(shuffled_array_participation[k])
                inter = pd.DataFrame(shuffled_array_participation_optimized3[sil_value_mask_array>0])
                inter = inter.values.tolist()
                shuffled_array_participation_optimized.append(inter)
            shuffled_array_participation= shuffled_array_participation_optimized


            #optimize the k_labels shuffled  vector and turn it into the same structure as original 
            k_labels_shuffle_optimized = []
            for k in np.arange(0,self.shuffle_num,1): #of shuffles
                k_labels_shuffle2 = []
                k_labels_shuffle3 = np.array(k_labels_shuffle[k])
                inter = pd.DataFrame(k_labels_shuffle3[sil_value_mask_array>0])
                inter = inter.values
                #inter = inter.values.tolist()
                k_labels_shuffle_optimized.append(inter)
            k_labels_shuffle  = k_labels_shuffle_optimized

            print('the optimized assembly spike freq is: ' + str(spike_freq_time_optimized))
                    
        
        else: 
            pass

    def assembly_contribution(self):
        global popevent_contr
        global cluster_contribution
        #####################test degree to which individual assemblies contribute to sync event######################
        #sum of spikes participation for each cluster
        participation_binary['cluster'] = k_labels
        participation_binary_assembly= pd.DataFrame(participation_binary)
        popevent_contr = participation_binary_assembly.groupby('cluster').sum()
        participation_binary_assembly['cluster'] = k_labels
        popevent_shuffle = []
        for j in np.arange(0,self.shuffle_num,1): #1000
            #print(j)
            inter_array = []
            for m in np.arange(0,len(shuffled_array3[1][1]),1): #200 
                #print(m)
                dic2 = {}
                dic2 = [item[m] for item in shuffled_array_participation[j]] #selects each sublist item in sequence using list comphrension
                inter_array.append(bltin_sum(dic2))#uses built in sum rather than numpy sum to create sum of spikes   
            popevent_shuffle.append(inter_array)      
            #output of this is the sum of spikes which is shuffle# x movie frames 
        

        hist_array_shuffle = []
        inter_array2 = []
        for m in np.arange(0,len(shuffled_array_participation[1][1]),1): #200 
            #print(m)
            dic3 = {}
            dic3 = [item[m] for item in popevent_shuffle]
            inter_array2.append(dic3)
        hist_array_shuffle.append(inter_array2) #this creates an list array of arbitrary supralist 0 --> movie frames x # shuffles
        #each sublist conists of all reshuffled spikes for one movie frame
        
        # statistical probability of observed event compared to 1000 reshuffled events, empirical cumulative density function
        # for each timepoint to determine assignment of synchronized calcium event to assembly
        #question answered with this code: which cluster (from kmeans) contributes sig. to each sig. sychronous event?
        popevent_prob_shuffle = []
        for k in np.arange(0,len(popevent_contr),1): #2
            inter2 = []
            for j in np.arange(0,len(data2_p),1): # 200
                popevent_prob_shuffle2 = []
                ecdf = ECDF(hist_array_shuffle[0][j]) # creates empirical cumulative density function based off shuffles for each frame
                interm = ecdf(popevent_contr.iloc[k,j]) # probability of observed event in each frame
                popevent_prob_shuffle2.append(interm)
                inter2.append(popevent_prob_shuffle2)
            popevent_prob_shuffle.append(inter2)
            
            #output = popevent_prob_shuffle 
            #each cluster is tested independently, does each clsuter contribute significantly (>95% cum. prob.) to the 
            # significant (99% cum. prob) sychronized population event? label those that do with a 1 and otherwise with 0
            #for each frame
        cluster_contribution = []
        for k in np.arange(0,len(popevent_contr),1): #2
            cluster_contribution2 = []
            cluster_contribution2 = np.where(np.array(popevent_prob_shuffle[k])>0.950000000,1,0)
            cluster_contribution.append(cluster_contribution2)
#
        for k in np.arange(0,optimal_cluster,1):
            print('cluster ' + str(k) + ' contributed to: ' + str(cluster_contribution[k].sum()) + ' synchronized events')
         
#
        # % single and % non single assembly synchronized calcium events
        #0 = no synchronized event, 1= one cluster contributed, 2 = two clusters contributed etc. 
        just = []
        just = cluster_contribution[0] + cluster_contribution[1]
        unique, counts = np.unique(just, return_counts=True)
        just2 = dict(zip(unique, counts))

        for k in np.arange(1,optimal_cluster+1,1):
            print('# synchronized population events where '+ str(k) + ' assembly contributed: ' + str(just2[k]))
          

        #sig contribution of each k cluster to sig. synch calcium events
        plt.figure()
        plt.plot(cluster_contribution[0], color='red', alpha= 0.7)
        plt.plot(popevent_binary, color='black', linestyle=':')
        #ax2.plot(popevent_binary, color='gray', linestyle=':')
        plt.ylabel('Contribution (95%) of cluster 1 to synchronized calcium events')
        plt.xlabel('Frames')
        #ax.set_title('Cell #21 and #41') 

        plt.figure()
        plt.ylabel('Contribution (95%) of cluster 2 to synchronized calcium events')
        plt.xlabel('Frames')
        plt.plot(cluster_contribution[1], color='green', alpha= 0.7)
        plt.plot(popevent_binary, color='black', linestyle=':')

        #ax.set_title('Cell #21 and #41')  
        plt.figure()
        #plot heatmap raster plot
        sb.heatmap(reordered_cluster_amp, yticklabels=reordered_cluster_amp.index)
        plt.axhline(y=np.count_nonzero(data2_tp['cluster']==0), color='red', linewidth=1) 
        plt.title('Activity')
        plt.xlabel("Frames")
        plt.ylabel("Cell #")
        plt.show()
        
        
    def spatial_analysis(self):   
        global shuffle_spatial_cluster
    ####################################analyze spatial data###################################################         
        spatial_dist = []
        for p in np.arange(0,2,1): # # of columns with spatial data (2), not including the cluster ID column
            spatial_dist3 = []    
            for j in np.arange(0,len(topo_centroids3),1): # # of cells
                test = (topo_centroids3.iloc[j,p] - topo_centroids3.iloc[:,p])**2 #diff between each coordinate and all other 
                spatial_dist3.append(test)
            spatial_dist.append(spatial_dist3)


        spatial_dist_euclid = []
        spatial_dist_pythag = []
        
        for h in np.arange(0,len(spatial_dist[0]),1): # of cells
            pythag = spatial_dist[0][h] + spatial_dist[1][h] #0,0,0 + 0,1,0
            spatial_dist_pythag.append(pythag)
            inter6 =pd.DataFrame(spatial_dist_pythag) #this creates a squared spatial distance matrix
        spatial_dist_euclid = inter6.transform('sqrt') #gets euclidean distance between cells in matrix
        #output = spatial_dist_euclid , gives distance matrix between all cells
            
        spatial_sil = sklearn.metrics.silhouette_samples(spatial_dist_euclid,topo_centroids3.iloc[:,2])
        spatial_sil = pd.DataFrame(spatial_sil)
            
        spatial_sil['cluster'] = k_labels
        spatial_sil_cluster = spatial_sil.groupby('cluster')[0].mean()
        
        for k in np.arange(0,optimal_cluster,1):
            print('Spatial Silhouette value for cluster ' + str(k) +  ' is '  + str(spatial_sil_cluster[k]))
            
            shuffle_spatial = []
            shuffle_spatial_reorder = []
        for j in np.arange(0,self.shuffle_num,1): # #of reshuffles to do, reshuffles each cells events across time
            shuffle_spatial2= []
            inter9 = sklearn.metrics.silhouette_samples(spatial_dist_euclid,k_labels_shuffle[j]) #gets new sil value using shuffled k labels  
            inter9 = pd.DataFrame(inter9)
            inter9['cluster'] = k_labels_shuffle[j]
            inter10 = inter9.groupby('cluster')[0].mean()  #gets mean sil value for each spatial cluster after reshuffling k mean labels
            shuffle_spatial.append(inter10) #
        
            shuffle_spatial_reorg= []
        for k in np.arange(0,len(shuffle_spatial[0]),1):
            shuffle_spatial_reorg2= []
            shuffle_spatial_reorg2 = [item[k] for item in shuffle_spatial] # reorganizes list so that parent list are the clusters
            shuffle_spatial_reorg.append(shuffle_spatial_reorg2)
                
        shuffle_spatial_cluster = []
        for j in np.arange(0,len(shuffle_spatial_reorg),1): # of clusters
            shuffle_spatial_cluster2 = []
            ecdf = ECDF(shuffle_spatial_reorg[j]) # creates empirical cumulative density function based off shuffles for each cell
            interm = ecdf(spatial_sil_cluster[j]) # probability of observed event in each frame
            shuffle_spatial_cluster.append(interm)
    
        for k in range(0,optimal_cluster,1):
            print('The cumulative prob of spatial clustering in assembly 1 is: ' + str(shuffle_spatial_cluster[k]))
     
        
    
        #plot spatial distribution
        #plot spatial with k means and average spike frequency
        t = ggplot(topo_centroids3) + geom_point(aes(x = 'x',y='y', size = 2, color = spike_freq_time, shape = 'factor(k_labels[0])')) 
        t + scale_color_gradient(low='blue', high='red') + theme_classic()
        t
        #plot basic topography with kmeans
        plt.figure()
        plt.scatter(topo_centroids3.iloc[:,0], topo_centroids3.iloc[:,1],  marker='o', c = k_labels.iloc[:,0],)
        plt.legend(loc='upper right')
        plt.title('topography')
        plt.xlabel("x pixels")
        plt.ylabel("y pixels")
        plt.show()
        #plt.tight_layout(pad=1)
        
        ############
        plt.figure()
        plt.hist(clusters2[0])
        plt.axvline(x=sil_score_bycluster[0] , color='red')
        plt.title('Cluster 1 cumulative probability after 1000 shuffles') 

        plt.figure()
        plt.hist(clusters2[1])
        plt.axvline(x=sil_score_bycluster[1] , color='red')
        plt.title('Cluster 2 cumulative probability after 1000 shuffles') 
        plt.show()
        

       
    def network_analysis(self, plot_network, corr_threshold, write_to, write_to_dir, file_name,dynamic_network_plot):
        ####################################analyze network activity##############################################


        data_corrm2 = pd.DataFrame(data_corrm)

        data_corrm2 = data_corrm2.stack().reset_index()

        data_corrm2.columns = ['var1', 'var2','value']
        data_corrm2_filtered=data_corrm2.loc[ (data_corrm2['value'] > corr_threshold) & (data_corrm2['var1'] != data_corrm2['var2']) ] #sets threshold for correlation value and removes self correlations
        data_corrm2_filteredxx = data_corrm2_filtered.reset_index(drop=True)
        ##
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
        eigenvector_dict = nx.eigenvector_centrality(G) # Run eigenvector centrality
        betweenness_dict = nx.betweenness_centrality(G) # Run betweenness centrality


        nx.draw_networkx(G, 
                         pos = node_pos, 
                         node_size= degree2,
                         node_color = k_labels_list,
                         alpha = 0.9,
                         with_labels=True,
                         edge_color=labels3,
                         edge_cmap=plt.cm.jet,
                         width = labels4,
                         font_color = 'black')
    
        #colorbar for edge weights
        cmap = mpl.cm.jet
        norm = mpl.colors.Normalize(vmin=min(labels3), vmax=max(labels3))
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
       
        
        
        
        
        if plot_network == 'true':
            
            plt.show()
        else:
            pass
        
        if write_to == 'true':
            os.chdir(write_to_dir)
            #nx.write_gexf(G,file_name) # need to modify this to work again
            #nx.write_edgelist(G,'edgelist.csv') # this is for gephi too
            #topo_centroids3.to_csv('topo_centroids3.csv') also for gephi
        else:
            pass


        if dynamic_network_plot == 'true':
            
            def simple_update(doodoo, G, ax,):
                ax.clear()
                colormin = 0
                colormax = 1
                corr_range = 0.1 * doodoo
                corr_range = round(corr_range,2)
                data_corrm2_filtered=data_corrm2.loc[ (data_corrm2['value'] > corr_range) & (data_corrm2['var1'] != data_corrm2['var2']) ] #sets threshold for correlation value and removes self correlations
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
                degree2 = [x*20 for x in degree]
                k_labels_list = k_labels[0].values.tolist()
    

                nx.draw_networkx(G, 
                         pos = node_pos, 
                         #node_size= degree2,
                         node_color = k_labels_list,
                         node_size = degree2,
                         alpha = 0.8,
                         with_labels=True,
                         edge_color=labels3,
                         edge_cmap=plt.cm.jet,
                         edge_vmin= colormin,
                         edge_vmax= colormax,
                         width = labels4,
                         font_color = 'black')

                # Set the title
    
                ax.set_title("Correlation > {}".format(corr_range))


            def simple_animation():

                # Build plot
                fig, ax = plt.subplots()
                colormin = 0 #min(labels3)
                colormax = 1 #max(labels3)
                cmap2 = mpl.cm.jet
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap2))


                ani = animation.FuncAnimation(fig, simple_update, frames=10, interval = 1000, fargs=(G, ax))
                ani.save(file_name, writer='imagemagick')

                #plt.show()

            simple_animation()
    
        else:
            pass




        #print(nx.info(G))

        density = nx.density(G)
        print("Network density:", density)
        # counts the number of edges, an edge is a connection
        degree_dict = dict(G.degree(G.nodes()))
        nx.set_node_attributes(G, degree_dict, 'degree')
        sorted_degree = sorted(degree_dict.items(), key=itemgetter(1), reverse=True)
        print("Top 20 nodes by degree:")
        for d in sorted_degree[:20]:
            print(d)
        #eigenvector centrality (like degree, but cares about how many hubs connected to)
        #and betweeness centrality (broker, shortest path between nodes, least path nodes)
        betweenness_dict = nx.betweenness_centrality(G) # Run betweenness centrality
        eigenvector_dict = nx.eigenvector_centrality(G) # Run eigenvector centrality

        # Assign each to an attribute in your network
        nx.set_node_attributes(G, betweenness_dict, 'betweenness')
        nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')
        #
        sorted_betweenness = sorted(betweenness_dict.items(), key=itemgetter(1), reverse=True)

        print("Top 20 nodes by betweenness centrality:")
        for b in sorted_betweenness[:20]:
            print(b)
            #
        sorted_betweenness = sorted(eigenvector_dict.items(), key=itemgetter(1), reverse=True)
            
        print("Top 20 nodes by eigenvector centrality:")
        for b in sorted_betweenness[:20]:
            print(b)
            #
        communities = community.greedy_modularity_communities(G)
        modularity_dict = {} # Create a blank dictionary
        for i,c in enumerate(communities): # Loop through the list of communities, keeping track of the number for the community
            for name in c: # Loop through each person in a community
                modularity_dict[name] = i # Create an entry in the dictionary for the person, where the value is which group they belong to.
            # Now you can add modularity information like we did the other metrics
        nx.set_node_attributes(G, modularity_dict, 'modularity')
        # First get a list of just the nodes in that class
        class0 = [n for n in G.nodes() if G.nodes[n]['modularity'] == 0]
    
        # Then create a dictionary of the eigenvector centralities of those nodes
        class0_eigenvector = {n:G.nodes[n]['eigenvector'] for n in class0}

        # Then sort that dictionary and print the first 5 results
        class0_sorted_by_eigenvector = sorted(class0_eigenvector.items(), key=itemgetter(1), reverse=True)

        print("Modularity Class 0 Sorted by Eigenvector Centrality:")
        for node in class0_sorted_by_eigenvector[:5]:
            print("Name:", node[0], "| Eigenvector Centrality:", node[1])
            #
            for i,c in enumerate(communities): # Loop through the list of communities
                if len(c) > 2: # Filter out modularity classes with 2 or fewer nodes
                    print('Class '+str(i)+':', list(c)) # Print out the classes and their members    



    def summary_plots(self, t_sne_perplex, plot_1,plot_2,plot_3,plot_4):
        global participation_binary_assembly
        if plot_1 == 'true':
                #for plotting.

            cell_contribute_perc_dfindex = cell_contribute_perc_df.reset_index() #turn index into column
            cell_contribute_perc_dfindex['cluster'] = k_labels
            #this gives the actual index numbers for each cluster
            cell_contribute_perc_dfindex_sort = cell_contribute_perc_dfindex.index[cell_contribute_perc_dfindex['cluster'] == 0].tolist() 
            cell_contribute_perc_dfindex_sort2 = cell_contribute_perc_dfindex.index[cell_contribute_perc_dfindex['cluster'] == 1].tolist() 
            
            cell_contribute_perc_dfindex_sort = pd.DataFrame(cell_contribute_perc_dfindex_sort)
            cell_contribute_perc_dfindex_sort2 = pd.DataFrame(cell_contribute_perc_dfindex_sort2)
            
            reordered_cell_contribute = [cell_contribute_perc_dfindex_sort, cell_contribute_perc_dfindex_sort2]
            reordered_cell_contribute = pd.concat(reordered_cell_contribute) #this is the indexes in the right order
            
            reordered_cell_contribute2 = cell_contribute_perc_df.reindex(reordered_cell_contribute.iloc[:,0])
            reordered_cell_contribute2['index'] = reordered_cell_contribute2.index #not sure why i cant reindex here, do it manually

            ################################################plot########################################
            #print(plt.style.available)
            #plot raw trace df/f decon
            plt.style.use('default')
        
            plt.figure()
            ax = plt.subplot(2,2,1) 
            ax2 = ax.twinx()
            ax.plot(data2.iloc[:,21], color='black')
            ax.plot(data2.iloc[:,41], color='blue')
            ax2.plot(popevent_binary, color='gray', linestyle=':')
            ax.set_ylabel('Deconvoluted df/f')
            ax.set_xlabel('Frames')
            ax.set_title('Cell #21 and #41') 
            plt.legend(loc='upper right')


            plt.subplot(2,2,2)
            #plot tsne here
            data2_tsne = data2_tp.drop(['cluster','sil_value'], axis=1)
            pca = PCA(n_components=10)
            pca_result = pca.fit_transform(data2_tsne)
            pca_one = pca_result[:,0]
            pca_two = pca_result[:,1]
            print(np.sum(pca.explained_variance_ratio_)) #cumulative explained variance
            colors = k_labels.iloc[:,0].tolist()
            #plt.scatter( x=pca_one, y=pca_two,c=colors, alpha=0.8)
            tsne = TSNE(n_components=2, verbose=1, perplexity=t_sne_perplex, n_iter=5000)
            tsne_results = tsne.fit_transform(pca_result)
            tsne_2d_one = tsne_results[:,0]
            tsne_2d_two = tsne_results[:,1]
            plt.ylabel('tSNE1')
            plt.xlabel('tSNE2')
            plt.title('tSNE')
            plt.scatter(x=tsne_2d_two, y=tsne_2d_one,c=colors)


            #plot cell contribution
            plt.subplot(2,2,3)
            plt.bar(x=reordered_cell_contribute2.loc[:,'index'], height= reordered_cell_contribute2.iloc[:,0], color='blue')
            plt.ylabel('% Contribution to All Events')
            plt.xlabel('Cell #')    
            plt.axvline(x=np.count_nonzero(data2_tp['cluster']==0), color='red', linewidth=1)  #this will plot red line based on size of cluster 1, because indexes are screwed in bar plot or something
            #plt.set_xticklabels(cell_contribute_perc_df_sort_index)
            plt.title('Contribution of Each Cell to Synchronous Calcium Events')

            #plot spatial distribution
            plt.subplot(2,2,4)
            plt.scatter(topo_centroids3.iloc[:,0], topo_centroids3.iloc[:,1],  marker='o', c = k_labels.iloc[:,0],)
            plt.legend(loc='upper right')
            plt.title('topography')
            plt.xlabel("x pixels")
            plt.ylabel("y pixels")
            #plt.tight_layout(pad=1)
           
            plt.subplots_adjust(left=0.09, bottom=0.02, right=0.7, top=0.80, wspace=0.4, hspace=0.5)
            plt.show()
            
        else: 
            pass
        
        if plot_2== 'true':
            plt.figure()
            #plot heatmap raster plot
            sb.heatmap(reordered_cluster_amp, yticklabels=reordered_cluster_amp.index)
            plt.axhline(y=np.count_nonzero(data2_tp['cluster']==0), color='red', linewidth=1) 
            plt.title('Activity')
            plt.xlabel("Frames")
            plt.ylabel("Cell #")
            plt.show()
            
            
            plt.figure()
            t = ggplot(topo_centroids3) + geom_point(aes(x = 'x',y='y', size = 2, color = spike_freq_time, shape = 'factor(k_labels[0])')) 
            t + scale_color_gradient(low='blue', high='red') + theme_classic()
            t
            #
            #topo_centroids2[(topo_centroids2['cluster']==1)].iloc[:,0]

            plt.figure()
            #sig contribution of each k cluster to sig. synch calcium events
            ax = plt.subplot(2,1,1) 
            ax.plot(cluster_contribution[0], color='red', alpha= 0.7)
            ax.plot(popevent_binary, color='black', linestyle=':')
            #ax2.plot(popevent_binary, color='gray', linestyle=':')
            ax.set_ylabel('Contribution (95%) of each cluster to synchronized calcium events')
            ax.set_xlabel('Frames')
            #ax.set_title('Cell #21 and #41') 

            #
            ax = plt.subplot(2,1,2) 
            ax.set_ylabel('Contribution (95%) of each cluster to synchronized calcium events')
            ax.set_xlabel('Frames')
            ax.plot(cluster_contribution[1], color='green', alpha= 0.7)
            ax.plot(popevent_binary, color='black', linestyle=':')
        
            plt.show()
        #
        else: 
            pass

        if plot_3 == 'true':
            plt.figure()
            #plot raw trace df/f decon
            plt.subplot(3,2,1) 
            plt.plot(data2.iloc[:,1], color='black')
            plt.plot(data2.iloc[:,7], color='blue')
            plt.ylabel('Deconvoluted df/f')
            plt.xlabel('Frames')
            plt.title('Cell #1 and #7') 
            plt.legend(loc='upper right')
          
            #plot hist frame 1          
            plt.subplot(3,2,2)
            plt.hist(hist_array[0][175], alpha=0.5, label='hist_array', color='gray')
            plt.ylabel('Count')
            plt.xlabel('Spikes')
            plt.axvline(x=sum_o_spikes_array[0][175] , color='red')
            plt.title('Frame #175') 
            plt.legend(loc='upper right')
          
            #plot hist frame 2
            plt.subplot(3,2,3)
            plt.hist(hist_array[0][20], alpha=0.5, label='hist_array', color='gray')
            plt.ylabel('Count')
            plt.xlabel('Spikes')
            plt.axvline(x=sum_o_spikes_array[0][20], color='red')
            plt.title('Frame #20') 
            plt.legend(loc='upper right')

            #plot synch calcium event prob
            plt.subplot(3,2,4)
            plt.plot(popevent_prob,color='gray')
            plt.ylabel('Synchronous Calcium Event Probability')
            plt.xlabel('Frames')
            plt.title('All Cells')
            plt.axhline(y=0.99, color='red', linestyle=':')     

            #plot binary pop events
            plt.subplot(3,2,5)
            plt.plot(popevent_binary, color='gray')
            plt.ylabel('Binary Population Events')
            plt.xlabel('Frames')
            plt.title('All Cells')
            plt.show()

            #plot cell contribution
            plt.subplot(3,2,6)
            plt.plot(cell_contribute_perc, color='blue')
            plt.ylabel('% Contribution to All Events')
            plt.xlabel('Cell #')
            #plt.set_xticklabels(cell_contribute_perc_df_sort_index)
            plt.title('Contribution of Each Cell to Synchronous Calcium Event')

            #plt.tight_layout(pad=2)
            plt.subplots_adjust(left=0.09, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
            plt.show()
            
            
        else:
            pass

        if plot_4 == 'true':
            plt.figure()
            plt.subplot(3,2,1)
            plt.plot(data_corrm[:,1], color='blue')
            plt.plot(data_corrm[:,2], color='blue')
            plt.plot(data_corrm[:,7], color='red', alpha=0.4) 
            plt.plot(data_corrm[:,10], color='red', alpha=0.4)

            plt.title('correlation of Cell X with all other cells')
            plt.ylabel('Correlation Coefficient')
            plt.xlabel('Cell #')
           

            plt.subplot(3,2,2)
            plt.plot(dist_sq_euc[:,1], color='blue')
            plt.plot(dist_sq_euc[:,2], color='blue')
            plt.plot(dist_sq_euc[:,7], color='red', alpha=0.4)
            plt.plot(dist_sq_euc[:,10], color='red', alpha=0.4)


            plt.title('similarity metric')
            plt.ylabel('sq euclidean distance')
            plt.xlabel('Cell #')
 
            plt.subplot(3,2,3) 
            plt.plot(data2.iloc[:,1], color='black')
            plt.plot(data2.iloc[:,7], color='blue')
            plt.ylabel('Deconvoluted df/f')
            plt.xlabel('Frames')
            plt.title('Cell #1 and #7') 
            plt.legend(loc='upper right')

            plt.subplot(3,2,4) 
            plt.plot(data2.iloc[:,7], color='black')
            plt.plot(data2.iloc[:,10], color='blue')
            plt.ylabel('Deconvoluted df/f')
            plt.xlabel('Frames')
            plt.title('Cell #7 and #10') 
            plt.legend(loc='upper right')

            #plot the significance of the clusters
            plt.subplot(3,2,5)
            plt.hist(clusters2[0])
            plt.axvline(x=sil_score_bycluster[0] , color='red')
            plt.title('Cluster 1 cumulative probability after 1000 shuffles') 

            plt.subplot(3,2,6)
            plt.hist(clusters2[1])
            plt.axvline(x=sil_score_bycluster[1] , color='red')
            plt.title('Cluster 2 cumulative probability after 1000 shuffles') 
            
            plt.show()
            
        else:
            pass
