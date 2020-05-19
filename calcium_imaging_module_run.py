import os
os.chdir("C:/Users/pchu1207/Google Drive/Data/Alexander/Code")
#from calcium_imaging_analysis_assemblies_synch import analyze_ca
import calcium_imaging_analysis_assemblies_synch as run
#%%
test = run.analyze_ca(file_directory= "//ucdenver.pvt/som/NEUROSURG/N_ LAB - Alexander/Data/calcium imaging/calcium imaging sample movie",
                  shuffle_num= 100, #minimum 1000
                  scale_factor_time = 1, #pixel/micron value
                  scale_factor_spatial = 1, #time (sec) / frame
                  activity_data_file = 'deconv.csv', 
                  spatial_data_file = 'topo_centroids.csv', 
                  optimization_choice = 'no') #whether to optimize kmeans cluster more tightly or not 'no','yes'
test.load_data() 
#%%
test.deconv_freq() #get freq

#%%
test.synch_network_events()
#%%
test.network_contributions()
    
#%%
test.kmeans_clustering()

#%%
test.cluster_optimization()
test.optimization_mod()
#%%
test.assembly_contribution()

#%%
test.spatial_analysis()

#%%
test.network_analysis(plot_network = 'true',
                      dynamic_network_plot = 'true',
                      corr_threshold = 0.75,
                      write_to = 'true',
                      write_to_dir = 'C:/Users/pchu1207/Desktop',
                      file_name = 'sample2.gif')
#%%
test.summary_plots(t_sne_perplex = 30, 
                   plot_1 = 'true', 
                   plot_2='true', 
                   plot_3='true', 
                   plot_4='true')
#%%



