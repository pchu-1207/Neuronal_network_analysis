# Neuronal_network_analysis_0.1
Scripts to analyze neuronal network activity in Python based on Duan, Che, Chu et al., 2019 Neuron (https://www.sciencedirect.com/science/article/pii/S0896627319308566?via%3Dihub). Strategy to detect neuronal assemblies used here was developed in the Cossart Lab (http://www.inmed.fr/en/developpement-des-microcircuits-gabaergiques-corticaux-en)

What does this code do?


-Detects activity clusters using K-means cluster analysis. validates using silhouette analysis and bootstrap reshuffling for statistical thresholds. 

-calculates activity frequency over time

-calculates sychronized calcium events and tests statistically using bootstrap reshuffling (1000x) using 95% threshold of empirical cumulative distribution (used as threshold for all bootstrap reshuffling tests)

-computes contribution of each cluster to statistically determined synchronized calcium events using bootstrap reshuffling

-computes the statistical signficance of spatial clustering using silhouette analysis and bootstrap reshuffling

-optimizes K-means detected clusters using bootstrap reshuffling to exclude detected neurons with low silhouette values relative to boostrap reshuffled

-plots spatial data and labels based on activity frequency

-plots spatial data with k-mean clusters analysis labels

-plots t-SNE 

-plots raster heatmap of activity 


![Heatmap of deconvolved df/f activity over time](Figure_1.png)
Fig 1. Binary raster plot displaying sychronized firing of interneuron activity in P7 pup in somatosensory cortex over time. Imaged using 2-photon microscopy and GCaMP6s. Cluster labels and silhouette values displayed as well.

-Requires inputs in spreadsheet format (.csv) with rows representing cells and columns representing time (movie frames). easily obtainable without programming experience from 2p calcium imaging movies analyzed with ezcalcium (https://porteralab.dgsom.ucla.edu/pages/matlab). 

-.mat script included allows extraction of spatial coordinates of each cell from matlab after ezcalcium run (using weighted centroids as location). outputs into a .csv file accepted by the included script.

Instructions:
-Put all files into same folder and run calcium_imaging_module_run. set the directories and other settings as desired.

-network_plot_plotly_interactive plots a sliding bar that scales the correlation coefficient between the cells in the network. Can be run in isolation. Useful for visualizing subnetworks.  

![Neuronal network analysis of 2-photon imaging P7 mouse expressing GCaMP-6s](test.gif)

Fig2: Spatially accurate plot of 2-photon imaged GCaMP6s labeled interneurons in P7 mouse somatosensory cortex. Size of circles represent number of connections. Color of circles represent the k-means cluster. Color of lines represent correlation coefficient of activity between pairs of neurons. gif displays varying correlation coefficient thresholds.

-network_plot_plotly_activityflow plots activity over time using a sliding bar. also useful for visualizing subnetworks and activity flow. requires calcium_imaging_module_run be run first. 


