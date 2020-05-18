%run this after loading the .mat file from ezcalcium into the workspace.
%to save x and y pixel coordinate data into a csv file. 

%create empty arrays
xcoord = NaN(1,size(A_or,2))
ycoord = NaN(1,size(A_or,2))

for i = 1:size(A_or,2)
    %convert from 1d to 2d
    single_roi = reshape(A_or(:,i), size(Cn,1), size(Cn,2))
    %calculate weighted centroids for all roi
    centroid = regionprops(true(size(single_roi)), single_roi, 'WeightedCentroid');  
    xcoord(i) = centroid.WeightedCentroid(:,1) %extract weighted centroid x and y from 1x1 str
    ycoord(i) = centroid.WeightedCentroid(:,2) %extract weighted centroid x and y from 1x1 str 
end

topo = rot90([xcoord;ycoord])
topo= rot90(topo)
topo=rot90(topo)
writematrix(topo,"topo_centroids.csv")
