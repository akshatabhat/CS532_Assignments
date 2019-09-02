%Question 4
[data, TEXT, raw]  = xlsread('nutrients.xlsx');
food_items = TEXT(2:end, 1:2);
X = data(:,1:end-1);

disp('4a)');
figure(1);
J = [];
cluster_centres = [];
for k=2:10
    [idx, C, sumd] = kmeans(X, k, 'Start', 'sample', 'Replicates',100, 'MaxIter',200);
    J = [J; sum(sumd)];
    cluster_centres = [cluster_centres ; C];
end
plot(2:10, J, '-o');
xlabel('k');
ylabel('J');
title('Elbow Pl0t')
snapnow;
disp('Based on the elbow plot, I would consider K=4, 5 or 6 as the elbow');

disp('4b)')
figure(2);
i=1;
for k=3:7
    subplot(2, 3, i);
    [idx, C] = kmeans(X, k, 'Start', 'sample', 'Replicates',100, 'MaxIter',200);
    [s, h] = silhouette(X, idx);
    title(strcat('k=',int2str(k)));
    fprintf("K=%d , Silhouette Score=%f \n", k, sum(s)/size(s, 1));
    i = i+ 1;
end
snapnow;

disp("4c)")
start_idx = 1;
for k=2:10
    indices = [];
    centroids = cluster_centres(start_idx: start_idx+k-1, :);
    for c=1:size(centroids,1)
        [~, idx] = min(vecnorm(X-centroids(c,:),2,2));
        indices = [indices; idx];
    end
    fprintf("K=%d, Food Representatives : \n", k);
    disp(food_items(indices,:));
    start_idx = start_idx +k;
end

disp('4d)');

disp(' From elbow plot, the conclusion was the choose K=4, 5, or 6.');
disp('From the silhouette values, though the average value is maximum when K=3, clusters don’t seem to be well distributed');
disp('- most of the items are in cluster 1 and extremely few in cluster 3. Similar observation with K=4 and K=5.'); 
disp('For K=6 and K=7,  the cluster size distribution seems better.');
disp('However, by looking at the average silhouette score for varying K, the score for K=7 is much lower.'); 

disp(' From food domain knowledge, we can see that if we take K=6, we have food items from 6 different groups.'); 
disp('(Whereas for K=4, 5 or 7, some food groups are repeated). ');

disp('Hence, I think we should select K=6.');
 
