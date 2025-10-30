function metrics = evaluate_clustering(labels_true, labels_pred, q)

labels_pure = labels_pred;
p = length(labels_true);
for j = 1:q
    idx = labels_pred == j;
    labels_pure(idx) = mode(labels_true(idx));
end

mask = labels_pure == labels_true;
purity = sum(mask)/p;




labels_pred_adj = labels_pred;
perm_mat = perms([1:q]);
acc_max = 0;
Nperms = size(perm_mat,1);

for k = 1:Nperms
    perm = perm_mat(k,:);
    for j = 1:q
        idx = labels_pred == j;
        labels_pred_adj(idx) = perm(j);
    end

    mask = labels_pred_adj == labels_true;
    acc = sum(mask)/p;
    if (acc>= acc_max)
        acc_max = acc;
        ind_max = k;
    end
end
           
perm = perm_mat(ind_max,:);
for j = 1:q
    idx = labels_pred == j;
    labels_pred_adj(idx) = perm(j);           
end

mask = labels_pred_adj == labels_true;
accuracy_adj_metric = sum(mask)/p;

labels_pred_sorted = sort(labels_pred, 'descend');
k = labels_pred_sorted(1);
clust_size = zeros(1,k);
for i = 1:k
    clust_size(i) = sum(labels_pred == i);
end
if (k<q)
    
    clust_size = [zeros(1,q-k), clust_size];
end
num =  sum(abs(clust_size - p/q)); 

clust_size = [zeros(1,q-1), p-(q-1)]; 
denum =  sum(abs(clust_size - p/q)); 
balanced_norm = num/denum;

NMI = normalized_mutual_information(labels_pred_adj, labels_true);
[RI, ARI] = randindex(labels_pred_adj, labels_true);



metrics.purity = purity;
metrics.accuracy_adj_metric = accuracy_adj_metric;
metrics.balanced = balanced_norm;
metrics.NMI = NMI;
metrics.RI = RI;
metrics.ARI = ARI;

end