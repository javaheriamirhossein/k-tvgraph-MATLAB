addpath(genpath('D:\Sharif\Project\Code\Source\gspbox-0.7.5'))
addpath(genpath(cd))



maindir = cd;
newdir = [maindir, '\Demo_results\'];
if ~isfolder(newdir)
    mkdir(newdir)
end


rng(1)
close all
stock_prices_orig = load('SP500_xts_2014_2024.mat').Data;
sectors = load('SP500_505_sectors.mat').sectors;


selected_sectors = ["Consumer Staples", "Energy", "Financials",...
                      "Health Care","Industrials",...
                      "Materials", "Real Estate", "Utilities"];

all_sectors = string(sectors.GICSSector);
all_symbols = sectors.Symbol;
exlude_symb_idx = ~matches(all_sectors,selected_sectors);
exlude_symbols = all_symbols(exlude_symb_idx);
inclde_symbols = all_symbols(~exlude_symb_idx);


all_stocknames = string(stock_prices_orig.Properties.VariableNames)';
include_symb_idx = ~matches(all_stocknames,exlude_symbols);
stock_prices_orig = stock_prices_orig(:,include_symb_idx);
N_tot = size(stock_prices_orig,2);


q = length(selected_sectors); 
p = 100;
K = 1;   % number of clusters: choose 1 for spectral clustering
rho = 1;
nu = 4;
sr = 1;
std_n = 0;
W_thr = 0;

Ntrial = 20; % number of algorithms
N_alg = 2; % number of algorithms

winLen =  200;
T = 1000;
frame_len = winLen;
N_f = fix(T/frame_len);





purity = zeros(N_f,N_alg,Ntrial);
accuracy = zeros(N_f,N_alg,Ntrial);
balanced = zeros(N_f,N_alg,Ntrial);
NMI = zeros(N_f,N_alg,Ntrial);
ARI = zeros(N_f,N_alg,Ntrial);
RI = zeros(N_f,N_alg,Ntrial);
MOD = zeros(N_f,N_alg,Ntrial);
Times = zeros(N_f,N_alg,Ntrial);


% Monte Carlo experiments
Nday = size(stock_prices_orig,1) ;
start_idx = randi(Nday-T,1,Ntrial);

for iter = 1:Ntrial
    stock_idx = randperm(N_tot, p);
    stock_idx = sort(stock_idx);
    start_id = start_idx(iter);
    stock_prices = stock_prices_orig(start_id:start_id+T,stock_idx);
    
    stocknames = stock_prices.Properties.VariableNames;
    
    idx = find(ismember(all_symbols,stocknames));
    
    true_labels = cellstr(sectors.GICSSector(idx));
    true_indices = grp2idx(true_labels);
    uniqe_labels = unique(true_indices);
    
    X = diff(log(table2array(stock_prices)));
    X = X';
    X = normalize(X,2);
    
    %% ===============================
    [N, T] = size(X);
    
    SampleNum = floor(N*sr); 
    SampleMatrix = zeros(N,T);
    for i = 1:T
        SampleMatrix(randperm(N, SampleNum),i) = 1;
    end 
    Mask = SampleMatrix;  % sampling mask
    
    
    noise = std_n * randn(size(X)); % measurement noise
    Y = Mask.*(X + noise);
     
  
    Y_f_cell = cell(1,N_f);
    Mask_cell = cell(1,N_f);
    
    for i = 1:N_f   
        Y_f = Y( :, ((i-1)*frame_len+1): (i*frame_len));
        Mask_f = Mask( :, ((i-1)*frame_len+1): (i*frame_len));
        Y_f_cell{i} = Y_f;
        Mask_cell{i} = Mask_f;
    end
    

    
    %% =========================
    L_cell = cell('');
    title_cell = cell('');
    times = cell('');
    
    
    Laplacian_cell_f = cell(N_f,1);
      
    
    %% Algorithms
    % ==========================================
    
    params = struct;
    params.frame_len = frame_len;
    params.maxiter = 100;
    params.sigma_e = exp(0.1);
    params.nu = nu;
    params.eta = 1e-8;
    params.update_eta = 1;
    params.rho = rho;
    params.K = K;
    params.d = 1;
    params.gamma = 0.1;
    params.std_n = 0;
    params.W_thr = W_thr;
    
    [w_cell, a_var, Xhat, times_f] = learn_kcomp_heavytail_tv_graph(Y, Mask, params);
    times{1} = times_f;
    
    for k=1:N_f
          Laplacian = L_operator_mex(w_cell{k});
          Laplacian_cell_f{k} = Laplacian;
    end
    L_cell{1} = Laplacian_cell_f;
    rank(Laplacian_cell_f{end})
    title_cell{1} = 'kTVGL';
    

    %-------------------------------------------------------
    method = 1;
    if method == 1
        s = sqrt(2*(N-1))/2 / 3;
    else
        s = 1/2/sqrt(2);
    end
    
    params = struct;
    
    t0 = tic;
    times_f = zeros(N_f,1);
    for k=1:N_f
        Y_data = Y_f_cell{k};
        Z = gsp_distanz(Y_data').^2;
        theta = gsp_compute_graph_learning_theta(Z,8);
        W = gsp_learn_graph_l2_degrees(theta *Z, 1, params);
        Laplacian = diag(sum(W,2)) - W;
        Laplacian_cell_f{k} = Laplacian;
        params.w_0 = w_from_L(Laplacian);    
        params.c = 10;
        times_f(k) = toc(t0);
    end
    times{end+1} = times_f;
    L_cell{end+1} = Laplacian_cell_f;
    title_cell{end+1} = 'GSPBOX-L2';
    


    

    %% Evaluate clustering
    
    for i = 1:N_alg
        Laplacian_cell_f = L_cell{i};
              
        for k=1:N_f  
            Laplacian = Laplacian_cell_f{k};
    
            % fname = ['Laplacian_',num2str(i),'_',num2str(k),'.mat'];
            % save(fname,'Laplacian');

            W = -Laplacian;
            W(1:N+1:end) = 0;
            W(W<W_thr*max(max(W))) = 0;
            Laplacian = diag(sum(W,2))-W;

            % spectral clustering
            [U,S] = eig(0.5*(Laplacian+Laplacian'),'vector');
            U_q = U(:,1:q);
            [indices, C] = kmeans(U_q, q, 'Replicates', 10, 'Start', 'plus');
            metrics = evaluate_clustering(true_indices, indices, q);

            accuracy(k,i,iter) = metrics.accuracy_adj_metric;
            purity(k,i,iter) = metrics.purity;
            balanced(k,i,iter) = metrics.balanced;
            NMI(k,i,iter) = metrics.NMI;
            ARI(k,i,iter) = metrics.ARI;
            RI(k,i,iter) = metrics.RI;
    
            [Mod, Mod_vec] = modularity(W, true_indices);
            MOD(k,i,iter) = Mod;

            Times(k,i,iter) = times{i}(k);
        end

  
    end


end

Times_padded = zeros(N_f+1,N_alg,Ntrial);
Times_padded(2:end,:,:) = Times;
Times_diff = diff(Times_padded);


%% Plot and save the results

Results = struct;
Results.purity = purity;
Results.accuracy = accuracy;
Results.balanced = balanced;
Results.NMI = NMI;
Results.ARI = ARI;
Results.RI = RI;
Results.MOD = MOD; 
Results.Times = Times_diff; 

field_names = fieldnames(Results);


N_figs = length(field_names);
ylabls = {'Purity', 'Accuracy', 'Balancedness', 'NMI', 'ARI', 'RI', 'MOD', 'Time'};

for k=1:N_figs
    figure(k);
    
    figname = field_names{k};
    mean_results = squeeze(mean(Results.(field_names{k}),1));
    boxplot(mean_results', 'Labels', title_cell)
    ylabel(ylabls{k})
    

    set(gcf, 'Position', [100 100 500 400])   
    title(ylabls{k}, 'interpreter', 'latex','FontSize',15)  
    
    
    figsavename = [newdir, figname, '.fig'];
    savefig(gcf,figsavename);
    % figsavename = [newdir, figname, '.png'];
    % saveas(gcf,figsavename);
    close(gcf)

end


save([newdir, 'data_TVGL_clustering_',num2str(frame_len),'.mat'], 'L_cell', 'title_cell', ...
                                                           'accuracy', 'purity', 'balanced', ...
                                                           'ARI', 'RI', 'MOD', 'times');

