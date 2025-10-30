addpath(genpath(cd))

% download the GSPBOX toolbox and put it in the main directory
addpath(genpath([cd,'\gspbox-0.7.5']))

maindir = cd;
newdir = [maindir, '\Demo results\Visual_tvgraphs\'];
if ~isfolder(newdir)
    mkdir(newdir)
end


clc
close all
rng(1);

frame_len = 200;
noise_type = 't';
Normalization = 'trace';  % Normalization: 'max','trace'
type = 2;                 % graph type
params_struct.type = type; 
params_struct.noise_type = noise_type;
params_struct.frame_len = frame_len;
params_struct.N = 100;
params_struct.T = 10*params_struct.N;
params_struct.Normalization = Normalization;

rho = 2;
K = 1;
d = 1;
nu = 4;
W_thr = 0.001;

sr = 1;
std_n = 0.0;


%% =======================
% Synthetic data 

data_struct = synthetic_data_TVG( params_struct );
X = data_struct.X;
W_N_true = data_struct.W_N_true;


[N,T] = size(X);
N_f = size(W_N_true,2);

SampleNum = floor(N*sr); % the number of sampled points at each time
SampleMatrix = zeros(N,T);
for i = 1:T
    SampleMatrix(randperm(N, SampleNum),i) = 1;
end 
Mask = SampleMatrix;


noise = std_n * randn(size(X)); % measurement noise
Y = SampleMatrix.*(X + noise);
 


% framing
Y_f_cell = cell(1,N_f);
Mask_cell = cell(1,N_f);
for i = 1:N_f   
    Y_f = Y( :, ((i-1)*frame_len+1): (i*frame_len));
    Mask_f = Mask( :, ((i-1)*frame_len+1): (i*frame_len));
    Y_f_cell{i} = Y_f;
    Mask_cell{i} = Mask_f;
end


%% =======================
% Results cells
L_cell = cell(1);
X_cell = cell(1);
title_cell = cell(1);
title_cell_X =  cell(1);
times = cell(1);


Laplacian_cell_f = cell(N_f,1);
X_cell_f = cell(N_f,1);




%% =======================
% Running algorithms

params = struct;
params.frame_len = frame_len;
params.maxiter = 100;
params.sigma_e =  exp(0.1);
params.nu = nu;
params.rho = rho;
params.K = K;
params.d = d;
params.gamma = 0.1;
params.std_n = std_n;

t0 = tic;
[w_cell, a_var, Xhat] = learn_kcomp_heavytail_tv_graph(Y, Mask, params);
times{1} = toc(t0);

for k=1:N_f
      Laplacian = L_operator_mex(w_cell{k});
      Laplacian_cell_f{k} = Laplacian;
end
L_cell{1} = Laplacian_cell_f;
X_cell{1} = Xhat;
rank(Laplacian_cell_f{end})
title_cell{1} = 'kTVGL';
title_cell_X{1} = 'kTVGL';


% -------------------------------------------------------
method = 1;
if method == 1
    s = sqrt(2*(N-1))/2 / 3;
else
    s = 1/2/sqrt(2);
end

params = struct;
t0 = tic;
for k=1:N_f
    Z = gsp_distanz(Y_f_cell{k}').^2;
    theta = gsp_compute_graph_learning_theta(Z,8);
    W = gsp_learn_graph_l2_degrees(theta *Z, 1, params);
    Laplacian = diag(sum(W,2)) - W;
    Laplacian_cell_f{k} = Laplacian;
    params.w_0 = w_from_L(Laplacian);
    params.c = 10;
end
times{end+1} = toc(t0);
W(W<1e-3) = 0;
G = gsp_graph(W / sum(sum(W)) * N);
G_temp = gsp_graph(W);
Laplacian = G_temp.L;
L_cell{end+1} = Laplacian_cell_f;
title_cell{end+1} = 'GSPBOX-L2';








%% Evaluation and visualization
N_pt = 400;


W_N_true_norm = W_N_true/max(max(W_N_true));
imagesc(W_N_true_norm(1:N_pt,:));
colorbar 
colormap hot
title('$W$ true', 'interpreter', 'latex')
saveTightFigure(gcf,[newdir, 'W_true','_',num2str(frame_len),'.pdf'])

N_x_Alg = length(title_cell_X);
X_rel_err = zeros(1,N_x_Alg);
for i = 1:N_x_Alg
    X_rel_err(i) = norm(X_cell{i} - X,'fro')/norm(X,'fro');
end



N_alg = length(L_cell);

RelativeEr = zeros(N_f,N_alg);
FDR = zeros(N_f,N_alg);
Fscore = zeros(N_f,N_alg);
TCER = zeros(N_f,N_alg);


for i = 1:N_alg
    Laplacian_cell_f = L_cell{i};
    
    W_N_hat = zeros(N_pt, N_f);
    for k=1:N_f

        w_true = W_N_true(:,k);
        Ltrue = L_operator_mex(w_true);
        if strcmp(Normalization, 'trace')==1
            Ltrue = Ltrue/trace(Ltrue)*N;
        elseif strcmp(Normalization, 'max')==1
            Ltrue = Ltrue/max(w_true);
        end
        A_mask = Ltrue<0;

        Laplacian = Laplacian_cell_f{k};
        Laplacian(1:N+1:end) = 0;
        W = abs(Laplacian);

        W = W/max(max(W));
        
        W(W<W_thr) = 0;   

        
        Laplacian = diag(sum(W,2))-W;   
        w = w_from_L(Laplacian);
        W_N_hat(:,k) = w(1:N_pt);
        
        if strcmp(Normalization, 'trace')==1
            Laplacian = Laplacian/trace(Laplacian)*N;
        elseif strcmp(Normalization, 'max')==1
            Laplacian = Laplacian/max(max(w));
        end

        Adjacency = Laplacian<0;

        RelativeEr(k,i) = norm(Laplacian-Ltrue,'fro')/norm(Ltrue,'fro');       
        FDR(k,i) = sum(sum(abs(Adjacency-A_mask)))/sum(sum(A_mask));
        tp = sum(sum(Adjacency.*A_mask));
        fp = sum(sum(Adjacency.*(~A_mask)));
        fn = sum(sum((~Adjacency).*A_mask));
        Fscore(k,i) = 2*tp/(2*tp+fn+fp);
    end
    


    figure
    imagesc(W_N_hat/max(max(W_N_hat)));
    colormap hot
    colorbar 
    title(title_cell{i}, 'interpreter', 'latex')
    saveTightFigure(gcf,[newdir, title_cell{i},'_',num2str(frame_len),'.pdf'])

end



save([newdir, 'data_tvgl_demo_',num2str(frame_len),'.mat'], 'L_cell', 'title_cell', 'Fscore','times','FDR','RelativeEr', 'data_struct');

    