function data_struct = synthetic_data_TVG( params_struct )

% Generating random synthetic Data
% -------------------------------------
% The following graph types can be chosen 
% 1:   'stochastic block'
% 2:   '2D-grid'
% 3:   'barabasi_albert'
% 4:   'erdos_renyi'
% 5:   'sensor'
% 6:   'random_geometric'
% -------------------------------------
% You should first load the 'graph_types.mat' data.

warning("off")

Graph_Data = load('graph_types.mat','G_cell');
G_cell = Graph_Data.G_cell;

if isfield(params_struct, 'type')            % graph type 
    type = params_struct.type;
else
    type = 1;
end
 
if isfield(params_struct, 'N')               % Number of nodes
    N = params_struct.N;
else
    N = 64;
end

if isfield(params_struct, 'T')               % Number of measurments 
    T = params_struct.T;
else
    T = 10*N;
end

if isfield(params_struct, 'order')           % order of the VAR model for the signal
    order = params_struct.order;
else
    order = 1;
end



if isfield(params_struct, 'DCval')              % Signal DC value 
    DCval = params_struct.DCval;
else
    DCval = 0;
end


if isfield(params_struct, 'weight_dist')    	% Weight distribution
    weight_dist = params_struct.weight_dist;
else
    weight_dist = 'uniform';
end


if isfield(params_struct, 'edge_weight_par')    % The std of the weight distribution 
    edge_weight_par = params_struct.edge_weight_par;
else
    edge_weight_par = 1;
end


if isfield(params_struct, 'AVAR_type')          % AVAR matrix distribution type 
    AVAR_type = params_struct.AVAR_type;
else
    AVAR_type = 'randl';
end

if isfield(params_struct, 'noise_type')        % Noise distribution 
    noise_type = params_struct.noise_type;
else
    noise_type = 'randn';
end

if isfield(params_struct, 'mask_AVAR')   % is AVAR masked by Laplacian
    mask_AVAR = params_struct.mask_AVAR;
else
    mask_AVAR = 0;
end

if isfield(params_struct, 'AVAR_symmetric')   % is AVAR symmetric 
    AVAR_symmetric = params_struct.AVAR_symmetric;
else
    AVAR_symmetric = 0;
end


if isfield(params_struct, 'Normalization')     % Normalization = 'max','trace';
    Normalization = params_struct.Normalization;
else
    Normalization = 'trace';
end

if isfield(params_struct, 'W_thr')   % Weight threshold
    W_thr = params_struct.W_thr;
else
    W_thr = 1e-2;
end

if isfield(params_struct, 'A_thr')   % AVAR entries threshold
    A_thr = params_struct.A_thr;
else
    A_thr = 2;
end


if isfield(params_struct, 'frame_len')              % Frame length
    frame_len = params_struct.frame_len;
else
    frame_len = 200;
end


if isfield(params_struct, 'nu')              % nu 
    nu = params_struct.nu;
else
    nu = 4;
end


if isfield(params_struct, 'a_var_type')          % AVAR matrix distribution type 
    a_var_type = params_struct.a_var_type;
else
    a_var_type = 'rand';
end

% if isfield(params_struct, 'std_n')            % noise level 
%     std_n = params_struct.std_n;
% else
%     std_n = 0.1;
% end
%   
% if isfield(params_struct, 'sr')               % sampling rate 
%     sr = params_struct.sr;
% else
%     sr = 0.8;
% end


%% Generate the Laplacian
%-------------------------

if isfield(params_struct,'G')
    G = params_struct.G;
else
    G = G_cell{type};
end


W_connectivity = G.A; 

W = W_connectivity.* ( rand(N)*(edge_weight_par-0.1)+0.1 );
W = 0.5* (W+W');
L = diag(sum(W,2)) - W;

if strcmp(Normalization, 'trace')==1
  W = W/trace(L)*N;
  L = L/trace(L)*N;
  
elseif strcmp(Normalization, 'max')==1
  W = W/(max(max(W)));
  L =  diag(sum(W,2)) - W;
end


[U,E] = eig(full(L));
e = diag(E) ;
e(e <= 1e-6) = 0;
e = 1./e; 
e(e == Inf) = 0;
e(isnan(e)) = 0;
L_Scale_matrix = (U*diag(e)*U');  



%% Generate the AVAR transition matrix 
%-------------------------

if isfield(params_struct,'AVAR_true_cell')   
    AVAR_true_cell = params_struct.AVAR_true_cell;

else
    AVAR_true_cell = cell(1,order);
    for p=1:order
        AVAR = edge_weight_par* generate_AVAR( N, AVAR_type );   
        
        if AVAR_symmetric
            AVAR = 0.5*(AVAR+AVAR');
        end
        

        if mask_AVAR
            AVAR = AVAR.* (full(abs(G.L)>0)) ;
        else
            Ind = rand(size(AVAR ))<A_thr;
            AVAR(Ind) = 0;
        end


        AVAR(1:N+1:end) = abs( AVAR(1:N+1:end) );

        if norm(AVAR,2)>0
            AVAR = AVAR/norm(AVAR,2);
        end
        
        AVAR_true_cell{p} = AVAR;

    end

end

%% Generate the time-varying graph and signal

N_f = fix(T/frame_len);
N_w = N*(N-1)/2;

% ====== a_tvg ==========
w0 = w_from_L(L);

params = struct;
params.type = a_var_type;
params.mu = 1;
a_tvg = generate_random_matrix(N_w,1, params);
a_tvg = 1+0.2*(a_tvg-0.5);
Noise_w = 0.5*std(w0)*randl(N_w,N_f);


W_N = zeros(N_w, N_f);
W_N(:,1) =  w0;


%----------------------------
X = zeros(N, T);


params = struct;
params.type = noise_type;
params.nu = nu;
Noise = generate_random_matrix(N,T, params);


for k = 1:frame_len
    X(:,k) = L_Scale_matrix* Noise(:,k);
end


for i_f = 2:N_f
    W_N(:,i_f) = (w0>0).* max(0,  a_tvg.*W_N(:,i_f-1) + Noise_w(:,i_f));
    [U,E] = eig(L_operator_mex(W_N(:,i_f)));
    e = diag(E) ;
    e(e <= 10^-10) = 0;
    e = 1./e; 
    e(e == Inf) = 0;    
    e(isnan(e)) = 0;
    L_Scale_matrix = U* diag(e)*U'; 

    for j = 1:frame_len
        k = (i_f-1)*frame_len + j;
        f = Noise(:,k);
        %  f = f / norm(f) * epsilon; 
        fdc = randn * DCval * ones(N,1); % DC component
        temp = zeros(N,1);
        for p=1:order
            if k>p
                temp = temp + AVAR_true_cell{p}*X(:,k-p) ;
            end
        end
        X(:,k) = temp + L_Scale_matrix*f + fdc;
    end

end


% ----- sampling matrix ------
% SampleNum = floor(N*sr); % the number of sampled points at each time
% SampleMatrix = zeros(N,T);
% 
% for i = 1:T
%     SampleMatrix(randperm(N, SampleNum),i) = 1;
% end 
%
%
% ----- measurement noise -------
% noise = std_n * randn(size(X)); % measurement noise
% Y = SampleMatrix.*X + noise;

X = X - repmat(mean(X,2),1,T);
std_rows = std(X,[],2);
X = diag(1./std_rows)*X;

data_struct.Wconn_true = W_connectivity;
data_struct.Wtrue = W;
data_struct.W_N_true = W_N;
data_struct.A_N_true = W_N>0;
data_struct.AVAR_true = AVAR;
data_struct.AVAR_true_cell = AVAR_true_cell;
data_struct.Ltrue = L;
data_struct.Utrue = U;
data_struct.X = X;
data_struct.std_rows = std_rows;
data_struct.coords = G.coords;
% data_struct.S = SampleMatrix;
% data_struct.N = noise;
% data_struct.Y = Y;


end

