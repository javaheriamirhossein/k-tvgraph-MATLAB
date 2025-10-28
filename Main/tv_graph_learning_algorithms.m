function [Results, algnames_cell] = tv_graph_learning_algorithms( data_params_struct, params_struct, methods)



if isfield(params_struct, 'Normalization')   % Type of normalization of the Laplacian: 'max','trace';
    Normalization = params_struct.Normalization;
else
    Normalization = 'trace';
end

if isfield(params_struct, 'W_thr')           % threshold for removing the small weights
    W_thr = params_struct.W_thr;
else
    W_thr = 1e-2;
end


if isfield(params_struct, 'alpha')
    alpha = params_struct.alpha;
else
    alpha = 0.1;
end

if isfield(params_struct, 'beta')
    beta = params_struct.beta;
else
    beta = 0.1;
end

if isfield(params_struct, 'gamma')
    gamma = params_struct.gamma;
else
    gamma = 0.1;
end


if isfield(params_struct, 'nu')
    nu = params_struct.nu;
else
    nu = 4;
end

if isfield(params_struct, 'sigma_e')
    sigma_e = params_struct.sigma_e;
else
    sigma_e = exp(0.1);
end

if isfield(params_struct, 'rho')
    rho = params_struct.rho;
else
    rho = 1;
end

if isfield(params_struct, 'd')
    d = params_struct.d;
else
    d = 1;
end


if isfield(params_struct, 'K')
    K = params_struct.K;
else
    K = 1;
end

if isfield(params_struct, 'sr')
    sr = params_struct.sr;
else
    sr = 1;
end


if isfield(params_struct, 'std_n')
    std_n = params_struct.std_n;
else
    std_n = 0;
end



if isfield(params_struct, 'Ntrial')
    Ntrial = params_struct.Ntrial;
else
    Ntrial = 2;
end


if isfield(data_params_struct, 'X_fixed')
    X_fixed = data_params_struct.X_fixed;
else
    X_fixed = 1;
end

if isfield(data_params_struct, 'noise_fixed')
    noise_fixed = data_params_struct.noise_fixed;
else
    noise_fixed = 0;
end

if isfield(data_params_struct, 'sampling_fixed')
    sampling_fixed = data_params_struct.sampling_fixed;
else
    sampling_fixed = 0;
end


if nargin<3
    methods = {
        'kTVGL', ...
        'GSPBOX',
        };
end



T = data_params_struct.T;
N = data_params_struct.N;
frame_len = data_params_struct.frame_len;
N_f = fix(T/frame_len);


%% =====================================

RelativeEr_all_cell = cell(1,Ntrial);
Fscore_all_cell = cell(1,Ntrial);
times_all_cell = cell(1,Ntrial);

data_struct = synthetic_data_TVG( data_params_struct );
X = data_struct.X;

SampleMatrix = zeros(N,T);
SampleNum = floor(N*sr);
% sampling mask
for i = 1:T
    SampleMatrix(randperm(N, SampleNum),i) = 1;
end

noise = std_n * randn(size(X));    % measurement noise

%%  ===========================

for trial = 1:Ntrial

    % Synthetic data generation

    if ~X_fixed
        data_struct = synthetic_data_TVG( data_params_struct );
        X = data_struct.X;
    end


    if ~sampling_fixed
        SampleNum = floor(N*sr);
        SampleMatrix = zeros(N,T);

        for i = 1:T
            SampleMatrix(randperm(N, SampleNum),i) = 1;
        end
    end

    if ~noise_fixed
        noise = std_n * randn(size(X));
    end

    % true synthetic data
    W_N_true = data_struct.W_N_true;
    X = data_struct.X;

    % corrupted synthetic data
    Y = SampleMatrix.*(X + noise);
    Mask = SampleMatrix;


    % framing
    Y_f_cell = cell(1,N_f);
    Mask_cell = cell(1,N_f);

    for i = 1:N_f
        Y_f = Y( :, ((i-1)*frame_len+1): (i*frame_len));
        Mask_f = Mask( :, ((i-1)*frame_len+1): (i*frame_len));
        Y_f_cell{i} = Y_f;
        Mask_cell{i} = Mask_f;
    end


    %% Define cell variables for storing the results
    L_cell = cell(1);
    algnames_cell = cell(1);
    times_cell = cell(1);
    Laplacian_cell_f = cell(N_f,1);


    %% Running algorithms
    % ==========================================

    ind = 0;

    if any(strcmp(methods,'kTVGL'))
        ind = ind + 1;
        params = struct;
        params.frame_len = frame_len;
        params.maxiter = 100;
        params.sigma_e = sigma_e;
        % params.alpha = 0.1;
        % params.beta  = 0.01;
        params.nu = nu;
        params.update_beta = 1;
        params.rho = rho;
        params.K = K;
        params.d = d;
        params.gamma = gamma;
        params.std_n = 0.1;
        params.W_thr = 0;

        [w_cell, a_var, Xhat, times] = learn_kcomp_heavytail_tv_graph( Y, Mask, params);

        for k=1:N_f
            Laplacian = L_operator_mex(w_cell{k});
            Laplacian_cell_f{k} = Laplacian;
        end
        L_cell{ind} = Laplacian_cell_f;
        times_cell{ind} = times;
        algnames_cell{ind} = 'kTVGL (Propoed)';
    end



    if any(strcmp(methods,'GSPBOX'))
        ind = ind + 1;
        params = struct;
        times = nan(N_f,1);
        t0 = tic;
        for k=1:N_f
            Z = gsp_distanz(Y_f_cell{k}').^2;
            theta = gsp_compute_graph_learning_theta(Z,8);
            W = gsp_learn_graph_l2_degrees(theta *Z, 1, params);
            Laplacian = diag(sum(W,2)) - W;
            Laplacian_cell_f{k} = Laplacian;
            params.w_0 = w_from_L(Laplacian);
            params.c = 10;
            times(k) = toc(t0);
        end
        times_cell{ind} = times;
        L_cell{ind} = Laplacian_cell_f ;
        algnames_cell{ind} = 'GSPBOX-L2';
    end


    %% ==========================================
    % Evaluation

    N_alg = length(L_cell);

    RelativeEr = zeros(N_f,N_alg);
    Fscore = zeros(N_f,N_alg);
    times = zeros(N_f,N_alg);

    for i = 1:N_alg
        Laplacian_cell_f = L_cell{i};

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

            % normalize and threshold the inferred Laplacian
            W = W/max(max(W));
            W(W<W_thr) = 0;

            Laplacian = diag(sum(W,2))-W;
            w = w_from_L(Laplacian);

            if strcmp(Normalization, 'trace')==1
                Laplacian = Laplacian/trace(Laplacian)*N;
            elseif strcmp(Normalization, 'max')==1
                Laplacian = Laplacian/max(max(w));
            end
            Adjacency = Laplacian<0;

            RelativeEr(k,i) = norm(Laplacian-Ltrue,'fro')/norm(Ltrue,'fro');
            tp = sum(sum(Adjacency.*A_mask));
            fp = sum(sum(Adjacency.*(~A_mask)));
            fn = sum(sum((~Adjacency).*A_mask));
            Fscore(k,i) = 2*tp/(2*tp+fn+fp);

        end

        times(:,i) = times_cell{i};

    end


    RelativeEr_all_cell{trial} = RelativeEr;
    Fscore_all_cell{trial} = Fscore;
    times_all_cell{trial} = times;

end


%% ----------------------------------------

RelativeEr_all = zeros(N_f, N_alg, Ntrial);
Fscore_all  = zeros(N_f, N_alg, Ntrial);
times_all  = zeros(N_f, N_alg, Ntrial);

for trial = 1:Ntrial
    RelativeEr_all(:,:,trial) = RelativeEr_all_cell{trial};
    Fscore_all(:,:,trial)  = Fscore_all_cell{trial};
    times_all(:,:,trial)  = times_all_cell{trial};
end

Results = struct;
Results.Fscore = Fscore_all;
Results.RelativeEr = RelativeEr_all;
Results.times = times_all;
% Results.Algnames = algnames_cell;


end