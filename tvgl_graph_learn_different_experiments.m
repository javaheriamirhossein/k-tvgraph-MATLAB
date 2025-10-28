maindir = cd;
addpath(genpath(maindir))
addpath(genpath([cd,'\gspbox-0.7.5']))


clc
close all
rng(1);

% experiments = {'noise', 'sr', 'gamma', 'd', 'rho', 'sigma_e'};
experiments = { 'sr', 'noise', 'rho', 'gamma', 'd', 'sigma_e', 'nu', 'K', 'frame_len'};
experiments = { 'sr'};
t0 = tic;

 
type = 2;     % graph type  1: 'stochastic block', 2: '2D-grid',  3: 'barabasi_albert',  
              % 4: 'erdos_renyi', 5: 'sensor', 6: 'random_geometric'
N = 100;      % number of nodes
T = 10*N;     % number of measurment samples
sr = 1;       % sampling rate
std_n = 0;    % noise standard deviation
a_var_type = 'rand';     % prior distribution of the VAR parameters
noise_type = 't';        % type of innovation noise 't': Student-t,  'randn': Gaussian, 'rand': uniform, etc
Normalization = 'trace'; % Laplacian normalization type
W_thr = 0.001;   % weights threshold
framelen = 200; % frame length
nu = 4;         % Student-t degree of freedom parameter 
Ntrial = 10;     % number of trails



data_params_struct.frame_len = framelen;
data_params_struct.type = type;
data_params_struct.noise_type = noise_type;
data_params_struct.nu = nu;
data_params_struct.N = N;
data_params_struct.T = T;
data_params_struct.sr = sr;
data_params_struct.std_n = std_n;
data_params_struct.a_var_type = a_var_type;
data_params_struct.Normalization = Normalization;
data_params_struct.W_N_fixed = 0;
data_params_struct.X_fixed = 0;
data_params_struct.sampling_fixed = 0;
data_params_struct.noise_fixed = 0;


params_struct.W_thr = W_thr;
params_struct.Ntrial = Ntrial;
params_struct.Normalization = Normalization;




task = 'graph learning under';

Nexp = length(experiments);
all_results = cell(1,Nexp);



code = [num2str(data_params_struct.W_N_fixed), num2str(data_params_struct.X_fixed), num2str(data_params_struct.sampling_fixed), num2str(data_params_struct.noise_fixed)];

W_thr_str = num2str(W_thr);
W_thr_str = W_thr_str(3:end);

description = [num2str(N), ' ', num2str(T), ' ', num2str(framelen), ' ', num2str(10*sr), ' ', num2str(10*std_n), ' ', a_var_type, ...
    ' ', Normalization, ' ', W_thr_str, ' ', noise_type, ' ', code];

newdir = [maindir, '\Results_TVGL\', date,'\', description,'\' ];


if ~isfolder(newdir)
    mkdir(newdir)
end



x_axis_log = 0;

for count=1:Nexp
    x_axis_log = 0;
    params_struct.frame_len = framelen;
    params_struct.nu = 4;
    params_struct.d = 1;
    params_struct.K = 1;
    params_struct.rho = 1;
    params_struct.sigma_e = exp(0.1);
    params_struct.gamma = 0.1;
    params_struct.std_n = std_n;
    params_struct.sr = sr;
    experiment = experiments{count};

    % Specify the type of experiment
    switch experiment
        case 'noise'
            param_vec = 10*[0, 0.03, 0.05, 0.1, 0.15];
            labelx = 'Noise Std';
            param_field = 'std_n';

        case 'sr'
            param_vec = [0.2, 0.4, 0.6, 0.8, 1];
            param_vec = sort(param_vec, 'descend');
            labelx = 'Sampling Rate';
            param_field = 'sr';

        case 'd'
            param_vec = [1,2,3,4,5,6];
            labelx = '$d$';
            param_field = 'd';

        case 'a_type'
            input_cell  = {'rand', 'exp', 'randl', 'randn', 'eye'};
            param_vec = 1:length(input_cell);
            labelx = 'VAR type';
            param_field = 'a_var_type';

        case 'N'
            param_vec = [50, 100, 200, 500];
            labelx = '$N$';
            param_field = 'N';

        case 'frame_len'
            param_vec = [20, 50, 100, 200, 500];
            labelx = '$F_n$ (frame length)';
            param_field = 'frame_len';
        

        case 'rho'
            param_vec = [1, 2, 3, 4, 6, 10];
            labelx = '$\rho$';
            param_field = 'rho';

        case 'gamma'
            param_vec = logspace(-3,2,6);
            labelx = '$\gamma$';
            x_axis_log = 1;
            param_field = 'gamma';

        case 'sigma_e'
            param_vec = logspace(-3,2,6);
            labelx = '$\sigma_e$';
            x_axis_log = 1;
            param_field = 'sigma_e';

        case 'K'
            param_vec = [1, 2, 3, 4, 5, 6];
            labelx = '$k$';
            param_field = 'K';

        case 'nu'
            param_vec = logspace(0.1,2,6);
            labelx = '$\nu$';
            x_axis_log = 1;
            param_field = 'nu';


    end

    % Extract the results
    N_param = length(param_vec);
    Results_params = cell(1,N_param);

    for n = 1:N_param
        param_val = param_vec(n);
        params_struct.(param_field) = param_val;
        if strcmp(param_field,'frame_len')
            data_params_struct.frame_len =  param_val;
        end
        if strcmp(param_field,'N')
            data_params_struct.N =  param_val;
        end
        [Results, Algnames_cell] = tv_graph_learning_algorithms( data_params_struct, params_struct);
        Results_params{n} = Results;
    end

    Results = Results_params{1};
    field_names = fieldnames(Results);
    N_alg = size(Results.(field_names{1}),2);


    titleStr = [task, ' different ', experiment];
    legend_cell = Algnames_cell;

    % Plot the results
    N_figs = length(field_names);
    ylabls = {'F-score', 'Relative Error', 'time (s)'};

    for k = 1:N_figs
        result_mat = zeros(N_alg, N_param);
        error_mat = zeros(N_alg, N_param);
        for n=1:N_param
            Results = Results_params{n};
            temp = Results.(field_names{k});
            if strcmp(field_names{k}, 'times')
                size_temp = size(temp);
                size_temp(1) = size_temp(1) + 1;
                temp2 = zeros(size_temp);
                temp2(2:end,:,:) = temp;
                temp = diff(temp2,1,1);
            end
            result = squeeze(nanmean(temp,1));
            if size(result,2) ~= Ntrial
                result = result';
            end
            result_mean = nanmean(result,2);
            result_std = std(result,[],2);
            result_mat(:,n) = result_mean;
            error_mat(:,n) = result_std;
        end



        figure
        figname = field_names{k};
        for i=1:N_alg
             errorbar(param_vec,result_mat(i,:), error_mat(i,:), 'LineWidth',1.2);
             hold on;
        end
        if x_axis_log
            set(gca,'XScale','log');
        end
        xlabel(labelx, 'interpreter', 'latex','FontSize',13)
        ylabel(ylabls{k},'interpreter','latex','FontSize',13)
        legend(legend_cell, 'interpreter', 'latex','location','northeast')  

        % xlim([param_vec(1), param_vec(end)*1.1])
        % axis([-inf inf -inf inf])

        set(gcf, 'Position', [100 100 500 400])

        title(titleStr, 'interpreter', 'latex','FontSize',15)
        grid on



        figsavename = [newdir, '\TVGL diff ', experiment, ' ', figname, '.fig'];
        savefig(gcf,figsavename);
        figsavename = [newdir, '\TVGL diff ', experiment, ' ', figname, '.png'];
        saveas(gcf,figsavename);
        close(gcf)

    end

    matsavename = [newdir, '\Results_params_', param_field, '.mat'];
    save(matsavename, 'Results_params')
end

elapsed_time = toc(t0)


