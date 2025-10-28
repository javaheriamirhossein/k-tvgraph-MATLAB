function [w_cell, a_var, X, times] = learn_kcomp_heavytail_tv_graph( Y, SampleMask, params)
% ==================================================
% learn_kcomp_heavytail_tv_graph: Learn time-varying graphs from heavy-tailed incomplete data 
% --------------------------------------------------
% This algorithm learns a time-varying graph modeled as piecewise constant
% within each frame. It is applicable for learning graphs from heavy-tailed
% data modeled by the Student-t distribution, and also suitable for Gaussian
% data by properly adjusting the shape parameter, nu. The algorithm supports
% learning K-component graphs, enabling K-class clustering. It is based on
% a majorization-minimization variant of ADMM to solve the following
% problem:
%
%  Objective Function:
%    Minimize the function f(w_n, X_n, a, u_n, L_n, V_n) consisting of:
%       - Negative log-determinant of matrix L_n:         -logdet*(L_n)
%       - Weighted L1 norm penalty for u_n:               + alpha * norm(u_n, 1)
%       - Weighted L0 norm penalty for w_n:               + beta * norm(w_n,0)
%       - Logarithmic sum over frame set F_n:             + ((nu+p)/T_n) * sum_{t in F_n} log(1 + x_t' * L(w_n) * x_t / nu)
%       - Data fidelity term with Frobenius norm:         + (1/(T_n * sigma_n^2)) * norm(Y_n - M_n .* X_n, 'fro')^2
%       - Weighted sum of a:                              + gamma * a' * ones
%       - Weighted trace regularization:                  + eta * trace(L(w_n) * V_n * V_n')
%
%  Variables to optimize:
%    w_n >= 0, a >= 0, X_n, u_n, L_n, V_n
%
%  Subject to constraints:
%   - L_n equals L applied to w_n:                  L_n = L(w_n)
%   - u_n is w_n minus elementwise product:         u_n = w_n - (a .* w_{n-1})
%   - Linear constraint on w_n:                     d(w_n) = d
%   - Rank constraint for L_n:                      rank(L_n) = p - k
%   - Orthonormality constraint for V_n:            V_n' * V_n = I
%
%
% --------------------------------------------------
%  Usage:  
%         output = learn_STSRGL(Y)
%         output = learn_STSRGL(Y, SampleMask)
%         output = learn_STSRGL(Y, SampleMask, params)
%
%  Inputs:
%         Y          : Noisy and missing data matrix for all time frames
%         SampleMask : Sampling mask matrix (all time frames)
%         params     : Algorithm parameters
%
%  Outputs:
%         w_cell  : Inferred time-varying graph weights at all time frames
%         a_var   : Inferred VAR model parameters for the weights
%         X       : Reconstructed data matrix for all time frames
%         times   : Elapsed time in each iteration
%
%  params:
%    Normalization   : Specify how the Laplacian matrix should be normalized
%    alpha           : Regularization parameter for the L1 norm penalty on the weights differences
%    beta            : Regularization parameter for the L0 norm penalty on the weights
%    sigma_e         : The parameter to specify alpha and beta, if they arenot given
%    gamma           : Regularization parameter for L1 norm penalty on the VAR parameters, a
%    eta             : Regularization parameter for the trace term on V_n
%    frame_len       : Length of the time frame
%    maxiter         : Maximum number of iterations. Default: 100
%    W_thr           : Weight pruning threshold
%    reltol          : Relative error tolerance for stopping criterion. Defaul: 1e-5
%    nu              : Shape parameter of the Student-t model
%    K               : number of clusters or components
%    d               : maximum degree of each node
%    rho             : Parameter of the ADMM
%    w0              : Initial value of the weights (at the first time frame)
%    X0              : Initial value of X
%    std_n           : Standard deviation of noise: Default: 0
% --------------------------------------------------
%
%
%
% Author: Amirhossein Javaheri
% Date: Sept 2025
% ==================================================

if nargin<3
    params = struct;
end
if nargin<2
    SampleMask = ones(size(Y));
end

[N, T] = size(Y);


if isfield(params, 'frame_len')
    frame_len = params.frame_len;
else
    frame_len = fix(0.2*T);
end

Y_f = Y( :, 1: frame_len);
Y_f =  normalize(Y_f,2);
S_cov_f = cov(Y_f',1);
S_inv_F = pinv(S_cov_f);


% if isfield(params, 'W_thr')
%     W_thr = params.W_thr;
% else
%     W_thr = 0.01;
% end
% 
% 
% if isfield(params, 'Normalization')
%     Normalization = params.Normalization;
% else
%     Normalization = 'trace';
% end



if isfield(params, 'sigma_e')
    sigma_e = params.sigma_e;
else
    sigma_e = exp(0.1);
end


if isfield(params, 'alpha')
    alpha = params.alpha;
else
    alpha = 2/(frame_len* sigma_e);
end

if isfield(params, 'beta')
    beta = params.beta;
else
    beta = 2*log(sigma_e)/frame_len;
end


if isfield(params, 'gamma')
    gamma = params.gamma;
else
    gamma = 0.1;
end


if isfield(params, 'eta')
    eta = params.eta;
else
    eta = 1e-8;
end


if isfield(params, 'nu')
    nu = params.nu;
else
    nu = 4;
end


if isfield(params, 'K')
    K = params.K;
else
    K = 1;
end

if isfield(params, 'd')
    d = params.d;
else
    d = 1;
end

if isfield(params, 'rho')
    rho = params.rho;
else
    rho = 3;
end


if isfield(params, 'w0')
    w0 = params.w0;
else
    w0 = w_from_L(S_inv_F);
    W0 = W_from_w(w0);
    d0 = sum(W0,2);
    D_inv_sqrt = diag(1./sqrt(d0));
    W0 = (D_inv_sqrt*D_inv_sqrt'*W0);
    L = diag(sum(W0,2)) - W0;
    w0 = w_from_L(L');
end


if isfield(params, 'X0')
    X = params.X0;
else
    X = Y;
end



if isfield(params, 'maxiter')
    maxiter = params.maxiter;
else
    maxiter = 1000;
end

if isfield(params, 'update_eta')
    update_eta = params.update_eta;
else
    update_eta = 1;
end


if isfield(params, 'std_n')
    std_n = params.std_n;
else
    std_n = 0.0;
end


if isfield(params, 'reltol')
    reltol = params.reltol;
else
    reltol = 1e-5;
end





%============================
[S, ~] = sum_squareform(N);
N_f = fix(T/frame_len);
w_cell = cell(1,N_f);
times = nan(N_f,1);



t0 = tic;

% Initialize a_var
a_var = ones(size(w0));
w_lagged = zeros(size(w0));

for i_f = 1:N_f
    
    
    Mask_f = SampleMask( :, ((i_f-1)*frame_len+1): (i_f*frame_len) );
    Y_f = Y( :, ((i_f-1)*frame_len+1): (i_f*frame_len) );

    % Initialzie primal variables
    X_f = X( :, ((i_f-1)*frame_len+1): (i_f*frame_len) );
    w = w0;
    Lw = L_operator_mex(w);
    Laplacian = Lw;
    [U,~] = eig(0.5*(Lw+Lw'),'vector');
    U = U(:,1:K);
    u = w - a_var.*w_lagged;

    % Initialzie dual variables
    Phi = zeros(N,N);
    mu_vec = zeros(length(w),1);
    z = zeros(N,1);



    has_converged_w  = 0;
    has_converged_u  = 0; 
    has_converged_a  = 0;
    has_converged_L  = 0;
    has_converged_X  = 0;
    for j = 1:maxiter
        

        % update w
        coeffs = student_cov_weights(w, X_f, nu);
        X_f_norm_weighted = repmat(coeffs, N, 1).*X_f;
        S_f =  (X_f_norm_weighted*X_f')/frame_len + eta * (U*U') + Phi + rho * (Lw - Laplacian) ;

        grad = conj_L_operator_mex( S_f );
        grad = grad - mu_vec - rho*(u + a_var.*w_lagged);
        d_grad = conj_d_operator_mex( z -rho*(d - S*w) );
        grad = grad + d_grad;
        
        step = 1 / (rho + 2*N*rho + rho*(2*N-2));
        w_new = (1-rho*step)*w - step *  grad;

        thr = sqrt( 2*beta *step );
        w_new(w_new<thr) = 0;
        
        rel_err = norm(w_new - w) / norm(w) ;
        if (rel_err < reltol) && (j > 1)
            has_converged_w = 1;
        end

        w = w_new;
        Lw = L_operator_mex(w);

        % update u
        temp = w -a_var.* w_lagged - mu_vec/rho;
        thr = alpha/rho;

        u_new = Threshold_Soft(temp, thr);
        rel_err = norm(u_new - u) / norm(u) ;
        if (rel_err < reltol) && (j > 1)
            has_converged_u = 1;
        end
        u = u_new;
        

        % update a_var
        temp = (w -u - mu_vec/rho)./ w_lagged;
        temp(temp<0) = 0;
        temp(temp == Inf) = 0;
        temp(isnan(temp)) = 0;
        thr = gamma./(rho*w_lagged.^2);
        thr(thr == Inf) = 0;
        thr(isnan(thr)) = 0;
        a_var_new = Threshold_Soft(temp, thr) .* (w_lagged>0);
        rel_err = norm(a_var_new - a_var) / norm(a_var) ;
        if (rel_err < reltol) && (j > 1)
            has_converged_a = 1;
        end
        a_var = a_var_new;


        


        % update Laplacian
        temp =  Lw + Phi/rho;
        [V1,lamb] = eig(0.5*(temp+temp'), 'vector');
        V1 = V1(:,K+1:N);
        lamb = lamb(K+1:N);
        Laplacian_new = V1 * diag((lamb + sqrt(lamb.^2 + 4 /rho)) / (2)) * V1';

        rel_err = norm(Laplacian_new - Laplacian, 'fro') / norm(Laplacian, 'fro') ;
        if (rel_err < reltol) && (j > 1)
            has_converged_L = 1;
        end

        Laplacian = Laplacian_new;
        

        % update U
        [U,lamb] = eig(0.5*(Lw+Lw'),'vector');
        U = U(:,1:K);

         
        % update X
        coeffs = student_cov_weights(w, X_f, nu);
        coeffs_mat = repmat(coeffs,N,1);       
        Noise_res = Mask_f.*X_f - Y_f;
        tau_mat = 1 + std_n^2*coeffs_mat*lamb(N);
        LwX = std_n^2*Lw*X_f;
        X_f_new = X_f - 1./tau_mat.*(Noise_res + coeffs_mat.*LwX);

        rel_err = norm(X_f_new - X_f, 'fro') / norm(X_f, 'fro') ;
        if (rel_err < reltol) && (j > 1)
            has_converged_X = 1;
        end

        X_f = X_f_new;




        % update dual variables
        R1 = Lw - Laplacian ;
        Phi = Phi + rho * R1;
        
        R0 = u - w + a_var.*w_lagged ;
        mu_vec = mu_vec + rho * R0;

        R2 = diag(Lw) - d;
        z = z + rho * R2;

        if (has_converged_w && has_converged_u && has_converged_a && has_converged_L  && has_converged_X)
            fprintf('stopped at iteration = %d\n',j);
            break;
        end

        
        if (update_eta)
            eig_vals = lamb;
            n_zero_eigenvalues = sum(eig_vals < 1e-6);
            if (K < n_zero_eigenvalues)
                eta =  eta/1.5;
            elseif (K > n_zero_eigenvalues)
                eta = 1.5 * eta;
            end
        end

    end

    fprintf('kTVGL running, frame = %d\n',i_f);

    w0 = w;
    w_cell{i_f} = w;
    w_lagged  = w;
    X( :, ((i_f-1)*frame_len+1): (i_f*frame_len) ) = X_f;
    times(i_f) = toc(t0);
end



end



%% =============================================
function z = student_cov_weights(w,X,nu)

[N,T] = size(X);
LX = L_operator_mex(w)*X;
z = zeros(1,T);
for t=1:T
    g_t = X(:,t)'*LX(:,t);
    z(t) = (nu+N)/(g_t+nu);
end

end


function ahat = Threshold_Soft(A,thr)
    a = A(:);
    temp = abs(a)-thr;
    supp = temp>=0;
    ahat = zeros(size(a));
    ahat(supp) = sign(a(supp)).*temp(supp);
end