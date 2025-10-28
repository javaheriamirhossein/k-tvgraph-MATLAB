function [w_mat, times] = learn_tv_graph_Saboksayr(X,params)

[N, T] = size(X);


if isfield(params, 'frame_len')
    frame_len = params.frame_len;
else
    frame_len = 10;
end


if isfield(params, 'alpha')
    alpha = params.alpha;
else
    alpha = 0.1;
end


if isfield(params, 'beta')
    beta = params.beta;
else
    beta = 0.01;
end


if isfield(params, 'gamma')
    gamma = params.gamma;
else
    gamma = 0.1;
end


if isfield(params, 'maxiter')
    maxiter = params.maxiter;
else
    maxiter = 10;
end



X_f = X( :, 1: frame_len);
S_cov_f = cov(X_f',1);
S_inv_F = pinv(S_cov_f);


if isfield(params, 'w0')
    w0 = params.w0;
else
    w0 = w_from_L(S_inv_F);
%     w0 = w0/sum(w0)*N/2;

end


reltol = 1e-4;

%============================
[S, St] = sum_squareform(N);


N_w = length(w0);

w_mat = zeros(N_w, T);
w_mat(:,1) = w0;



Z = gsp_distanz(X_f').^2;
zhat = squareform(Z)';

times = nan(T,1);
t0 = tic;

for t=1:T-1

    w = w_mat(:,t) ;
    
    for i=1:maxiter
        X_f =  X(:,t);
        Z_t = gsp_distanz(X_f').^2;
    
         
        Sw = S*w + 1e-3;
        grad = 4*beta *w -  alpha* St* (1./ (Sw));
        z_t = squareform(Z_t)';
        zhat = (1-gamma)*zhat + gamma* z_t;
        mu = 1/( 4*beta + 2*alpha*(N-1)/ (min(Sw)^2) );
        w_new = w - mu*(grad + 2*zhat);
    
        w_new = max(0,w_new);
        rel_err = norm(w_new - w) / norm(w) ;

        if (rel_err < reltol) && (i > 1)
            break
        end

        w = w_new;
        
        fprintf('Saboksayr TVGL t = %d\n',t);
    end
    w_mat(:,t+1) = w_new;
    times(t) = toc(t0);
end


end



