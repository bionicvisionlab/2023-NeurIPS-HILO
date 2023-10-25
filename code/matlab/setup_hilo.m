function hiloModel = setup_hilo(kernelname, ub, lb, thetacov, acqFnName, maxiter, nopt)
%SETUP_HILO returns a struct with all of the relevant 
% objects to do hilo
addpath(genpath('.'));

d = numel(ub);

if size(ub, 2) ~= 1
    % need to make it a column vector
    ub = ub.';
end
if size(lb, 2) ~= 1
    % need to make it a column vector
    lb = lb.';
end

rand_acq = @() rand(size(ub)).*(ub - lb) + lb;
if strcmpi(acqFnName, 'random')
    acq = @(theta, xtrain_norm, ctrain, model, post, approximation, optim) [rand_acq(); rand_acq()];
    disp('doing random')
else
%     disp('acq:')
%     disp(acqFnName)
    switch (acqFnName)
        case 'MUC'
            acq = @MUC;
        case 'Dueling_UCB'
            acq = @Dueling_UCB;
        case 'bivariate_EI'
            acq = @bivariate_EI;
        case 'Thompson_challenge'
            acq = @Thompson_challenge;
        otherwise
            disp(acqFnName)
            error("Unknown acquisition")
    end
end

%  TODO theta
theta.cov = thetacov;
theta.mean = 0;

post = [];  
regularization = 'nugget';
condition.x0 = zeros(d,1);
condition.y0 = 0;
link = @normcdf;
modeltype = 'exp_prop';
meanfun = @constant_mean;
hyperparameters.ncov_hyp =numel(theta.cov); % number of hyperparameters for the covariance function
hyperparameters.nmean_hyp =1; % number of hyperparameters for the mean function
hyperparameters.hyp_lb = -10*ones(hyperparameters.ncov_hyp  + hyperparameters.nmean_hyp,1);
hyperparameters.hyp_ub = 10*ones(hyperparameters.ncov_hyp  + hyperparameters.nmean_hyp,1);
switch(kernelname)
    case 'Gaussian'
        base_kernelfun =  @Gaussian_kernelfun;
    case 'ARD'
        base_kernelfun =  @ARD_kernelfun;
    case 'Matern52'
        base_kernelfun = @Matern52_kernelfun;
    case 'Matern32'
        base_kernelfun = @Matern32_kernelfun;
end

ns = 0;
task_type = 'max';
update_period = 1000;

% IMPORTANT
identification = 'mu_g';
model = gp_preference_model(d, meanfun, base_kernelfun, regularization, ...
    hyperparameters, lb,ub, 'preference', link, modeltype, kernelname, condition, 0);
optim = preferential_BO([], task_type, identification, maxiter, nopt, nopt, update_period, 'all', acq, d,  ns);

%% Compute the kernel approximation if needed
if strcmp(model.kernelname, 'Matern52') || strcmp(model.kernelname, 'Matern32') || strcmp(kernelname, 'ARD')
    approximation.method = 'RRGP';
else
    approximation.method = 'SSGP';
end
approximation.decoupled_bases = 1;
approximation.nfeatures = 6561;
model = approximate_kernel(model, theta, approximation);



hiloModel.theta = theta;
hiloModel.model = model;
hiloModel.optim = optim;
hiloModel.maxiter = maxiter;
hiloModel.nopt = nopt;
hiloModel.update_period = update_period;
hiloModel.approximation = approximation;
hiloModel.post = post;
hiloModel.rand_acq = rand_acq;
hiloModel.ub = ub;
hiloModel.lb = lb;


end

