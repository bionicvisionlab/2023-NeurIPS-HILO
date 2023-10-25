function theta = get_theta(xtrain, ctrain, kernelname, ub, lb)
%GET_THETA Gives the maximally likely theta for the training data
% ctrain = y values = probability of selecting option 1 over option 2
%   xtrain should be (d*2, npoints)
%  lb, ub, should be col vectors
% c should be row vector
addpath(genpath('.'));
kernelnames = {'Matern52', 'ARD', 'Gaussian', 'Matern32'};
if ~any(strcmp(kernelnames,kernelname))
    error('Invalid kernelname')
end

d = numel(ub);
theta.cov = [0, 0];
theta.mean = 0;
regularization = 'nugget';

condition.x0 = zeros(d,1);
condition.y0 = 0;
link = @normcdf;
modeltype = 'exp_prop';



switch(kernelname)
    case 'Gaussian'
        base_kernelfun =  @Gaussian_kernelfun;
        theta.cov = -4*ones(2,1);
    case 'ARD'
        base_kernelfun =  @ARD_kernelfun;
        theta.cov = zeros(d+1, 1);
    case 'Matern52'
        base_kernelfun = @Matern52_kernelfun;
    case 'Matern32'
        base_kernelfun = @Matern32_kernelfun;
end
meanfun = @constant_mean;
hyperparameters.ncov_hyp =numel(theta.cov); % number of hyperparameters for the covariance function
hyperparameters.nmean_hyp =1; % number of hyperparameters for the mean function
hyperparameters.hyp_lb = -10*ones(hyperparameters.ncov_hyp  + hyperparameters.nmean_hyp,1);
hyperparameters.hyp_ub = 10*ones(hyperparameters.ncov_hyp  + hyperparameters.nmean_hyp,1);
model = gp_preference_model(d, meanfun, base_kernelfun, regularization, ...
    hyperparameters, lb,ub, 'preference', link, modeltype, kernelname, condition, 0);

%% Compute the kernel approximation if needed
if strcmp(model.kernelname, 'Matern52') || strcmp(model.kernelname, 'Matern32') || strcmp(kernelname, 'ARD')
    approximation.method = 'RRGP';
else
    approximation.method = 'SSGP';
end
approximation.decoupled_bases = 1;
approximation.nfeatures = 6561;
model = approximate_kernel(model, theta, approximation);

xtrain_norm = (xtrain - [lb; lb])./([ub; ub]- [lb; lb]);
theta = model.model_selection(xtrain_norm, ctrain, theta, 'all');



