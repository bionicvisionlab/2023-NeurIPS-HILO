function probs = get_probs_theta(x, c, theta, kernelname, ub, lb, ntrain)
%GET_PROBS_THETA predicted probabilities for x
% x should be (2*d, npoints)
% c should be row vector
% ub and lb should be column vectors
if nargin < 7
    ntrain = round(size(c) * 0.5);
end

theta.mean = 0;

addpath(genpath('.'));
kernelnames = {'Matern52', 'ARD', 'Gaussian', 'Matern32'};
if ~any(strcmp(kernelnames,kernelname))
    error('Invalid kernelname')
end

d = numel(ub);
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


xnorm = (x - [lb; lb])./([ub; ub]- [lb; lb]);

test_idx = ntrain+1;
if ntrain == size(c)
    test_idx=1;
end

probs =  model.prediction(theta, xnorm(:,1:ntrain+1), c(1:ntrain+1), xnorm(:, test_idx:end), []);






