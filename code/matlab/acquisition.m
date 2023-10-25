function newDuel = acquisition(hiloModel, xtrain, ctrain, iter)
%ACQUISITION Summary of this function goes here
if nargin < 4
    iter = hiloModel.maxiter;
end
ub = hiloModel.ub;
lb = hiloModel.lb;
d = numel(ub);
if iter <= hiloModel.nopt
    x_duel1 = hiloModel.rand_acq();
    x_duel2 = hiloModel.rand_acq();
    newDuel = [x_duel1; x_duel2];
else
    xtrain_norm = (xtrain - [lb; lb])./([ub; ub]- [lb; lb]);
    new_x = hiloModel.optim.acquisition_fun(hiloModel.theta, xtrain_norm, ...
        ctrain, hiloModel.model, hiloModel.post, hiloModel.approximation, hiloModel.optim);
    % already unnormalized
    x_duel1 = new_x(1:d);
    x_duel2 = new_x((d+1):end);
    newDuel= [x_duel1; x_duel2];
end



end

