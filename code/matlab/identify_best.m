function currentBest = identify_best(hiloModel, xtrain, ctrain)
%IDENTIFY_BEST get the most likely parameters
ub = hiloModel.ub;
lb = hiloModel.lb;
xtrain_norm = (xtrain - [lb; lb])./([ub; ub]- [lb; lb]);
x_best_norm  = hiloModel.optim.identify(hiloModel.model, hiloModel.theta, xtrain_norm, ctrain, hiloModel.post);                
currentBest = x_best_norm .*(ub - lb) + lb;
end

