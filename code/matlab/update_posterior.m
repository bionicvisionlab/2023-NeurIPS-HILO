function hiloModel = update_posterior(hiloModel, xtrain, ctrain)
%UPDATE_POSTERIOR 
lb = hiloModel.lb;
ub = hiloModel.ub;
xtrain_norm = (xtrain - [lb; lb])./([ub; ub]- [lb; lb]);
hiloModel.post =  hiloModel.model.prediction(hiloModel.theta, xtrain_norm, ctrain, [], hiloModel.post);
end

