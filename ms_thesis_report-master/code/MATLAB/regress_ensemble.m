function [predictions,posterior] = regress_ensemble(net, data, labels)
%%% TODO: MAKE INTO REGRESSION.
n_experts = length(net.classifiers);
weights = net.w(end,:);
if n_experts ~= length(weights)
  error('What are there are different number of weights and experts!')
end
p = zeros(n_experts, size(data,2));
normedWeights = weights/sum(weights);
predictions = zeros(1,size(data,2));
for k = 1:n_experts
  p(k,:) = net.classifiers{k}(data);
  predictions = predictions + p(k,:) * normedWeights(k);
end

posterior = 0; %%% FIX %p./repmat(sum(p,2),1,net.mclass);
