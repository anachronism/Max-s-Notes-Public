function [predictions,posterior] = classify_ensemble(net, data, labels)
%%% TODO: MAKE INTO REGRESSION.
n_experts = length(net.classifiers);
weights = net.w(end,:);
if n_experts ~= length(weights)
  error('What are there are different number of weights and experts!')
end
p = zeros(n_experts, size(data,1));
for k = 1:n_experts
  p(k,:) = classifier_test(net.classifiers{k}, data);
  
  % this is inefficient, but it does the job 
  %%% APply weights to regression
end

%%% Weighted Median
    % Sort by weight size (small -> large)
    [weights_sorted,ind_weights] = sort(weights,'ascend'); %%% TODO: sort by Y? not sure how to do with a vector.
    % sum log(1/beta_t) until we get to the smallest t where the sum is g>=
    % 1/2 total sum.
    runningSum = 0;
    compSum = sum(log10(1./net.beta(net.t-1,:)))/2;
    for i = 1:n_experts
        runningSum = runningSum + log10(1/net.beta(net.t-1,i)); %%% TODO: check if net.t or net.t-1
        if runningSum >= compSum
            ind_weightMed = i;
        end
    end
    % Take y_t as the value.
    ind_result = ind_weights(ind_weightMed);
    predictions = p(ind_result,:);
% [~,predictions] = max(p');
predictions = predictions';
posterior = 0; %%% FIX %p./repmat(sum(p,2),1,net.mclass);
