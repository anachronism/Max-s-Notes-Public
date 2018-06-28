function y = decision_ensemble(net, data, labels, n_experts)
if numel(labels) ~= 1
    y = zeros(numel(labels), n_experts);
else
    y = zeros(labels,n_experts);
end
     
for k = 1:n_experts
  y(:, k) = net.classifiers{k}(data);%classifier_test(net.classifiers{k}, data);
end
if n_experts > 1
    y = mean(y);
end