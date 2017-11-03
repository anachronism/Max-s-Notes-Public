%% Nearest Neighbors. Classify images using nearest neighbors and L2 Norm. 
% Inputs:
% Outputs:
function labels = nearestNeighbors(test,comparison,eigSpace)
    num_tests = size(test,1);
    num_compare = size(comparison,1); % VERIFY THIS IS THE CORRECT DIM
    
    labels = zeros(num_tests,1);
    ssd_store = zeros(num_compare,1);
    
    % n by 2500
    for i = 1:num_tests
        test_eig = eigSpace.'* test(i,:);
        for j = 1:num_compare
            ssd_store(j) = sum((test_eig - comparison(j))^2)% Sum of (test_eig - comparison(j))^2
        end
        [~,labels(i)] = min(ssd_store);
        % label(i) = index of min(ssd_store);
    end
end