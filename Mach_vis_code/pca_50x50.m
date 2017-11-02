%% pca_50x50. Vectorize and run PCA on the images. 
% Inputs:
% images:   Array of BW images. First dim is the index of the image, the 
%           other two are the dims of the image itself (50x50).
function [eig_vec,lambda_sorted] = pca_50x50(images,d)
    x = reshape(images, [2500, size(images,1)]); % CHECK THAT THIS RESHAPE ACTUALLY WORKS
    % SCALE x
    x_mean = 1/size(images,1) * sum(images,2); % Average the image. 
    
    %% Make non-iterative
    cov = zeros(size(images,2),size(images,2));
    for i = 1:size(images,1)
        cov = cov + (x(:,i)-x_mean) * (x(:,i)-x_mean).';
    end
    cov = cov .* 1/size(images,1);
    
    [u,lambda] = eig(cov.'*cov);
    [lambda_sorted, ind] = sort(lambda,'descend');
    u_sorted = u(ind);
    v = u_sorted.'*x;
    eig_vec = v(1:d);
    
end