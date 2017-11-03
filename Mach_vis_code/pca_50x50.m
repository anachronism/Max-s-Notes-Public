%% pca_50x50. Vectorize and run PCA on the images. 
% Inputs:
% images:   Cell Array of BW images. First dim is the index of the image, the 
%           other two are the dims of the image itself (50x50).
function [x_norm,eig_vec,lambda] = pca_50x50(images,d)
    num_img = length(images);
    x = zeros(num_img, 2500);
    x_norm = zeros(num_img,2500);

    for i = 1:num_img
        x(i,:) = reshape(images{i},[1,2500]);
        x_norm(i,:) = (x(i,:) - mean(x(i,:)))/std(x(i,:));
    end
   
    x_mean = mean(x,1);
    mean_rep = repmat(x_mean,num_img,1);
    x_noMean = x_norm - mean_rep;
    cov = 1/(num_img-1) *( x_noMean.' * x_noMean); %There may be a trick discussed in the class about this part.
    [U,V] = eig(cov);
    eig_vec = U(:,1:d);
    lambda = V(:,1:d);
    
end