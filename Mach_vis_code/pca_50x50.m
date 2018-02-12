%% pca_50x50. Vectorize and run PCA on the images. 
% Inputs:
% Outputs:
function [x_norm,eig_vec,lambda] = pca_50x50(images,d)
    num_img = length(images);
    for i = 1:size(images,1)
        for j = 1:size(images,2)
            tmp = images{i,j};
            images{i,j} = (tmp - mean2(tmp))/std2(tmp);
        end
    end
    x = cell2feat(images);
    x_norm = zeros(num_img,2500);
    

    x_mean = mean(x,1);    
    mean_rep = repmat(x_mean,num_img,1);
    x_noMean = x - mean_rep;
    x_norm = x_noMean;
    cov = ( x_noMean * x_noMean.'); %Should I normalize by 1/n?
    [U,L] = eig(cov);
    
    U = rot90(rot90(U));
    eig_vec = x_noMean.'*U;
    
    eig_vec = eig_vec(:,1:d);
    %eig_vec = rot90(rot90(eig_vec));
    lambda = L(:,end-d+1:end);
    lambda = rot90(rot90(lambda));
    
end