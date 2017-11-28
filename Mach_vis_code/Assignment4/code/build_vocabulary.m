% Starter code prepared by James Hays for Computer Vision

% This function will extract a set of feature descriptors from the training images,
% cluster them into a visual vocabulary with k-means,
% and then return the cluster centers.

% Notes:
% - To save computation time, we might consider sampling from the set of training images.
% - Per image, we could randomly sample descriptors, or densely sample descriptors,
% or even try extracting descriptors at interest points.
% - For dense sampling, we can set a stride or step side, e.g., extract a feature every 20 pixels.
% - Recommended first feature descriptor to try: HOG.

% Function inputs: 
% - 'image_paths': a N x 1 cell array of image paths.
% - 'vocab_size' the size of the vocabulary.

% Function outputs:
% - 'vocab' should be vocab_size x descriptor length. Each row is a cluster centroid / visual word.

function vocab = build_vocabulary( image_paths, vocab_size )
training_downsamp = 5;
len_descriptor = 9;
size_img = size(imread(image_paths{1}));
blks_per_img = floor((size_img ./ [8 8]-[2 2])./([2 2] - [1 1]) + 1);
N_features = prod([blks_per_img, [2 2],len_descriptor]);
features=zeros(len_descriptor, N_features/len_descriptor,length(image_paths)/training_downsamp);

for i = 1:length(image_paths)
    if mod(i,training_downsamp) == 0
        img = imread(image_paths{i});
        % Resize based on first image (Can probably get rid of this if I modify the reshape).
        size_curr = size(img);
        dim1_large = size_curr(1) > size_img(1);
        dim2_large = size_curr(2) > size_img(2);
        dim1_small = size_curr(1) < size_img(1);
        dim2_small = size_curr(2) < size_img(2);
        if dim1_small || dim2_small || dim1_large || dim2_large
            img = imresize(img, size_img);
        end

        hold = extractHOGFeatures(img,'NumBins',len_descriptor);
        features(:,:,i/training_downsamp) = reshape(hold,[len_descriptor,N_features/len_descriptor]);
    end
end
% Start K-means clustering.
features_reshape = reshape(features,[9,N_features/9*length(image_paths)/training_downsamp]);
[~,vocab] = kmeans(features_reshape.',vocab_size);
end
