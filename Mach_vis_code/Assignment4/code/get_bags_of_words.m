% Starter code prepared by James Hays for Computer Vision

%This feature representation is described in the handout, lecture
%materials, and Szeliski chapter 14.

function image_feats = get_bags_of_words(image_paths)
% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x feature vector length
% matrix 'vocab' where each row is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every run.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram
% ('vocab_size') below.

% You will want to construct feature descriptors here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% feature descriptors will look very different from a smaller version of the same
% image.

load('vocab.mat')
vocab_size = size(vocab, 1);
num_images = length(image_paths);
testing_downsamp = 1;
len_descriptor = 9;
image_feats = zeros(num_images,vocab_size);

size_img = size(imread(image_paths{1}));
blks_per_img = floor((size_img ./ [8 8]-[2 2])./([2 2] - [1 1]) + 1);
N_features = prod([blks_per_img, [2 2],len_descriptor]);
features=zeros(len_descriptor, N_features/len_descriptor,num_images/testing_downsamp);

for i = 1:num_images
    if mod(i,testing_downsamp) == 0
        img = imread(image_paths{i});    
        % Resize based on first image.
        size_curr = size(img);
        dim1_large = size_curr(1) > size_img(1);
        dim2_large = size_curr(2) > size_img(2);
        dim1_small = size_curr(1) < size_img(1);
        dim2_small = size_curr(2) < size_img(2);
        if dim1_small || dim2_small || dim1_large || dim2_large
            img = imresize(img, size_img);
        end

        hold = extractHOGFeatures(img,'NumBins',len_descriptor);
        features(:,:,i/testing_downsamp) = reshape(hold,[len_descriptor,N_features/len_descriptor]);
    end
end

%features_reshape = reshape(features,[9,N_features/9*num_images/testing_downsamp]);

% For each image, do NN search for each feature and add to count of the features. 
for i = 1:num_images/testing_downsamp
    current_features = reshape(features(:,:,i/testing_downsamp),[9,N_features/9]);
    % For each feature in the current image, match with a word:
    for j = 1:N_features/9
        [ind,dist] = nnL2_bagOfWords(current_features(:,j),vocab);
        image_feats(i,ind) = image_feats(i,ind) + 1;
    end
    % Normalize.
    image_feats(i,:) = image_feats(i,:);%/sum(image_feats(i,:));
end




