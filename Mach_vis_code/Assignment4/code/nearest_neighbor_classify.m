% Starter code prepared by James Hays for Computer Vision

%This function will predict the category for every test image by finding
%the training image with most similar features. Instead of 1 nearest
%neighbor, you can vote based on k nearest neighbors which will increase
%performance (although you need to pick a reasonable value for k).

function predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
% train_image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.
    predicted_categories=cell(size(test_image_feats,1),1);
    k = 5; % Number of neighbors to take into account.
% Useful functions:
%  matching_indices = strcmp(string, cell_array_of_strings)
%    This can tell you which indices in train_labels match a particular
%    category. Not necessary for simple one nearest neighbor classifier.
l2_norm = zeros(size(test_image_feats,1),1);
 for i = 1:size(test_image_feats,1)
    for j = 1:size(train_image_feats,1)
        l2_norm(j) = sum( (test_image_feats(i,:)- train_image_feats(j,:)).^2);
    end
    [ssd,ind_ssd] = sort(l2_norm, 'ascend');
    ssd_restrict = ssd(1:k);
    ind_restrict = ind_ssd(1:k);
    
    labels_restrict = train_labels(ind_restrict);
    ind_store = [];
    num_same_word = [];
    for m = 1:k
        labels_overlap = strcmp(labels_restrict{m},labels_restrict);
        ind_store = [ind_store;find(labels_overlap)];
        num_same_word = [num_same_word,sum(labels_overlap)];
    end
    [Y,ind_repeats] = max(num_same_word);
    predicted_categories{i} = labels_restrict{ind_repeats};
    
%   [Y,I] = MIN(X) if you're only doing 1 nearest neighbor, or
%   [Y,I] = SORT(X) if you're going to be reasoning about many nearest
%   neighbors 

 end










