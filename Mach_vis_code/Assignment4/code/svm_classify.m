% Starter code prepared by James Hays for Computer Vision

function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

% This function should train a linear SVM for every category (i.e., one vs all)
% and then use the learned linear classifiers to predict the category of
% every test image. Every test feature will be evaluated with all 15 SVMs
% and the most confident SVM will "win".
% 
% Confidence, or distance from the margin, is W*X + B:
% The learned hyperplane is represented as:
% - W, a row vector
% - B, a scalar bias or offset
% X is a column vector representing the feature, and
% * is the inner or dot product.

%
% A Strategy
% 
% - Use fitclinear() to train a 'one vs. all' linear SVM classifier.
% - Observe the returned model - it defines a hyperplane with weights 'Beta' and 'Bias'.
% - For a test feature point, manually compute the distance form the hyperplane using the model parameters.
% - Store the confidence.
% - Once you have a confidence for every category, assign the most confident category.
% 

% unique() is used to get the category list from the observed training category list. 
% 'categories' will not be in the same order as unique() sorts them. This shouldn't really matter, though.
categories = unique(train_labels);
num_categories = length(categories);
ind_compare = 1:length(train_labels);
predicted_categories_int = zeros(length(test_image_feats),1);

% Train SVM models for each category.
for i = 1:num_categories
    index_labels = find(strcmp(train_labels, categories(i)));
    cat_labels = ismember(ind_compare,index_labels);
    cat_models{i} = fitcsvm(train_image_feats,cat_labels,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf'); % A linear kernel I think fits the fitclinear, couldn't get a better result that way though.
end


% Classify test images based on the svm's that were trained.
for i = 1:length(test_image_feats)
    confidence = zeros(1,num_categories);
    for j = 1:num_categories
        [~,conf_cat] = predict(cat_models{j},test_image_feats(i,:));
        confidence(j) = conf_cat(2);
        %confidence(j) = dot(cat_models{j}.Beta,test_image_feats(i,:)) + cat_models{j}.Bias; 
    end
    [~,predicted_categories_int(i)] = max(confidence);
end

%%% Change this to be the actual labels.
predicted_categories = categories(predicted_categories_int);

