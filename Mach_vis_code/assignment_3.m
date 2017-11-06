clearvars;
close all;


faceDir = 'faces';
dim = 30;

[images,personID,lightingCond,subset] = readFaceImages(faceDir);
[x_norm,pcaSpace,lambda] = pca_50x50(images,dim);
x_encoded = pcaSpace.' * x_norm.';
% Assume testData is in the same format as x_norm.
labels = nearestNeighbors(testData,x_encoded,pcaSpace);

% Look at eigenfaces generated.
x_mean = mean(x_norm,1);
testImg = x_mean + pcaSpace(:,1).';
testImg = reshape(testImg,[50,50]);

figure();
imshow(testImg);