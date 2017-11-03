clearvars;
close all;


faceDir = 'faces';
dim = 30;

[images,personID,lightingCond,subset] = readFaceImages(faceDir);
[x_norm,pcaSpace,lambda] = pca_50x50(images,dim);



% Look at eigenfaces generated.
x_mean = mean(x_norm,1);
testImg = x_mean + pcaSpace(:,1).';
testImg = reshape(testImg,[50,50]);

figure();
imshow(testImg);