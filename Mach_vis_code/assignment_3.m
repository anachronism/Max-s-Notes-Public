clearvars;
close all;
faceDir = 'faces';
[images,personID,lightingCond,subset] = readFaceImages(faceDir);
x = pca_50x50(images);
y = 1;