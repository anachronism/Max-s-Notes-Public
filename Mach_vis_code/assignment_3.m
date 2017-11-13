clearvars;
close all;


faceDir = 'faces';
%dim1 = 9;
dim1 = 60;
dim2 = 30;



[images,personID,lightingCond,subset] = readFaceImages(faceDir);
images = images.';
%images = normImages(images);

% If I had more time, I'd make the subsets into objects, so that one object
% can hold both the image and the truth label.
person1 = images(personID == 1);
person2 = images(personID == 2);
person3 = images(personID == 3);
person4 = images(personID == 4);
person5 = images(personID == 5);
person6 = images(personID == 6);
person7 = images(personID == 7);
person8 = images(personID == 8);
person9 = images(personID == 9);
person10 = images(personID == 10);

% Create subgroups and truth labels.
subset1 = [person1(1:7);person2(1:7);person3(1:7);person4(1:7);person5(1:7);...
    person6(1:7);person7(1:7);person8(1:7);person9(1:7);person10(1:7)];
subset2 = [person1(8:19);person2(8:19);person3(8:19);person4(8:19);person5(8:19);...
    person6(8:19);person7(8:19);person8(8:19);person9(8:19);person10(8:19)];
subset3 = [person1(20:31);person2(20:31);person3(20:31);person4(20:31);person5(20:31);...
    person6(20:31);person7(20:31);person8(20:31);person9(20:31);person10(20:31)];
subset4 = [person1(32:45);person2(32:45);person3(32:45);person4(32:45);person5(32:45);...
    person6(32:45);person7(32:45);person8(32:45);person9(32:45);person10(32:45)];
subset5 = [person1(46:64);person2(46:64);person3(46:64);person4(46:64);person5(46:64);...
    person6(46:64);person7(46:64);person8(46:64);person9(46:64);person10(46:64)];
% Labels
labels_s1 = [ones(7,1);2*ones(7,1);3*ones(7,1);4*ones(7,1);5*ones(7,1);...
    6*ones(7,1);7*ones(7,1);8*ones(7,1);9*ones(7,1);10*ones(7,1)];
labels_s2 = [ones(12,1);2*ones(12,1);3*ones(12,1);4*ones(12,1);5*ones(12,1);...
    6*ones(12,1);7*ones(12,1);8*ones(12,1);9*ones(12,1);10*ones(12,1)];
labels_s3 = [ones(12,1);2*ones(12,1);3*ones(12,1);4*ones(12,1);5*ones(12,1);...
    6*ones(12,1);7*ones(12,1);8*ones(12,1);9*ones(12,1);10*ones(12,1)];
labels_s4 = [ones(14,1);2*ones(14,1);3*ones(14,1);4*ones(14,1);5*ones(14,1);...
    6*ones(14,1);7*ones(14,1);8*ones(14,1);9*ones(14,1);10*ones(14,1)];
labels_s5 = [ones(19,1);2*ones(19,1);3*ones(19,1);4*ones(19,1);5*ones(19,1);...
    6*ones(19,1);7*ones(19,1);8*ones(19,1);9*ones(19,1);10*ones(19,1)];

%% Part 1:
[s1_norm,pcaSpace,lambda] = pca_50x50(subset1,dim1);
[test_norm,pcaSpace_larger,lambda2] = pca_50x50([subset1;subset5],dim1);
mu_pcaSpace = mean(pcaSpace,2);
%s1_encoded =  pcaSpace.'*s1_norm;

s1_encoded = pcaSpace.'* s1_norm.';
s1_encoded_larger = pcaSpace_larger.'*test_norm.';
% Get a mean for each label.
% Since there are labels, k-means doesn't apply, but the concept still
% should.
%labelMeans = getCenters(s1_encoded,labels_s1,dim1);


%% PUT OFF Fisher training until later.
% Fisher face would expand here.

% Assume testData is in the same format as x_norm.
%labels = nearestNeighbors(testData,x_encoded,pcaSpace);
s1_in = cell2feat(subset1);
s1_in = (s1_in - repmat(mean(s1_in,1),size(s1_in,1),1))./repmat(std(s1_in,1),size(s1_in,1),1);
s1_noMean = s1_in - repmat(mean(s1_in,1),size(s1_in,1),1);

[s1,s1_labels] = nearestNeighbors(s1_noMean,s1_encoded,labels_s1,pcaSpace);
correct = length(find(labels_s1 == s1_labels));
total = length(labels_s1);
s1_stats = correct/total
[s1_larger, s1_labels_larger] = nearestNeighbors(s1_noMean,s1_encoded_larger,[labels_s1;labels_s5],pcaSpace_larger);
correct = length(find(labels_s1 == s1_labels_larger));
total = length(labels_s1);
s1_stats_larger = correct/total
%
s2_in = cell2feat(subset2);
s2_in = (s2_in - repmat(mean(s2_in,1),size(s2_in,1),1))./repmat(std(s2_in,1),size(s2_in,1),1);
s2_noMean = s2_in - repmat(mean(s1_in,1),size(s2_in,1),1);
s2_eigSpace = pcaSpace.' * s2_noMean.';
[s2,s2_labels] = nearestNeighbors(s2_noMean,s1_encoded,labels_s1,pcaSpace);
correct = length(find(labels_s2 == s2_labels));
total = length(labels_s2);
s2_stats = correct/total

[s2_larger, s2_labels_larger] = nearestNeighbors(s2_noMean,s1_encoded_larger,[labels_s1;labels_s5],pcaSpace_larger);
correct = length(find(labels_s2 == s2_labels_larger));
total = length(labels_s2);
s2_stats_larger = correct/total

%
s3_in = cell2feat(subset3);
s3_in = (s3_in - repmat(mean(s3_in,1),size(s3_in,1),1))./repmat(std(s3_in,1),size(s3_in,1),1);
s3_noMean = s3_in - repmat(mean(s1_in,1),size(s3_in,1),1);
s3_eigSpace = pcaSpace.' * s3_noMean.';
[s3,s3_labels] = nearestNeighbors(s3_noMean,s1_encoded,labels_s1,pcaSpace);

correct = length(find(labels_s3 == s3_labels));
total = length(labels_s3);
s3_stats = correct/total
[s3_larger, s3_labels_larger] = nearestNeighbors(s3_noMean,s1_encoded_larger,[labels_s1;labels_s5],pcaSpace_larger);
correct = length(find(labels_s3 == s3_labels_larger));
total = length(labels_s3);
s3_stats_larger = correct/total

%
s4_in = cell2feat(subset4);
s4_in = (s4_in - repmat(mean(s4_in,1),size(s4_in,1),1))./repmat(std(s4_in,1),size(s4_in,1),1);
s4_noMean = s4_in - repmat(mean(s1_in,1),size(s4_in,1),1);
s4_eigSpace = pcaSpace.' * s4_noMean.';
[s4,s4_labels] = nearestNeighbors(s4_noMean,s1_encoded,labels_s1,pcaSpace);

correct = length(find(labels_s4 == s4_labels));
total = length(labels_s4);
s4_stats = correct/total
[s4_larger, s4_labels_larger] = nearestNeighbors(s4_noMean,s1_encoded_larger,[labels_s1;labels_s5],pcaSpace_larger);
correct = length(find(labels_s4 == s4_labels_larger));
total = length(labels_s4);
s4_stats_larger = correct/total

%%%
s5_in = cell2feat(subset5);
s5_in = (s5_in - repmat(mean(s5_in,1),size(s5_in,1),1))./repmat(std(s5_in,1),size(s5_in,1),1);
s5_noMean = s5_in - repmat(mean(s1_in,1),size(s5_in,1),1);
s5_eigSpace = pcaSpace.' * s5_noMean.';
[s5,s5_labels] = nearestNeighbors(s5_noMean,s1_encoded,labels_s1,pcaSpace);

correct = length(find(labels_s5 == s5_labels));
total = length(labels_s5);
s5_stats = correct/total
[s5_larger, s5_labels_larger] = nearestNeighbors(s5_noMean,s1_encoded_larger,[labels_s1;labels_s5],pcaSpace_larger);
correct = length(find(labels_s5 == s5_labels_larger));
total = length(labels_s5);
s5_stats_larger = correct/total

% Display Eigenfaces
figure(1);
subplot(3,3,1)
axis image, axis off;
colormap gray;
imagesc(reshape(pcaSpace(:,1),[50,50]));
subplot(3,3,2)
axis image, axis off;
colormap gray;
imagesc(reshape(pcaSpace(:,2),[50,50]));
subplot(3,3,3)
axis image, axis off;
colormap gray;
imagesc(reshape(pcaSpace(:,3),[50,50]));
subplot(3,3,4)
axis image, axis off;
colormap gray;
imagesc(reshape(pcaSpace(:,4),[50,50]));
subplot(3,3,5)
axis image, axis off;
colormap gray;
imagesc(reshape(pcaSpace(:,5),[50,50]));
subplot(3,3,6)
axis image, axis off;
colormap gray;
imagesc(reshape(pcaSpace(:,6),[50,50]))
subplot(3,3,7)
axis image, axis off;
colormap gray;
imagesc(reshape(pcaSpace(:,7),[50,50]));
subplot(3,3,8)
axis image, axis off;
colormap gray;
imagesc(reshape(pcaSpace(:,8),[50,50]));
subplot(3,3,9)
axis image, axis off;
colormap gray;
imagesc(reshape(pcaSpace(:,9),[50,50]));

%
figure(2);
subplot(2,5,1)
axis image, axis off;
colormap gray;
imagesc(person1{1});
subplot(2,5,2)
axis image, axis off;
colormap gray;
imagesc(person2{1});
subplot(2,5,3)
axis image, axis off;
colormap gray;
imagesc(person3{1});
subplot(2,5,4)
axis image, axis off;
colormap gray;
imagesc(person4{1});
subplot(2,5,5)
axis image, axis off;
colormap gray;
imagesc(person5{1});
subplot(2,5,6)
axis image, axis off;
colormap gray;
s1_reconst = repmat(mu_pcaSpace,1,size(s1_encoded,2)) + pcaSpace * s1_encoded;
imagesc(reshape(s1_reconst(:,1),[50,50]));
subplot(2,5,7)
axis image, axis off;
colormap gray;
s2_reconst = repmat(mu_pcaSpace,1,size(s2_eigSpace,2)) +pcaSpace * s2_eigSpace;
imagesc(reshape(s2_reconst(:,1),[50,50]));
subplot(2,5,8)
axis image, axis off;
colormap gray;
s3_reconst = repmat(mu_pcaSpace,1,size(s3_eigSpace,2)) + pcaSpace * s3_eigSpace;
imagesc(reshape(s3_reconst(:,1),[50,50]));
subplot(2,5,9)
axis image, axis off;
colormap gray;
s4_reconst = repmat(mu_pcaSpace,1,size(s4_eigSpace,2)) +pcaSpace * s4_eigSpace;
imagesc(reshape(s4_reconst(:,1),[50,50]));
subplot(2,5,10)
axis image, axis off;
colormap gray;
s5_reconst = repmat(mu_pcaSpace,1,size(s5_eigSpace,2)) + pcaSpace * s5_eigSpace;
imagesc(reshape(s5_reconst(:,1),[50,50]));

%% PART 2: FISHER
c1 = 10;
c2 = 31;
[s_norm,W_pca,~] = pca_50x50(subset1,c1);
W_pca = W_pca.' * s_norm.'; 
mu_pca = mean(W_pca,2);
S_i = zeros(70);
S_w = zeros(70);
S_b = zeros(70);

for i = 1:7
    mu_class(i,:) = mean(W_pca(:,(i-1)*10+1:i*10),2);
    for j = 1:c1
        S_i = S_i + (W_pca(:,j).'-mu_class(i,:)) *(W_pca(:,j).'-mu_class(i,:)).'; 
    end
    S_w = S_w + S_i;
    S_b = S_b + (mu_class(i,:)-mu_pca.') *(mu_class(i,:)-mu_pca.').';
    
end
