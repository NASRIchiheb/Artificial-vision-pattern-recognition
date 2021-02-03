clc;
close all;
clear;

% To choose the model to predict with please refer to line 76
%Preset configuration = LBP + SVM
%Load dataset =============================================================

% Load path for training and test dataset
TrainSet   = fullfile("dataset", 'TrainSet');
TestSet = fullfile("dataset", 'TestSet');

% Get images from for test and train set using labels as folder names
trainingSet = imageDatastore(TrainSet,   'IncludeSubfolders',...
    true, 'LabelSource', 'foldernames');
testSet     = imageDatastore(TestSet, 'IncludeSubfolders',...
    true, 'LabelSource', 'foldernames');

% Prepare training set and feature extraction:=============================

% Pre-process the image before feature extraction.
% Extract HOG/LLBP features from each image for training set.
nImages = numel(trainingSet.Files);
for i = 1:nImages
    img = readimage(trainingSet, i);
    
    %Pre-process
    img = rgb2gray(img); %Convert the image in grey scale image
    img = imgaussfilt(img,2); %Apply gaussian filter
    img = imresize(img,[256 256]); %Resize the image
    img = histeq(img); %Apply HE
    
    %Extract HOG and LBP features
    trainingFeaturesHOG(i, :) = extractHOGFeatures(img, 'CellSize',...
        [2 2],'BlockSize',[4 4],'BlockOverlap',[2 2]);
    trainingFeaturesLBP(i, :) = extractLBPFeatures(img);

end

%Model training===========================================================
% Get labels for each image.
trainingLabels = trainingSet.Labels;


% Train models /  choose testFeaturesHOG to use HOG features
KNN_Model = fitcknn(trainingFeaturesLBP,trainingLabels,'NumNeighbors',10,'Standardize',1);
SVM_Model = fitcecoc(trainingFeaturesLBP,trainingLabels,'Coding','onevsall');


%Prepare test set and feature extraction:==================================

% Pre-process the image before feature extraction.
% Extract HOG/LLBP features from each image for test set.
nImages = numel(testSet.Files);
for i = 1:nImages
    img = readimage(testSet, i);
    
    %Pre-process
    img = rgb2gray(img); %Convert the image in grey scale image
    img = imgaussfilt(img,2); %Apply gaussian filter
    img = imresize(img,[256 256]); %Resize the image
    img = histeq(img); %Apply HE
    
    %Extract HOG and LBP features
    testFeaturesHOG(i, :) = extractHOGFeatures(img, 'CellSize',...
        [2 2],'BlockSize',[4 4],'BlockOverlap',[2 2]);
    testFeaturesLBP(i, :) = extractLBPFeatures(img);
    
end

%Model testing and prediction:=============================================
% Get labels for each image.
testLabels = testSet.Labels;

% Make predictions with the test set.
%choose KNN_Model for KNN prediction / choose testFeaturesHOG to use HOG
%features
predictedLabels = predict(SVM_Model, testFeaturesLBP);

%Resutls : ================================================================
%Create confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);
plotconfusion(testLabels, predictedLabels);

%Calculate the confusion matrix
acc=sum(diag(confMat))/sum(confMat(:));
fprintf('Accuracy = %4.2f %%\n',acc * 100);
