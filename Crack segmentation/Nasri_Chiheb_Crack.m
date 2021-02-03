% Crack Segmentation using deep learning
% This programm runs crack segmentation using U-net
%Due to compilation timea pre-trained network is available:
% 1 - If you want to train again the net go line 72
% 2 - If you want to use per-trained network go line 100
%Default configuration is 2
clc;
close all;
clear;

%% Load the Data

%Load training images and pixel labels into the workspace.
imageDir = fullfile('train_img');
labelDir = fullfile('train_lab');
%Create an imageDatastore object to store the training images.
imds = imageDatastore(imageDir);
%Define the class names and their associated label IDs.
%Label IDs correspond the the pixel value, Here we use binary images images
%so IDs are 255 = white for crack and 0 = black for noCrack
classNames = ["CRACK","noCrack"];
labelIDs = [255 0];
%Create a pixelLabelDatastore object to store the ground truth pixel labels
%for the training images.
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);


%% Prepocessing

%We resize the images for performance needs but also the network needs
%images of the same size
inputSize = [64 64];
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);
pxds.ReadFcn = @(loc)imresize(imread(loc),inputSize);
%Create a datastore for training the network.
ds = pixelLabelImageDatastore(imds,pxds);

%% Create the U-Net network

%Preset options for the graph
imageSize = [64 64 3];
numClasses = 2;
lgraph = unetLayers(imageSize,numClasses);


%% Network training 

%Set training options
opts = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'L2Regularization',0.0005,... %Used to avoid overfitting
    'MaxEpochs',50,...
    'MiniBatchSize',64,... 
    'Shuffle','every-epoch',...
    'Plots','training-progress', ...
    'VerboseFrequency',10);

%Here our class proportion is imbalanced so we add weighted classes to our
%network
tbl = countEachLabel(ds);
totalNumberOfPixels = sum(tbl.PixelCount);
classFrequency = tbl.PixelCount / totalNumberOfPixels
classWeights = 1./classFrequency

%Change last layer to add weighted classes because unet layers are in read
%only mode
lgraph = removeLayers(lgraph,'Segmentation-Layer'); 
layerlast = pixelClassificationLayer('Classes',tbl.Name,'ClassWeights',...
classWeights,'Name','Segmentation-Layer'); 
layer_to_add = [layerlast]; 
lgraph = addLayers(lgraph,layer_to_add); 
lgraph = connectLayers(lgraph,'Softmax-Layer','Segmentation-Layer');

%Train network
% net = trainNetwork(ds,lgraph,opts)
%net1 = net
%save net

%% Test the trained network


%Load test images and pixel labels into the workspace.=====================
testImagesDir = fullfile('test_img');
%Create an imageDatastore object holding the test images.
imdst = imageDatastore(testImagesDir);
%Define the location of the ground truth labels.
testLabelsDir = fullfile('test_lab');
%Create a pixelLabelDatastore object to store the ground truth pixel labels
%for the training images.We use same class names and label ids as for the
%training part
pxdsTruth = pixelLabelDatastore(testLabelsDir,classNames,labelIDs);

%Run a Semantic Segmentation Classifier====================================
%Load the net work
load('net2.mat');
net = net2;
%Run the network on the test images.
pxdsResults = semanticseg(imdst,net,"WriteLocation",tempdir);


%% Evaluate the Quality of the Prediction

%The predicted labels are compared to the ground truth labels.
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

%Display the classification accuracy, jaccard index, and the F-1
%score for each class in the data set.
metrics.ClassMetrics
%Display the Confusion Matrix
%metrics.ConfusionMatrix
%Visualize the normalized confusion matrix.
normConfMatData = metrics.NormalizedConfusionMatrix.Variables;
figure,
h = heatmap(classNames,classNames,100*normConfMatData);
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
h.Title = 'Normalized Confusion Matrix (%)';

%Visualize the histogram of the Jaccard index
imageIoU = metrics.ImageMetrics.MeanIoU;
figure
histogram(imageIoU)
title('Image Mean Jaccard Index')
%% %Show best example of prediction
%find the test image with the highest Jaccard Index.
[maxIoU, bestImageIndex] = max(imageIoU);
maxIoU = maxIoU(1);
bestImageIndex = bestImageIndex(1);
%Read, convert, and display the test image with the best Jaccard index with
%its ground truth and predicted labels.
bestTestImage = readimage(imdst,bestImageIndex);
bestTestImage = rgb2gray(bestTestImage);
bestTrueLabels = readimage(pxdsTruth,bestImageIndex);
bestPredictedLabels = readimage(pxdsResults,bestImageIndex);
bestTrueLabelImage = im2uint8(bestTrueLabels == classNames(1));
bestPredictedLabelImage = im2uint8(bestPredictedLabels == classNames(1));
bestMontage = cat(4,bestTestImage,bestTrueLabelImage,bestPredictedLabelImage);
bestMontage = imresize(bestMontage,4,"nearest");

figure, montage(bestMontage,'Size',[1 3])
title(['Test Image vs. Truth vs. Prediction. IoU = ' num2str(maxIoU)])
%% Plot of the metrics
evaluationMetrics = ["accuracy" "iou"];
%Compute these metrics for test data.
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth,"Metrics",evaluationMetrics);
%Display the chosen metrics for each class.
metrics.ClassMetrics
