%Initialization
clc;
clear;
close;
%We get the input image
%Just insert image to process
Original_image = imread('IR_0100.png');

%Segmentation part to extract only the body
%If the image is not a greyscale image we transform it
[~, ~, numberOfColorChannels] = size(Original_image);
if numberOfColorChannels > 1
    GreyS_image = rgb2gray(Original_image);
end

%We apply the mask with threshold 50
binaryImage = GreyS_image < 50;

% Clean the image by filling the background pixel for better quality
binaryImage = imfill(binaryImage, 'holes');

% Apply closing and dilatation method to the image 
se_close = strel('disk',10);
closed_mask = imclose(binaryImage, se_close);

se_dilate = strel('disk',10);
dilated_mask = imdilate(closed_mask, se_dilate);

%Revert the image
mask = imcomplement(dilated_mask);

%Segmentation of the image
Seg = GreyS_image;
Seg(~mask) = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Thresholding algorithm
Simage=mat2gray(Seg);

%Use median operator with local neighborhood of 15 pixel and constant of 0,03
%Apply median filter
median=medfilt2(Simage,[15 15]);

%Deduct the median image from the segmented image
median_result=median-Simage;

%Imbinzarize the image
Nipple_candidate=imbinarize(median_result,0.03);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Nipple Selection
%First rule is already implemented by image segmentation

%2nd rule, Nipples occupy 2 small regions of the body
%Take off too small regions
Nipple_candidate_filtered = bwareaopen(Nipple_candidate,20);


%3rd rule, Nipples are not in upper or lower part of thermogram
%Using a threshold we will exclude these parts of the image
[Width, Length] = size(Nipple_candidate_filtered);
T_height_up = 0.35 * Width;
T_height_down = 0.3 * Width;

%Draw the part to keep
imshow(Nipple_candidate_filtered);
h = drawrectangle('Position',[0,Width-(T_height_down+T_height_up),Length,Width-(T_height_down+T_height_up)]);

%Create the mask
BW = createMask(h);
mask = imcomplement(BW);
%Delete the upper and lower part
Segm = Nipple_candidate_filtered;
Segm(~BW) = 0;


%4rth fact, divide in two regions
center = Length/2;

%5th fact, Nipples are round

% Clean the image by fillin background pixel for better quality
im = imfill(Segm, 'holes');

%Calculate centroid to determin if it belongs to right or left
%Calculate circulatrity and area of the candidates to get the nipples

stats =  regionprops(im,'Centroid','Circularity','Area');

%All our candidates are now in stats variable
%Analysis of the candidate using centroids ,circularity and Area as criteria
%Start by separating left and right region candidates
%We fill two dictionnaries with the candidates and their characteristics

leftC=struct('Centroid',[],'Circularity',{},'Area',{});
rightC=struct('Centroid',[],'Circularity',{},'Area',{});
for i = 1:length(stats)
    if stats(i).Centroid(1) < center
    leftC(end+1) = stats(i);
    else
        rightC(end+1) = stats(i);
    end
end    

 %For each right and left candidates we determin wich is the nipple
 %by comparing first circularity and taking index of it
 [~,Lc] = max([leftC.Circularity]);
 left_nipple_index_maxC = Lc;
 [~,Rc] = max([rightC.Circularity]);
 right_nipple_index_maxC = Rc;
 
 %Then we compare the area
 [~,La] = max([leftC.Area]);
 left_nipple_index_maxA = La;
 [~,Ra] = max([rightC.Area]);
 right_nipple_index_maxA = Ra;
 
 %If the index of the candidate with biggest Circularity correspond
 %The index of the biggest area we skip else we chose the one with biggest
 %Area
 
 if left_nipple_index_maxC == left_nipple_index_maxA
    Left_nipple_index = left_nipple_index_maxC;
 else
     Left_nipple_index = left_nipple_index_maxA;
 end
 %We do the same with the right part
 if right_nipple_index_maxC == right_nipple_index_maxA
    right_nipple_index = right_nipple_index_maxC;
 else
     right_nipple_index = right_nipple_index_maxA;
 end
 
 %Add markers on the nipple position
 
 pos = [leftC(Left_nipple_index).Centroid;rightC(right_nipple_index).Centroid]; 
 final = insertMarker(Original_image,pos,'color','green','size',7);
 
 %Final plot of the process
 subplot(2,2,1);
 imshow(Original_image);
 title('Original Image');
 subplot(2,2,2);
 imshow(Seg);
 title('Segmented Image');
 subplot(2,2,3);
 imshow(Nipple_candidate);
 title('Nipple candidate');
 subplot(2,2,4);
 imshow(final);
 title('Final result');
 
 
 
 
     
     
 
 
     
     
 
 
 

     
     

















