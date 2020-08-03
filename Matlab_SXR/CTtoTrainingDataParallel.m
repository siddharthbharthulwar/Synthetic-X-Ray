% Original Authors: 

%Copyright (C) 2018-2019
% Abhishek Moturu, moturuab@cs.toronto.edu
% Alex Chang, le.chang@mail.utoronto.ca

%Script Modified by:
%Siddharth Bharthulwar, siddharth.bharthulwar@colorado.edu

function CTtoTrainingDataParallel(CTFolderName, specificationsFileName, rotation)
CTFolderName = strcat(CTFolderName, '/');
warning('off','all');
write_nodule = 0;
if exist('numpy_nodules', 'dir')
    rmdir('numpy_nodules','s');
end

CTstr = extractBetween(CTFolderName, 52, 55);
CTstr = string(CTstr);
disp(CTstr);
if ~exist(strcat('CXR/', CTstr), 'dir')
    disp(strcat('CXR/', CTstr));
    mkdir(strcat('CXR/', CTstr));
end
CTnum = 0;

% read in the CT data
[CTarrayOriginal, floatVoxelDims] = dicomHandler(CTFolderName, rotation);


disp("size");
disp(size(CTarrayOriginal));

CTarrayOriginal = CTarrayOriginal - 1000;

CTarrayOriginal = fillmissing(CTarrayOriginal, 'linear');

CTarrayOriginal((0.1707) * (CTarrayOriginal / 1000 + 1) > 0.5) = 0;

% make XRay from empty CT so we only need to update chunks later on
[z, ~, y] = size(CTarrayOriginal);
leftTop = [1, 1];
rightBottom = [z, y];
scaleFactor = double(floatVoxelDims(1)/floatVoxelDims(3));

[projectionOriginal] = XRayMaker(CTarrayOriginal, zeros(2), leftTop, rightBottom, 1, floatVoxelDims(2));
minimum = min(min(projectionOriginal));
SaveXRay(projectionOriginal - minimum, CTstr, rotation);

file = fopen(specificationsFileName);
line = fgetl(file);
numPositions = -1;
while ischar(line)
    numPositions = numPositions + 1;
    line = fgetl(file);
end
fclose(file);

nodulePositions = NaN(numPositions,3);
noduleDimensions = NaN(numPositions,3);
noduleHUs = NaN(numPositions,1);
noduleSizes = NaN(numPositions,1);

file = fopen(specificationsFileName);
fgetl(file);
line = fgetl(file);
xraynum = 0;
while ischar(line)
    % read in data from file
    %disp(line);
    line = str2num(strrep(line,',',' ')); %#ok<ST2NM>
    
    xraynum = xraynum + 1;
    
    % position of nodule
    position_x = line(1);
    position_y = line(2);
    position_z = line(3);
    nodulePositions(xraynum,:) = [position_x,position_y,position_z];
    
    % hounsfield units of nodule in range [80,150] HU
    HU = randi([80,150]);
    noduleHUs(xraynum) = HU;
    
    % size of nodule in range (2,3) cm
    size_nodule = (3-2)*rand + 2;
    noduleSizes(xraynum) = size_nodule;
    
    % create the nodule(s)
    commandStr = strcat('python random_shape_generator.py -d', {' '}, num2str(floatVoxelDims(1)), {' -s '}, num2str(size_nodule));
    disp(commandStr{1});
    [status, commandOut] = system(commandStr{1});
    if status ~= 0
        disp('Error');
        disp(status);
        disp(commandOut);
    else
        disp('Nodule success');
    end
    
    CTarray = CTarrayOriginal;

    nodule = readNPY(strcat('./numpy_nodules/nodule_', int2str(xraynum), '.npy'));
    nodule = imresize3(nodule, [size(nodule, 1) size(nodule, 2)*scaleFactor size(nodule, 3)*scaleFactor], 'nearest');
    
    % dimensions of nodule
    d1 = size(nodule, 1);
    d2 = size(nodule, 2);
    d3 = size(nodule, 3);
    noduleDimensions(xraynum,:) = [d1,d2,d3];
    
    % write nodule to text file
    if write_nodule
        nodule_file = fopen(strcat('textNodules/nodule_',int2str(xraynum),'.txt'),'w');
        for i=1:d1
            for j=1:d2
                for k=1:d3
                        fprintf(nodule_file, int2str(nodule(i,j,k)));
                        fprintf(nodule_file, ',');
                end
            end
            fprintf(nodule_file, '\n');
        end
        fclose(nodule_file);
    end
    % insert nodule to CT data
    for i=1:size(nodule, 1)
        for j=1:size(nodule, 2)
            for k=1:size(nodule, 3)
                if nodule(i, j, k) ~= 0
                    CTarray(position_x - floor(d1/2) + i, ...
                        position_y - floor(d2/2) + j, ...
                        position_z - floor(d3/2) + k) = HU; % + 1000;
                end
            end
        end
    end

    % modify the XRay around the nodule and save it
    leftTop = [position_x - floor(d1/2), position_z - floor(d3/2)];
    rightBottom = [position_x + ceil(d1/2), position_z + ceil(d3/2)];
    
    [projection] = XRayMaker(CTarray, projectionOriginal, leftTop, rightBottom, 0, floatVoxelDims(2));
    disp("rot");
    disp(rotation);
    SaveXRay(projection - minimum, xraynum, CTstr, nodulePositions(xraynum,:), noduleDimensions(xraynum,:), noduleSizes(xraynum), noduleHUs(xraynum), rotation);
    line = fgetl(file);
    %end
    
end
fclose(file);

specs_file = fopen(strcat('nodule_specs_',int2str(CTnum),'.txt'),'w');
fprintf(specs_file,'xraynumber,positions(3),size(3),size(cm),HU\n');
fprintf(specs_file,'0,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan\n');
formatSpec = '%d,%d,%d,%d,%d,%d,%d,%.2f,%d\n';
xraynums = linspace(1,numPositions,numPositions);
fprintf(specs_file,formatSpec,transpose([transpose(xraynums),nodulePositions,noduleDimensions,noduleSizes,noduleHUs]));
fclose(specs_file);

end

function [projection] = XRayMaker(CTarray, projection, leftTop, rightBottom, full, voxDim)

%make the XRay image
%convert Hounsfield units to attenuations
CTarrayChunk = CTarray(leftTop(1):rightBottom(1),:,leftTop(2):rightBottom(2));
CTarrayChunk = CTarrayChunk / 1000;
CTarrayChunk = 0.1707 * (CTarrayChunk + 1);

%Make the projection using xrays from a source infinitely far away
%Uses the beer-lambert law
%Because the xrays are infinitely far away, applying the beer lambert
%law here becomes a simple vectorization calculation
CTarrayChunk = -1 * CTarrayChunk * double(voxDim/100);
% point of view
CTarrayChunk = sum(CTarrayChunk, 2);
CTarrayChunk = squeeze(CTarrayChunk);
projectionChunk = CTarrayChunk;

projectionChunk = exp(projectionChunk);

if full % full X-Ray
    projection = projectionChunk;
else % update chunk of X-Ray
    projection(leftTop(1):rightBottom(1), leftTop(2):rightBottom(2)) = projectionChunk;
end

end

function SaveXRay(projection, CTstr, rotation)
%image processing (colour inverting, scaling, flipping)

im = imshow (projection * (1000), [0, 255]);
im = get(im, 'CData');
im = 255 - im;
im = flip(im, 1);
im = mat2gray(im);
im = imresize(im, [2048, 2048]);

% gamma correction with gamma=2.5 and regular histogram equalization
% im = imadjust(im, [0 1],[0 1], 2.5);
% im = histeq(im, 2048);

% gamma correction with gamma=2.0 and contrast-limited adaptive histogram equalization
im = imadjust(im,[0 1], [0 1], 1);
im = adapthisteq(im);
im = rot90(im, 2);
%THESE LINES CHANGED

imshow(im, [0, 1], 'Border', 'tight');
set(gcf, 'Units', 'pixels', 'Position', [0 0 2048/2 2048/2]);
set(gcf, 'PaperPositionMode', 'auto');
img = getframe(gcf);

fileName = strcat('CXR/', char(CTstr), '/', char(int2str(rotation)), '.png');

disp("ctstr");
disp(class(CTstr));


imwrite(img.cdata, fileName);
end