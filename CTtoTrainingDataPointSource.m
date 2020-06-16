% Copyright (C) 2018-2019
% Abhishek Moturu, moturuab@cs.toronto.edu
% Alex Chang, le.chang@mail.utoronto.ca

% creates nodules, inserts them into CT data, and makes POINT-SOURCE X-Rays

% must have the following files in the same folder:

% random_shape_generator.py

% readNPY.m
% readNPYheader.m
% dicomHandler.m

% positions_0.txt (etc.)
% chestCT0 folder (etc.)

% Constants.h
% Chunk.hpp
% Coordinate.hpp
% methods.hpp
% SimulatedRay.hpp
% ProjectionPlane.hpp
% Voxel.hpp
% NoduleSpecs.hpp
% Pixel.hpp

% Coordinate.cpp
% methods.cpp
% SimulatedRay.cpp
% Voxel.cpp
% main.cpp

% Makefile

% to run, type into the console: CTtoTrainingDataPointSource(...);
% CTFolderName is the folder containing the CT slices files
% specificationsFileName is the file that contains nodule positions

% currently only making 7 X-rays from various point sources for testing

% CTtoTrainingDataPointSource('chestCT0/I/NVFRWCBT/5O4VNQBN/', 'positions_0.txt');



function CTtoTrainingDataPointSource(CTFolderName, specificationsFileName)
warning('off','all');

disp(getenv('PATH'));

write_ct = 1;
write_nodules = 0;
if exist('numpy_nodules', 'dir')
    rmdir('numpy_nodules','s');
end
disp(CTFolderName);
CTnum = str2num(CTFolderName(end)); %#ok<ST2NM>
if exist(strcat('chestXRays',int2str(CTnum)), 'dir')
    rmdir(strcat('chestXRays',int2str(CTnum)),'s');
end
mkdir(strcat('chestXRays',int2str(CTnum)));

% read in the CT data
[CTarrayOriginal, floatVoxelDims] = dicomHandler(CTFolderName);

CTarrayOriginal = CTarrayOriginal - 1000;

samp = zeros(3);


disp("Array Size:");
disp(size(CTarrayOriginal));
disp("samp size:");
disp(size(samp));

CTarrayOriginal(CTarrayOriginal > 500) = NaN;
CTarrayOriginal = fillmissing(CTarrayOriginal, 'linear');

if write_ct
    % write CT to text file
   % dlmwrite(strcat('textCTs/CT_',int2str(CTnum),'.txt'),CTarrayOriginal);
   matrixCTFile = fopen('textCTs/realCT.txt', 'wt');
   fprintf(matrixCTFile, '%d %d %d\n', CTarrayOriginal);
   fclose(matrixCTFile);
end

scaleFactor = double(floatVoxelDims(1)/floatVoxelDims(3));


file = fopen(specificationsFileName);
line = fgetl(file);
numPositions = -1;
while ischar(line)
    numPositions = numPositions + 1;
    line = fgetl(file);
end
fclose(file);

numPositions = 2;

nodulePositions = NaN(numPositions,3);
noduleDimensions = NaN(numPositions,3);
noduleHUs = NaN(numPositions,1);
noduleSizes = NaN(numPositions,1);

file = fopen(specificationsFileName);
fgetl(file);
line = fgetl(file);

xraynum = 0; % only making one X-ray for now
while ischar(line) && ~xraynum
    % read in data from file
    disp(line);
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
    
    nodule = readNPY(strcat('./numpy_nodules/nodule_', int2str(xraynum), '.npy'));
    
    nodule = imresize3(nodule, [size(nodule, 1) size(nodule, 2)*scaleFactor size(nodule, 3)*scaleFactor], 'nearest');
    
    % dimensions of nodule
    d1 = size(nodule, 1);
    d2 = size(nodule, 2);
    d3 = size(nodule, 3);
    noduleDimensions(xraynum,:) = [d1,d2,d3];
    
    % write nodule to text file
    if write_nodules
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


system('make');
cmd = strcat('lungnodulesynthesizer.exe', {' '}, 'textCTs/CT_',int2str(CTnum), '.txt', {' '}, 'nodule_specs_',int2str(CTnum),'.txt', {' '}, num2str(floatVoxelDims(3)), {' '}, num2str(floatVoxelDims(1)));
system(char(cmd));

% make c++ loop through all nodules, several sources, and name things accordingly

files=dir('textXRays');
files=files(~ismember({files.name},{'.','..'}));

for k=1:length(files)
    xraynum = k-1;
    %nodulePosition = nodulePositions(k,:);
    %noduleDimension = noduleDimensions(k,:);
    %noduleSize = noduleSizes(k);
    %noduleHU = noduleHUs(k);
    
    textfilename = files(k).name;
    disp(strcat('textXRays/',textfilename));
    img = importdata(strcat('textXRays/',textfilename));
    imagename = strcat('chestXRays', int2str(CTnum), '/Xray', int2str(xraynum));
    img = 1 - img;
    img = rot90(img, 3);
    img = flip(img, 2);
    
    % gamma correction with gamma=2.5 and regular histogram equalization
    % im = imadjust(im, [0 1],[0 1], 2.5);
    % im = histeq(im, 256);

    % gamma correction with gamma=2.0 and contrast-limited adaptive histogram equalization
    im = imadjust(im,[0 1], [0 1], 2);
    im = adapthisteq(im);
    
    imwrite(img, strcat(imagename, '.png'));
end

end
