function [returnedMatrix] = rotateMatrix(inputMatrix, size, rotations)

%output matrix will be same size as input matrix

tempMatrix = zeros(size(inputMatrix));

for index = 1:size
    
    slice = squeeze(inputMatrix(index,:,:));
    slice = rot90(slice, rotations);
    
    sz = [512 512];
    row = [];
    col = [];
    
    ind = sub2ind(
    
    tempMatrix(index, :, :) = slice;

returnedMatrix = tempMatrix;

end
