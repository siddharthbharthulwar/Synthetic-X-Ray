%turns the dicom files into a 3D array that represents the CT data
function [CTarray, voxelDims] = coronalHandler(folderName)

dirlist = dir(folderName);

maxSN = length(dirlist);
info = dicominfo(strcat(folderName,dirlist(10).name));

empty = -9999;

dArr = zeros(maxSN,info.Width,info.Height)+empty;
slicesPos = zeros(maxSN,1)+empty;

place = 1;


for i = 1:maxSN
    
    if i > 2
        x = dirlist(i).name;
        disp(x);
        disp(i);
    end

end

for i = 1:maxSN
    
    if i > 2
        x=dirlist(i).name;
        disp(strcat(folderName, x));
        info = dicominfo(strcat(folderName,x));
        slicesPos(place) = info.SliceLocation;
        place = place + 1;
    end
end

slicesPos = sort(unique(slicesPos));


for i = 1:maxSN
    
    if i > 2
        x=dirlist(i).name;
        info = dicominfo(strcat(folderName,x));
        dArr(slicesPos == info.SliceLocation,:,:) = (uint16(dicomread(strcat(folderName,x)));
    end
    
end

dArr(any(any(dArr == -9999,3),2),:,:) = [];

%myIm = squeeze(dArr(10,:,:)) - 1000;
%imshow(dArr, [0 256]);
CTarray = dArr;
disp(size(CTarray));
voxelDims = [info.SliceThickness, info.PixelSpacing(1), info.PixelSpacing(2)];
end