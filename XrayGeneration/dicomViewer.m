function dicomViewer(array, slices, rotations)

for index =1:slices
    
    slice = rot90(squeeze(array(index,:,:)), rotations);
    imshow(slice)
    pause(1);

end