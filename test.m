function test

a = randi([0, 1], [10, 4, 4]);

dicomViewer(a, 10, 0);

dicomViewer(rotateMatrix(a, 10, 1));


end

