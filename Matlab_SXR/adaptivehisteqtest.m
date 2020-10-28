function adaptivehisteqtest(path)

A = imread(char(path));
disp(size(A));
A = rgb2gray(A);
disp(size(A));
A = adapthisteq(A);
fileName = "adaptivehist.png";
%imwrite(A.cdata, fileName);

imshow(A, [0, 1], 'Border', 'tight');
set(gcf, 'Units', 'pixels', 'Position', [0 0 2048/2 2048/2]);
set(gcf, 'PaperPositionMode', 'auto');
A = getframe(gcf);

end