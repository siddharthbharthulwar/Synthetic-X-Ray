function Dataset(CTRootName)

dirlist = dir(CTRootName);
maxSN = size(dirlist);

for i = 1:maxSN
    
    x = dirlist(i).name;
    pathName = strcat(CTRootName, '\', x);
    disp(pathName);
    subdirlist = dir(pathName);
    maxSDN = size(subdirlist);
    
    for j = 1:maxSDN
        
        y = dirlist(j).name;
        subsubdirlist = strcat(pathName, '\', y);
        maxSSDN = size(subsubdirlist);
        
        disp(maxSSDN);
        
        
    end
    
end

end
