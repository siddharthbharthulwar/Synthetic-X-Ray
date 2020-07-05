function Dataset(CTRootName)

dirlist = dir(CTRootName);
maxSN = size(dirlist);

for i = 1:maxSN
    
    x = dirlist(i).name;
    pathName = strcat(CTRootName, '\', x);
    subdirlist = dir(pathName);
    maxSDN = size(subdirlist);
    
    for j = 1:maxSDN
        
        y = subdirlist(j).name;
        subPathName= strcat(pathName, '\', y);    
        
        if j == 3 || j == 4
            
            subsubDir = dir(subPathName);
            subsubPathName = subsubDir(3).name;
            
            finalDir = dir(strcat(subPathName, "\", subsubPathName));
            slices_temp_size = size(finalDir);
            
            dcm_slices_num = (slices_temp_size(1));
            if dcm_slices_num > 170
                
                disp(strcat(subPathName, "\", subsubPathName, "\"));

            end
        end
    end
    
end

end
