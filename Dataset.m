function Dataset(CTRootName)

dirlist = dir(CTRootName);
maxSN = size(dirlist);


for i = 1:maxSN
    
    if i ~= 1 && i ~= 2
        x = dirlist(i).name;
        pathName = strcat(CTRootName, '\', x);
        %disp(pathName);
        subdirlist = dir(pathName);
        maxSDN = size(subdirlist);

        for j = 1:maxSDN

            y = subdirlist(j).name;
            subPathName= strcat(pathName, '\', y);    
            disp(subPathName);

            if j == 3 || j == 4

                subsubDir = dir(subPathName);
                subsubPathName = subsubDir(3).name;

                finalDir = dir(strcat(subPathName, "\", subsubPathName));
                slices_temp_size = size(finalDir);

                dcm_slices_num = (slices_temp_size(1));

                if dcm_slices_num > 170

                    xrayHandler(strcat(subPathName, "\", subsubPathName, "\"));

                end
            end
        end
    end
    
end

end