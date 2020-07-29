function DatasetFromTXT()

    fid = fopen('dirs.txt');
    tline = fgetl(fid);
    
    while ischar(tline)
        disp(tline);
        xrayHandler(tline);
        tline = fgetl(fid);
    end
    
    fclose(fid);
    

end

