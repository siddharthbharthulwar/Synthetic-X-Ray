function [refined_dirlist] = listing(path)

org_dirs = dir(path);
refined_dirlist = org_dirs(3:end);

end
