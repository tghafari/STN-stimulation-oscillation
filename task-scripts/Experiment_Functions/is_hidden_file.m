function isHidden = is_hidden_file(file)
% Helper function to check if a file is hidden on Windows

    if ispc  % Check if the platform is Windows
        isHidden = bitand(file.attrib, 2) > 0;  % Hidden files have the "hidden" attribute (bit 2 set)
    else
        isHidden = false;  % On other platforms, rely on leading dot convention
    end

end