function [orig_data] = new_deNorm(new_data, old_data)
    orig_data = (new_data - pi/4) ./ (3*pi/2) .* (max(old_data(:)) - min(old_data(:))) + min(old_data(:));
end