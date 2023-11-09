function [orig_s] = neg_deNorm(new_s, orig_min, orig_max)
    orig_s = (new_s + pi/4) ./ (3*pi/2) .* (orig_max - orig_min) + orig_min;
    orig_s = mod(orig_s - orig_min, orig_max - orig_min) + orig_min;
end