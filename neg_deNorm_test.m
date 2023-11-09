function[orig_x] = neg_deNorm_test(x, orig_min, orig_max)
    orig_x = mod((x + pi/4) ./ (3*pi/2) .* (orig_max - orig_min) + orig_min - orig_min, orig_max - orig_min) + orig_min;
end