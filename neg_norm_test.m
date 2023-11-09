function[x] = neg_norm_test(x)
    x = mod(((x - min(x(:))) / (max(x(:)) - min(x(:)))) .* (3*pi/2) - (pi/4) + pi, 2*pi) - pi;
end