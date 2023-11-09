function [z] = deNorm(y,x)
    z = (y - pi/4) .* 2/3*pi .*(max(x(:)-min(x(:))) + min(x(:)));
    z = z.*.1;
end