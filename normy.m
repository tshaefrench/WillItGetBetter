function [z] = normy(y,x)
    z = (y - floorDiv(pi, 4)) .* floorDiv(2,3)*pi .*(max(x(:)-min(x(:))) + min(x(:)));
end