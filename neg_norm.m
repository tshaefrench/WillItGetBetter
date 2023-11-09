function [new_x] = neg_norm(x)
      new_x = ( (x - min(x(:)) ) / ( max(x(:)) - min(x(:)) ) ) .* (3*pi/2) - (pi/4);
      new_x = mod(new_x + pi, 2*pi) - pi;
end