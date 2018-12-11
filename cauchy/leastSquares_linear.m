function [error] = leastSquares_linear(datax, datay, slope, intercept)
error = sum((intercept.*datax.^(slope)-datay).^2);
end