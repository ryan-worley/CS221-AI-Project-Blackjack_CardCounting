function [error] = leastSquaresCauchey(datax, datay, median, sigmaln)
error = sum((datay - cauchypdf(datax, median, sigmaln)).^2);
end