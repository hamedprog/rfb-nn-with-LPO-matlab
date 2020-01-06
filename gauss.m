function z = gauss(net,sigma)

standardDeviation=sqrt(sigma);
z=exp(((net/standardDeviation)^2)*(-1/2));

end

