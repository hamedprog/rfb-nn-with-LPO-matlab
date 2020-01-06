function z = fitness(x)
global nfe;
global k;
global data_test;
m=size(data_test,1);
n=10;
w = x(1:k);
variances = x(k+1:2*k);
centroids = vec2mat(x((2*k)+1:end),n-1);

for j=1:m
    input=data_test(j,1:n-1);
    target=data_test(j,n);
    
    net =[];
    for p=1:k
        net = [net norm(input-centroids(p,:))];
    end
    
    o =[];
    for p=1:k
        o = [o gauss(net(p),variances(p))];
    end
    output = o*w';
%     error = target-output;
    error(j)=target-output;
end

nfe=nfe+1;
z=sqrt(mse(error));

end