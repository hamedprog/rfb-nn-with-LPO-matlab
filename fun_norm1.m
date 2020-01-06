function z = fun_norm1(data)

n=size(data,2);
for j=1:n-1
%     max_data=max(data(:,j));
%     min_data=min(data(:,j));
%     for p=1:numel(data(:,j))
%         data(p,j)=(data(p,j)-min_data)/(max_data-min_data);
%     end
end
z=data;
end

