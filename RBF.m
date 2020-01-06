clc;
clear all;
close all;

data_temp=xlsread('cancer.xlsx');
global data_test
global k
data=fun_norm1(data_temp);
num_data=size(data,1);
rate_train=0.75;
n=size(data,2);
num_class=2;
data_train=[];
data_test=[];
num_train_all=0;
num_test_all=0;
epoch_max=200;

num_class_zero=find(data(:,n)==0);
data = [data;data(num_class_zero(1:217),:)];

%% create test and train data
for i=0:num_class
    num_class_temp=find(data(:,n)==i);
    num_class_each=numel(num_class_temp);
    num_train=round(num_class_each*rate_train);
    num_test=num_class_each-num_train;
    
    data_train=[data_train; data(num_class_temp(1:num_train),:) ];
    data_test=[data_test; data(num_class_temp(1+num_train:end),:) ];
    num_train_all=num_train_all+num_train;
    num_test_all=num_test_all+num_test;
end

%% calculate centroids
k=12;
max_iteration_kmeans = 100;

centroids = data(1:k,1:n-1);

for p=1:max_iteration_kmeans
    for i=1:k
        class{i}=[];
    end
    
    for j=1:num_train_all
        distances=[];
        for i=1:k
            distances=[distances norm(data_train(j,1:n-1)-centroids(i,1:n-1))];
        end
        [a,b] = min(distances);
        class{b} = [class{b};data_train(j,1:n-1)];
    end
    for j=1:k
        centroids(j,:) = mean(class{j});
    end
end

%% compute variance of each class
variances = zeros(1,k);
for j=1:k
    n_class = size(class{j},1);
    sum_class=0;
    for p=1:n_class
        sum_class=sum_class+((class{j}(p,:)-centroids(j,:))*(class{j}(p,:)-centroids(j,:))');
    end
    variances(j)=sum_class/n_class;
end

%% optimize parameteres with LPO

global nfe;
nfe=0;
mc0=2;
nmember=50;
stagnation_times = 0;%stagnation times
long_stagnation_times = 0;%long stagnation times
ths = 2; %stagnation threshold
thls = 4; %long stagnation threshold
k1 = 1000;
k2 = 5;
num_changespace=0;

nvar=11*k;
L0=2;
max_iter_LPO=30;
best_cost=zeros(max_iter_LPO,1);
nfe_best_cost=zeros(max_iter_LPO,1);

member.pos=[];
member.cost=[];
pride=repmat(member,nmember,1);

for i=1:nmember
    pride(i).pos=unifrnd(-1,1,[1 nvar]);
    pride(i).cost=fitness(pride(i).pos);
end

centroids_all= reshape(centroids',1,[]);
w =unifrnd(-1,1,[1,k]);
bestlion1.pos = [w variances centroids_all];
bestlion1.cost = fitness(bestlion1.pos);
bestlion2 = pride(1);
best_cost_prv=Inf;

%% LPO main loop
for genaration=1:max_iter_LPO

    % %     select two best member of current generation using formula (1)
    [a,b]=sort([pride.cost]);
    best1 = pride(b(1));
    best2 = pride(b(2));
    
    % %     scaling of searching space
    if(best1.cost<bestlion1.cost)
        bestlion1 = best1;
        bestlion2 = best2;
    else
        bestlion2=best1;
    end
    
    % %     Replace worst member of this generation with the saved best member
    pride(b(nmember))=bestlion1;
        
    % %     For (each member in the pride), reproduce four children for each pair using (2)
    pride = crossover(bestlion1,bestlion2,pride,mc0);
    
    % %     Select the potentially best evolution directions using (6)
    % %     Search in the potentially best evolution directions for better member using (7)
    [bestlion1,bestlion2] = optimize(pride,bestlion1,bestlion2,L0,genaration);
    
    best_cost(genaration)=bestlion1.cost;
   
    % %     Select the best members to the next generation using (4)
    [a,b]=sort([pride.cost]);
    pride = pride(b(1:nmember)); %selection
  
    stagnation_times=stagnation_times+1
    
    % %     change the search space using formula (5)
    if(stagnation_times>ths)
        pride = changeSpace(pride,nvar,L0,k1,k2,stagnation_times,bestlion1);
        num_changespace=num_changespace+1
        if(best_cost_prv-best_cost(genaration)>0.0001)
            stagnation_times=0
        end
    end
    
    % %     Optimize the best member by one-dimension search in each dimension using(8)
    if(stagnation_times>thls)
        long_stagnation_times = long_stagnation_times+1
        genaration=genaration
        bestlion1 = StrongerBestLion(bestlion1,long_stagnation_times,nvar,L0);
    end

    if(long_stagnation_times>4)
        break
    end
    
    best_cost_prv = best_cost(genaration);
    
    nfe_best_cost(genaration)=nfe;
    figure(1);
    semilogy(nfe_best_cost(1:genaration),best_cost(1:genaration),'-r');
    title("number of function evaluation and correspound best cost")
    xlabel('nfe');
    ylabel('Best Cost');
    hold off;
end
saveas(figure(1),"effect of LPO.jpg")

%% train nn with levenberg marquardt

w = bestlion1.pos(1:k);
variances = bestlion1.pos(k+1:2*k);
centroids = vec2mat(bestlion1.pos((2*k)+1:end),n-1);

error_train=zeros(num_train_all,1);
error_test=zeros(num_test_all,1);

output_train=zeros(num_train_all,1);
output_test=zeros(num_test_all,1);

mse_train=zeros(epoch_max,1);
mse_test=zeros(epoch_max,1);

Jacobian=zeros(num_train_all,k);
I=eye(k);

data_train1=data_train;
for i=1:epoch_max
    for j=1:num_train_all
%         data_train=data_train(randperm(num_train_all),:);
        input=data_train(j,1:n-1);
        target=data_train(j,n);
        
        net =[];
        for p=1:k
            net = [net norm(input-centroids(p,:))];
        end
        
        o =[];
        for p=1:k
            o = [o gauss(net(p),variances(p))];
        end
        output = o*w';
        error = target-output;
        
        Jacobian(j,:)=[-o];
    end
    mu=error_train'*error_train;
%     mu=10;
    w = w-(inv(Jacobian'*Jacobian+mu*I)*Jacobian'*error_train)';
    
    for j=1:num_train_all
        input=data_train1(j,1:n-1);
        target=data_train1(j,n);
        
        net =[];
        for p=1:k
            net = [net norm(input-centroids(p,:))];
        end
        
        o =[];
        for p=1:k
            o = [o gauss(net(p),variances(p))];
        end
        output = o*w';
        error=target-output;
        
        error_train(j)=error;
        
        output_train(j)=round(output);
        
    end
    
    for j=1:num_test_all
        
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
        
        error=target-output;
        error_test(j)=error;
        
        output_test(j)=round(output);
    end
    
    mse_train(i)=mse(error_train);
    mse_test(i)=mse(error_test);
    
    figure(2);
    subplot(2,2,1),semilogy(mse_train(1:i));
    %     hold on;
    subplot(2,2,2),semilogy(mse_test(1:i),'-r');
    %     hold on;
    subplot(2,2,3),plot(data_train1(:,n),'-*');
    hold on;
    subplot(2,2,3),plot(output_train,'-or');
    hold off;
    
    subplot(2,2,4),plot(data_test(:,n),'-*');
    hold on;
    subplot(2,2,4),plot(output_test,'-or');
    hold off;
    if(i==1)
        saveas(figure(2),"initial state of neural network with weights from LPO.jpg")
    end 
    
end
saveas(figure(2),"model after levenberg.jpg")

figure(3);
plotregression(data_train(:,n),output_train,'Train',data_test(:,n),output_test,'Test')
saveas(figure(3),"regression.jpg")
figure(4);

Target_con=zeros(num_train_all,2);
Output_con=zeros(num_train_all,2);
for j=1:num_train_all
    switch data_train1(j,n)
        case 0
            Target_con(j,:)=[1 0];
        case 1
            Target_con(j,:)=[0 1];
            
            
    end
    
    switch output_train(j)
        case 0
            Output_con(j,:)=[1 0];
        case 1
            Output_con(j,:)=[0 1];
            
    end
end

plotconfusion(Target_con',Output_con')
saveas(figure(4),"plotconfusion.jpg")
save("RBF")