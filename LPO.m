clc;
close all;
clear all;
global nfe;
nfe=0;
data=xlsread('temperature.xlsx');
data = fun_norm(data,5);
global data_test
num_data=size(data,1);
rate_train=0.75;
n_data=size(data,2);
num_train=round(num_data*rate_train);
num_test=num_data-num_train;
data_train=data(1:num_train,:);
data_test=data(num_train+1:end,:);
eta=0.01;
epoch_max=1000;

mc0=2;
nmember=25;
stagnation_times = 0;%stagnation times
long_stagnation_times = 0;%long stagnation times
ths = 2; %stagnation threshold
thls = 4; %long stagnation threshold
k1 = 1000;
k2 = 5;

num_changespace=0;

nvar=31;
L0=2;
max_iter=100;
best_cost=zeros(max_iter,1);
nfe_best_cost=zeros(max_iter,1);

member.pos=[];
member.cost=[];
pride=repmat(member,nmember,1);

% % generete initial pride
for k=1:nmember
    pride(k).pos=unifrnd(-1,1,[1 nvar]);
    pride(k).cost=fitness(pride(k).pos);
end

bestlion1 = pride(1);
bestlion2 = pride(2);
best_cost_prv=Inf;

for k=1:max_iter

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
    [bestlion1,bestlion2] = optimize(pride,bestlion1,bestlion2,L0,k);
    
    best_cost(k)=bestlion1.cost;
   
    % %     Select the best members to the next generation using (4)
    [a,b]=sort([pride.cost]);
    pride = pride(b(1:nmember)); %selection
  
    stagnation_times=stagnation_times+1
    
    % %     change the search space using formula (5)
    if(stagnation_times>ths)
        pride = changeSpace(pride,nvar,L0,k1,k2,stagnation_times,bestlion1);
        num_changespace=num_changespace+1
        if(best_cost_prv-best_cost(k)>0.0001)
            stagnation_times=0
        end
    end
    
    % %     Optimize the best member by one-dimension search in each dimension using(8)
    if(stagnation_times>thls)
        long_stagnation_times = long_stagnation_times+1
        k=k
        bestlion1 = StrongerBestLion(bestlion1,long_stagnation_times,nvar,L0);
    end

    if(long_stagnation_times>4)
        break
    end
    
    best_cost_prv = best_cost(k);
    
    nfe_best_cost(k)=nfe;
    figure(1);
    semilogy(nfe_best_cost(1:k),best_cost(1:k),'-r');
    title("number of function evaluation and correspound best cost")
    xlabel('nfe');
    ylabel('Best Cost');
    hold off;
    if(nfe > 10000)
        break
    end
end
saveas(figure(1),"effect of LPO.jpg")

n1=4;
n2=4;
n3=3;
n4=1;

w1=reshape(bestlion1.pos(1:n1*n2),n2,n1);
net1=zeros(1,n2);
o1=zeros(1,n2);

w2=reshape(bestlion1.pos(n1*n2+1:n1*n2+n3*n2),n3,n2);

net2=zeros(1,n3);
o2=zeros(1,n3);

w3=reshape(bestlion1.pos(n1*n2+n3*n2+1:end),n4,n3);

net3=zeros(1,n4);
o3=zeros(1,n4);

error_train=zeros(num_train,1);
error_test=zeros(num_test,1);

output_train=zeros(num_train,1);
output_test=zeros(num_test,1);

mse_train=zeros(epoch_max,1);
mse_test=zeros(epoch_max,1);

for i=1:epoch_max
    for j=1:num_train
        input=data_train(j,1:n_data-1);
        target=data_train(j,n_data);
        
        net1=input*w1';
        o1=logsig(net1);
        
        net2=o1*w2';
        o2=logsig(net2);
        
        net3=o2*w3';
        o3=net3;
        
        error=target-o3;
        
        c2=o2.*(1-o2);
        A=diag(c2);
        
        c1=o1.*(1-o1);
        B=diag(c1);
        
        w1=w1-eta*error*-1*1*(w3*A*w2*B)'*input;
        w2=w2-eta*error*-1*1*(w3*A)'*o1;
        w3=w3-eta*error*-1*1*o2;
        
    end
    
    for j=1:num_train
        input=data_train(j,1:n_data-1);
        target=data_train(j,n_data);
        
        net1=input*w1';
        o1=logsig(net1);
        
        net2=o1*w2';
        o2=logsig(net2);
        
        net3=o2*w3';
        o3=net3;
        
        error=target-o3;
        
        error_train(j)=error;
        
        output_train(j)=o3;
    end
    for j=1:num_test
        input=data_test(j,1:n_data-1);
        target=data_test(j,n_data);
        net1=input*w1';
        o1=logsig(net1);
        
        net2=o1*w2';
        o2=logsig(net2);
        
        net3=o2*w3';
        o3=net3;
  
        error=target-o3;
        
        error_test(j)=error;
        
        output_test(j)=o3;
    end  
    mse_train(i)=mse(error_train);
    mse_test(i)=mse(error_test);
    
    figure(2);
    subplot(2,2,1),semilogy(mse_train(1:i));
    %     hold on;
    subplot(2,2,2),semilogy(mse_test(1:i),'-r');
    %     hold on;
    subplot(2,2,3),plot(data_train(:,n_data));
    hold on;
    subplot(2,2,3),plot(output_train,'-r');
    hold off;
    
    subplot(2,2,4),plot(data_test(:,n_data));
    hold on;
    subplot(2,2,4),plot(output_test,'-r');
    hold off;
    if(i==1)
        saveas(figure(2),"initial state of neural network with weights from LPO.jpg")
    end  
end
saveas(figure(2),"model after gradient descent.jpg")

figure(3);
plotregression(data_train(:,n_data),output_train,'Train',data_test(:,n_data),output_test,'Test')
saveas(figure(3),"regression.jpg")
save("LPO gradient descent")   