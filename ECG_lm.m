clc;
clear all;
close all;

data_temp=xlsread('ECG.xlsx');
m_data=5;
data=fun_norm(data_temp,m_data);
num_data=size(data,1);
rate_train=0.75;
n_data=size(data,2);
x=n_data-1;
n=15;
m=30;
k=25;
p=10;
t=10;
l=1;

num_train=round(num_data*rate_train);
num_test=num_data-num_train;
data_train=data(1:num_train,:);
data_test=data(num_train+1:end,:);

epoch_max=200;
a=-0.5;
b=0.5;

w1=unifrnd(a,b,[n,x]);
net1=zeros(1,n);
o1=zeros(1,n);

w2=unifrnd(a,b,[m,n]);
net2=zeros(1,m);
o2=zeros(1,m);

w3=unifrnd(a,b,[k,m]);
net3=zeros(1,k);
o3=zeros(1,k);

w4=unifrnd(a,b,[p,k]);
net4=zeros(1,p);
o4=zeros(1,p);

w5=unifrnd(a,b,[t,p]);
net5=zeros(1,t);
o5=zeros(1,t);

w6=unifrnd(a,b,[l,t]);
net6=zeros(1,l);
o6=zeros(1,l);

error_train=zeros(num_train,1);
error_test=zeros(num_test,1);

output_train=zeros(num_train,1);
output_test=zeros(num_test,1);

mse_train=zeros(epoch_max,1);
mse_test=zeros(epoch_max,1);
w1_t=reshape(w1,1,n*x);
w2_t=reshape(w2,1,m*n);
w3_t=reshape(w3,1,k*m);
w4_t=reshape(w4,1,p*k);
w5_t=reshape(w5,1,t*p);
w6_t=reshape(w6,1,l*t);

w_all=[w1_t w2_t w3_t w4_t w5_t w6_t];
Jacobian=zeros(num_train,n*x+m*n+k*m+p*k+t*p+l*t);
I=eye(n*x+m*n+k*m+p*k+t*p+l*t);


for i=1:epoch_max
    for j=1:num_train
        input=data_train(j,1:n_data-1);
        target=data_train(j,n_data);
        
        net1=input*w1';
        o1=logsig(net1);
        
        net2=o1*w2';
        o2=logsig(net2);
        
        net3=o2*w3';
        o3=logsig(net3);
        
        net4=o3*w4';
        o4=logsig(net4);
        
        net5=o4*w5';
        o5=logsig(net5);
        
        net6=o5*w6';
        o6=net6;
        
        error=target-o6;
        
        c5=o5.*(1-o5);
        
        A=diag(c5);
        c4=o4.*(1-o4);
  
        B=diag(c4);
        c3=o3.*(1-o3);
        C=diag(c3);
        c2=o2.*(1-o2);
        D=diag(c2);
        c1=o1.*(1-o1);
        E=diag(c1);
        

%         w1=w1-eta*error*-1*1*(w6*A*w5*B*w4*C*w3*D*w2*E)'*input;
%         w2=w2-eta*error*-1*1*(w6*A*w5*B*w4*C*w3*D)'*o1;
%         w3=w3-eta*error*-1*1*(w6*A*w5*B*w4*C)'*o2;
%         w4=w4-eta*error*-1*1*(w6*A*w5*B)'*o3;
%         w5=w5-eta*error*-1*1*(w6*A)'*o4;
%         w6=w6-eta*error*-1*1*o5;
%         
        gw1=-1*1*(w6*A*w5*B*w4*C*w3*D*w2*E)'*input;
        gw2=-1*1*(w6*A*w5*B*w4*C*w3*D)'*o1;
        gw3=-1*1*(w6*A*w5*B*w4*C)'*o2;
        gw4=-1*1*(w6*A*w5*B)'*o3;
        gw5=-1*1*(w6*A)'*o4;
        gw6=-1*1*o5;
        
        Jacobian(j,:)=[reshape(gw1,1,n*x) reshape(gw2,1,m*n) reshape(gw3,1,k*m) reshape(gw4,1,p*k) reshape(gw5,1,t*p) reshape(gw6,1,l*t)];
        
    end
    mu=error_train'*error_train;%ok
    w_all=w_all-(inv(Jacobian'*Jacobian+mu*I)*Jacobian'*error_train)';
    
    w1=reshape(w_all(1:x*n),n,x);
    w2=reshape(w_all(x*n+1:x*n+m*n),m,n);
    w3=reshape(w_all(x*n+m*n+1:x*n+m*n+k*m),k,m);
    w4=reshape(w_all(x*n+m*n+k*m+1:x*n+m*n+k*m+p*k),p,k);
    w5=reshape(w_all(x*n+m*n+k*m+p*k+1:x*n+m*n+k*m+p*k+t*p),t,p);
    w6=reshape(w_all(x*n+m*n+k*m+p*k+t*p+1:end),l,t);
    for j=1:num_train
        input=data_train(j,1:n_data-1);
        target=data_train(j,n_data);
        
        net1=input*w1';
        o1=logsig(net1);
        
        net2=o1*w2';
        o2=logsig(net2);
        
        net3=o2*w3';
        o3=logsig(net3);
        
        net4=o3*w4';
        o4=logsig(net4);
        
        net5=o4*w5';
        o5=logsig(net5);
        
        net6=o5*w6';
        o6=net6;
        
        error=target-o6;
        
        
        error_train(j)=error;
        
        output_train(j)=o6;
        
        
    end
    
    for j=1:num_test
        input=data_test(j,1:n_data-1);
        target=data_test(j,n_data);
        net1=input*w1';
        o1=logsig(net1);
        
        net2=o1*w2';
        o2=logsig(net2);
        
        net3=o2*w3';
        o3=logsig(net3);
        
        net4=o3*w4';
        o4=logsig(net4);
        
        net5=o4*w5';
        o5=logsig(net5);
        
        net6=o5*w6';
        o6=net6;
        
        error=target-o6;
        
        
        error_test(j)=error;
        
        output_test(j)=o6;
        
        
    end
    
    
    mse_train(i)=mse(error_train);
    mse_test(i)=mse(error_test);
    
    
    figure(1);
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

end

figure(2);
plotregression(data_train(:,n_data),output_train,'Train',data_test(:,n_data),output_test,'Test')

