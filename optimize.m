function [bestlion1,bestlion2] = optimize3(pride,bestlion1,bestlion2,L0,k)
n=numel(pride);
global nfe;

cubsb1 = pride(1:(n/2));
cubsb2 = pride((n/2)+1:end);
[min_val1,idx1]=min([cubsb1.cost]);
[min_val2,idx2]=min([cubsb2.cost]);

d1 = (cubsb1(idx1).pos-bestlion1.pos)/norm(cubsb1(idx1).pos-bestlion1.pos);
d1(isnan(d1))=0;
d2 = (cubsb2(idx2).pos-bestlion2.pos)/norm(cubsb2(idx2).pos-bestlion2.pos);
d2(isnan(d2))=0;
best1 = Inf;
best2 = Inf;
for i=1:100:2000
    epsilon = ((1001-i)*norm(L0))/(2000*log(k+2));
    n1 = fitness(bestlion1.pos+epsilon*d1);
    if n1<best1
        best1 = n1;
        epsilonbest1 = epsilon;
    end
    n2 = fitness(bestlion2.pos+epsilon*d2);
    if n2<best2
        best2 = n2;
        epsilonbest2 = epsilon;
    end
end
bestlion1.pos = bestlion1.pos+epsilonbest1*d1;
bestlion2.pos = bestlion2.pos+epsilonbest2*d2;

bestlion1.cost = fitness(bestlion1.pos);
bestlion2.cost = fitness(bestlion2.pos);
end