function pride = changeSpace(pride,nvar,L0,k1,k2,stagnation_times,bestlion1)
n=numel(pride);
L=(L0/(k1*log(stagnation_times+2)-k1+1));
sigma = (L/2)*eye(nvar);
if(mod(stagnation_times,k2)==0)
    for k=1:n
        pride(k).pos=unifrnd(-1,1,[1 nvar]);
        pride(k).cost=fitness(pride(k).pos);
    end
else
    for i=1:n
        pride(i).pos=mvnrnd(bestlion1.pos,sigma);
        pride(i).cost=fitness(pride(i).pos);
    end
end

