function Allcubs = crossover(bestlion1,bestlion2,pride,mc0,mc )
n=numel(pride);
mc=mc0*(unifrnd(0,1,[n,1])-0.5);
% cub.pos=[];
% cub.cost=[];
% cubsb1=repmat(cub,2*n,1);
% cubsb2=repmat(cub,2*n,1);
for i=1:n
    cubsb1(2*i-1).pos = bestlion1.pos + mc(i)*(bestlion1.pos-pride(i).pos);
    cubsb1(2*i).pos = bestlion1.pos - mc(i)*(bestlion1.pos-pride(i).pos);
end

for i=1:n
    cubsb2(2*i-1).pos = bestlion2.pos + mc(i)*(bestlion2.pos-pride(i).pos);
    cubsb2(2*i).pos = bestlion2.pos - mc(i)*(bestlion2.pos-pride(i).pos);
end

Allcubs=[cubsb1;cubsb2];
for i=1:4*n
    Allcubs(i).cost=fitness(Allcubs(i).pos);
end
end