function bestlion1 = StrongerBestLion(bestlion1,long_stagnation_times,nvar,L0)

% ebest = 0;
best1 = Inf;
for i=1:nvar
    e = zeros(1,nvar);
    for j=1:10:400
        epsilon = (L0*(201-j))/(200*(10^(2*(long_stagnation_times+2))));
        e(i) = epsilon;
        n1 = fitness(bestlion1.pos+e);
        if n1<best1
            best1 = n1;
            ebest = e;
        end
    end 
end

bestlion1.pos = bestlion1.pos+ebest;
bestlion1.cost = fitness(bestlion1.pos);

end

