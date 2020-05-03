function [reducedData] = fisherdiscriminant(datos,clases)
d=784;
vecclases=unique(clases');
numvecclases = 10;
dimensionreduccion=9;


mediaclase = cell(1,numvecclases);
CovVal = cell(1,numvecclases);
sizeC=zeros(1,numvecclases);

i=1;
%Para cada clase:
for j=1:numvecclases
    claseactual=vecclases(1,j);
    Xc=datos(:,clases==claseactual);
    mediaclase(i) = {mean(Xc,2)};
    CovVal(i) = {cov(Xc')};
    sizeC(i)=size(Xc,2);
    i=i+1;
end

mediaglob = mean(datos,2);
SB = zeros(d,d);
SW = zeros(d,d);

for i = 1:numvecclases
    SB = SB + sizeC(i).*(mediaclase{i}-mediaglob)*(mediaclase{i}-mediaglob)';
    SW = SW+CovVal{i};
end

invSw_by_SB = pinv(SW) * SB;
[V,D] = eig(invSw_by_SB);
diagonal=diag(D);

[ordenados,indices]=sort(diagonal,'descend');
W=V(:,indices(1:dimensionreduccion));

reducedData = W'*datos;
end