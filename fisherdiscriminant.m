function [reducedData] = fisherdiscriminant(datos,clases)
clases=clases';
vecclases=unique(clases');
numvecclases = 10;
dimensionreduccion=9;


mediaclase = cell(1,numvecclases);
Cova = cell(1,numvecclases);
tamanoclase=zeros(1,numvecclases);


%Para cada clase:
for j=1:numvecclases
    claseactual=vecclases(1,j);
    datosclases=datos(:,clases==claseactual);
    mediaclase(j) = {mean(datosclases,2)};
    tamanoclase(j)=size(datosclases,2);
    Cova(j) = {cov(datosclases')};
    
end

mediaglob = mean(datos,2);
SB = zeros(784,784);
SW = zeros(784,784);

for i = 1:numvecclases
    
    SW = SW+Cova{i};
    SB = SB + tamanoclase(i).*(mediaclase{i}-mediaglob)*(mediaclase{i}-mediaglob)';
    
end

Mat = pinv(SW) * SB;
[V,D] = eig(Mat);
diagonal=diag(D);

[ordenados,indices]=sort(diagonal,'descend');
W=V(:,indices(1:dimensionreduccion));

reducedData = W'*datos;
end