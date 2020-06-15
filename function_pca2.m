function [reducedData,transMatRed]=function_pca2(Trainnumbers,ncompca)

data_original = Trainnumbers.image;
mn = mean(data_original,2);
stdp = std(data_original')';

%para normalizar
% for i=1:size(data_original,2)
%     data(:,i)=(data_original(:,i)-mn)./stdp;
% end

data=data_original; %

data(isnan(data))=0;
matrizCov = cov(data');
[transMat,diagonal]=eig(matrizCov);

diagonal = diagonal./trace(diagonal)*100;
porcentaje = round(cumsum(flip(diag(diagonal))));

for i=1:ncompca
    transMatRed(i,:)=transMat(:,784+1-i)';
end

reducedData=(data'*transMatRed')';

end