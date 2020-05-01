close all
clear all
clc

%% Datos de trabajo.
load('Trainnumbers.mat')
data_original = Trainnumbers.image;
plot = 5; % Dimensión del subplot
index = 254; % Índice del plot a mostrar
PLOTS = 0; % Flag para plots

%% Calcula media y desviación
mn = mean(data_original,2);
stdp = std(data_original')';

for i=1:size(data_original,2)
    data(:,i)=(data_original(:,i)-mn)./stdp;
end

data(isnan(data))=0;

%% Calcula PCA
matrizCov = cov(data');
[transMat,diagonal]=eig(matrizCov);

%% Calcula pesos de las componentes principales
diagonal = diagonal./trace(diagonal)*100;
porcentaje = round(cumsum(flip(diag(diagonal))));

%% Reducción y reconstrucción de los datos

if PLOTS == 1
    figure('units','normalized','outerposition',[0 0 1 1])
    
    for j=1:plot*plot-1
        
        ncompca=251-(10*j);
        
        for i=1:ncompca
            transMatRed(i,:)=transMat(:,784+1-i)';
        end
        
        reducedData=(data'*transMatRed')';
        reconstructedData=(reducedData'*transMatRed)';
        
        for i=1:size(reconstructedData,2)
            reconstructedData(:,i)=reconstructedData(:,i).*stdp + mn;
        end
        
        subplot(plot,plot,j+1), digitdisp(reconstructedData(:,index));
        str = sprintf('Dimensión: %i, %i%%', ncompca,porcentaje(ncompca));
        title(str)
        clear transMatRed
    end
    
    subplot(plot,plot,1), digitdisp(data_original(:,index));
    title('original')
end
%% Para guardar variable
ncompca=200;

for i=1:ncompca
    transMatRed(i,:)=transMat(:,784+1-i)';
end

reducedData=(data'*transMatRed')';
save('reducedData.mat','reducedData')

clearvars -except reducedData Trainnumbers porcentaje