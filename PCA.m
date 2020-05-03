close all
clear all
clc

%% Datos de trabajo.
load('Trainnumbers.mat')
data_original = Trainnumbers.image;
plote = 5; % Dimensi�n del subplot
index = 254; % �ndice del plot a mostrar
PLOTS = 1; % Flag para plots
normalizacion=0; %Aplica normalización o no.

%% Calcula media y desviaci�n
mn = mean(data_original,2);
stdp = std(data_original')';
if normalizacion==1
    for i=1:size(data_original,2)
        data(:,i)=(data_original(:,i)-mn)./stdp;
    end
else
    data=data_original;
end

data(isnan(data))=0;

%% Calcula PCA
matrizCov = cov(data');
[transMat,diagonal]=eig(matrizCov);

%% Calcula pesos de las componentes principales
diagonal = diagonal./trace(diagonal)*100;
porcentaje = round(cumsum(flip(diag(diagonal))));

%% Reducci�n y reconstrucci�n de los datos

if PLOTS == 1
    figure('units','normalized','outerposition',[0 0 1 1]);
    
    for j=1:plote*plote-1
        
        ncompca=251-(10*j);
        
        for i=1:ncompca
            transMatRed(i,:)=transMat(:,784+1-i)';
        end
        
        reducedData=(data'*transMatRed')';
        reconstructedData=(reducedData'*transMatRed)';
        
        if normalizacion==1
            for i=1:size(reconstructedData,2)
                reconstructedData(:,i)=reconstructedData(:,i).*stdp + mn;
            end
        end
        
        subplot(plote,plote,j+1), digitdisp(reconstructedData(:,index));
        str = sprintf('Dimensi�n: %i, %i%%', ncompca,porcentaje(ncompca));
        title(str)
        clear transMatRed
        
    end
    
    subplot(plote,plote,1), digitdisp(data_original(:,index));
    title('original')
end

%% Grafica el MSE
if PLOTS == 1
    contador=1;
    
    for j=1:10:784
        
        ncompca=j;
        vecncompca(1,contador)=ncompca;
        
        for i=1:ncompca
            transMatRed(i,:)=transMat(:,784+1-i)';
        end
        
        reducedData=(data'*transMatRed')';
        reconstructedData=(reducedData'*transMatRed)';
        
        if normalizacion==1
            for i=1:size(reconstructedData,2)
                reconstructedData(:,i)=reconstructedData(:,i).*stdp + mn;
            end
        end
        
        clear transMatRed
        
        
        MSEPCA=784*mse(data_original-reconstructedData);
        vecMSEPCA(1,contador)=MSEPCA;
        contador=contador+1;
        
    end
end

figure;
plot(vecncompca,vecMSEPCA);

if normalizacion==1  
    title("MSE vs dimension (Normalized)");
else
    title("MSE vs dimension (Not normalized)");
end

xlabel("Dimension");
ylabel("MSE");

%% Para guardar variable
ncompca=200;

for i=1:ncompca
    transMatRed(i,:)=transMat(:,784+1-i)';
end

reducedData=(data'*transMatRed')';
save('reducedData.mat','reducedData')

clearvars -except reducedData Trainnumbers porcentaje