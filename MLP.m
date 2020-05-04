clear all
close all

figure(); hold on
cont=1;

for h=50:10:300
    %% PARÁMETROS
    TRAIN = 1;
    nombre = 'netMLP.mat'; %en el caso de TRAIN = 0
    
    TEST = 1;
    batch = 8000;
    capas = [100,50];
    pca = h;
    
    %% DATA
    
    load('Trainnumbers.mat')
    labels=zeros(10,length(Trainnumbers.label));
    for i=1:length(Trainnumbers.label) % Conversion to one-hot vector labels.
        if Trainnumbers.label(i)==0
            labels(10,i)=1;
        else
            labels(Trainnumbers.label(i),i)=1;
        end
    end
    
    [reducedData,porcentaje]=function_pca(Trainnumbers,pca);
    
    Trainnumbers.reduced=reducedData;
    
    %% ENTRENAMIENTO
    
    if TRAIN==1
        funcion = 1; % Elección de la función de 'fit'
        switch funcion
            case 1
                net=feedforwardnet(capas,'trainscg');
            case 2
                net = fitnet(capas);
            case 3
                entrenamiento = 'trainscg';
                performance = 'mse';
                
                net=patternnet(capas,entrenamiento,performance);
                
                net.trainParam.epochs = 10000; %1000
                net.trainParam.min_grad = 1e-10; %1e-6
                net.trainParam.max_fail = 50; %6
                net.trainParam.sigma = 5.0e-5; %5.0e-5
        end
        
        % set_index = round(10000 .* rand(batch,1))'; % Random set/batch
        set_index = 1:batch;
        
        [net,tr]=train(net,Trainnumbers.reduced(:,set_index),labels(:,set_index),'useGPU','yes','showResources','no');
        
    else
        load(nombre)
    end
    
    %% TEST
    
    if TEST==1
        
        output_onehot=sim(net,Trainnumbers.reduced(:,batch+1:10000),'useGPU','yes','showResources','no');
        %output_onehot=sim(net,Trainnumbers.reduced,'useGPU','yes','showResources','yes');
        
        for i=1:size(output_onehot,2) % Selection of most activated output.
            [M,I] = max(output_onehot(:,i));
            if I==10
                output(i)=0;
            else
                output(i)=I;
            end
        end
    end
    
    bien=0;
    for i=1:length(output) % Comporbación de aciertos.
        if  output(i)==Trainnumbers.label(batch+i)
            %if  output(i)==Trainnumbers.label(i)
            bien=bien+1;
        end
    end
    acierto=bien/length(output);
    
    s0='C:\Users\hugor\Google Drive\Universidad\Máster en Automática y Robótica\Inteligencia Artificial Aplicada IAA\Digits\digit_recognition\Nets\';
    s1=strcat('_(',num2str(capas(1)),'_',num2str(capas(2)),')');
    s2=strcat('_',num2str(round(acierto,3)),'%');
    s3=strcat('_',num2str(batch));
    
    save(strcat(s0,'MLP',s2,'_',num2str(pca),s1,s3,'.mat'),'net');
    
    pca_(cont)=pca;
    acierto_(cont)=acierto;
    disp(pca)
    cont=cont+1;
    clearvars -except acierto_ pca_ cont h
end
plot(pca,acierto_);
plot(pca,acierto_,'k.');
%clearvars -except acierto output output_onehot Trainnumbers