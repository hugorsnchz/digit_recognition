clear all
close all

cont=1;
resultados = struct;

for bucle=1:20
    %% PARÁMETROS
    
    TRAIN = 1;
    TEST = 1;
    
    nombre = 'netMLP.mat'; %en el caso de TRAIN = 0
    batch = 9000;
    capas = [100 50];
    pca = 200;
    
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
        funcion = 3; % Elección de la función de 'fit'
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
                net.trainParam.max_fail = 100; %6
                net.trainParam.sigma = 5.0e-5; %5.0e-5
        end
        
        set_index = 1:batch;
        
        tic
        [net,tr]=train(net,Trainnumbers.reduced(:,set_index),labels(:,set_index),'useGPU','yes','showResources','no');
        tiempo = toc;
        
    else
        load(nombre)
    end
    
    %% TEST
    
    if TEST==1
        
        output_onehot=sim(net,Trainnumbers.reduced(:,batch+1:10000),'useGPU','yes','showResources','no');
        
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
            bien=bien+1;
        end
    end
    acierto=bien/length(output);
    
%     s0='C:\Users\hugor\Google Drive\Universidad\Máster en Automática y Robótica\Inteligencia Artificial Aplicada IAA\Digits\digit_recognition\Nets\';
%     s1=strcat('_(',num2str(capas(1)),'_',num2str(capas(2)),')');
%     s2=strcat('_',num2str(round(acierto,3)),'%');
%     s3=strcat('_',num2str(batch));
%     
    save(strcat('MLP','_',num2str(bucle),'.mat'),'net');
    
    resultados.tiempo(cont)=tiempo;
    resultados.acierto(cont)=acierto;
    resultados.tr(cont)=tr;
    resultados.output(cont,:)=output;
    resultados.pca(cont)=pca;
    
    clearvars -except resultados cont
    
    disp(cont);
    cont=cont+1;
   
end

disp(max(resultados.acierto));