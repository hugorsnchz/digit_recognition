clear all
close all

%% PARÁMETROS
%100,[128,64]-94
%100,[64,32]-93
%50,[64,32]-93
%50,[128,32]-93
%50,[128,64]-93
%150,[128,64]-95
%200,[128,64]-95

TRAIN = 1;
TEST = 1;
batch = 9000;
capas = [128,64];

%% DATA

if exist('Trainnumbers','var') == 0
    load('Trainnumbers.mat')
    labels=zeros(10,length(Trainnumbers.label));
    for i=1:length(Trainnumbers.label) % Conversion to one-hot vector labels.
        if Trainnumbers.label(i)==0
            labels(10,i)=1;
        else
            labels(Trainnumbers.label(i),i)=1;
        end
    end
end

if exist('reducedData','var') == 0
    load('reducedData.mat')
end
Trainnumbers.image=reducedData;

%% ENTRENAMIENTO

if TRAIN==1
    net=feedforwardnet(capas);
    fprintf('Red creada. \n');
    
    % set_index = round(10000 .* rand(batch,1))'; % Random set/batch
    set_index = 1:batch;
    
    fprintf('Empezando entrenamiento.\n');
    [net,tr]=train(net,Trainnumbers.image(:,set_index),labels(:,set_index),'useGPU','yes','showResources','yes');
    save('netMLP.mat','net')
    fprintf('Entrenamiento finalizado y red guardada.\n');
else
    load('netMLP.mat')
end

%% TEST

if TEST==1
    fprintf('Empezando simulación.\n');
    
    output_onehot=sim(net,Trainnumbers.image(:,batch+1:10000),'useGPU','yes','showResources','yes');
    %output_onehot=sim(net,Trainnumbers.image,'useGPU','yes','showResources','yes');
    
    for i=1:size(output_onehot,2) % Selection of most activated output.
        [M,I] = max(output_onehot(:,i));
        if I==10
            output(i)=0;
        else
            output(i)=I;
        end
    end
end
fprintf('Simulación finalizada.\n');

bien=0;
for i=1:length(output) % Comporbación de aciertos.
   if  output(i)==Trainnumbers.label(batch+i)
   %if  output(i)==Trainnumbers.label(i)
       bien=bien+1;
   end
end
acierto=bien/length(output);

clearvars -except acierto output output_onehot reducedData Trainnumbers