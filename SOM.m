%% Datos
clc
clear all
close all

pca=150;
batch=9000;

%% Load y conversi�n a one hot vector labels

load('Trainnumbers.mat')

% labels=zeros(10,length(Trainnumbers.label));
% for i=1:length(Trainnumbers.label) % Conversion to one-hot vector labels.
%     if Trainnumbers.label(i)==0
%         labels(10,i)=1;
%     else
%         labels(Trainnumbers.label(i),i)=1;
%     end
% end

[reducedData,porcentaje]=function_pca(Trainnumbers,pca);

Trainnumbers.reduced=reducedData;

%% SOM, construcci�n y entrenamiento

inputs_train = Trainnumbers.reduced(:,1:batch);
dimension1 = 5;
dimension2 = 5;
net = selforgmap([dimension1 dimension2]);
[net,tr] = train(net,inputs_train);
%load('netSOM.mat','net')
outputs_train = net(inputs_train);
clase_SOM = vec2ind(outputs_train);

%% Correspondencia classes SOM (dim1*dim2) - clases de datos (n� clases)

correspondencia=zeros(1,dimension1*dimension2);
for i=1:(dimension1*dimension2)
    cont=zeros(1,10);
    ind=find(clase_SOM==i);
    for j=1:length(ind)
        if Trainnumbers.label(ind(j))==0
            cont(10)=cont(10)+1;
        else
            cont(Trainnumbers.label(ind(j)))=cont(Trainnumbers.label(ind(j)))+1;
        end
    end
    
    [M,I]=max(cont);
    correspondencia(i)=I;
end

correspondencia(correspondencia==10)=0;

%% Plot de correspondencia por colores

correspondencia_matrix = reshape(correspondencia,[dimension1,dimension2]);
% image(correspondecia_matrix,'CDataMapping','scaled')
% colormap(hsv); colorbar;

%% N�mero de errores train

error_train=0;
for i=1:length(clase_SOM)
    if correspondencia(clase_SOM(i))~=Trainnumbers.label(i)
        error_train=error_train+1;
    end
end

 
%% Clasificaci�n de test y error de test

output_test = net(Trainnumbers.reduced(:,batch+1:10000));
clase_SOM_test = vec2ind(output_test);
error_test=0;
for i=1:length(clase_SOM_test)
    if correspondencia(clase_SOM_test(i))~=Trainnumbers.label(:,batch+1:10000)
        error_test=error_test+1;
    end
end

%% Confusion matrix

figure()
confusionchart(confusionmat(double(Trainnumbers.label(:,1:batch)),double(correspondencia(clase_SOM))));
figure()
confusionchart(confusionmat(double(Trainnumbers.label(:,batch+1:10000)),double(correspondencia(clase_SOM_test))));

save('netSOM.mat','net');

%% Representaci�n gr�fica

print_SOM(dimension1,clase_SOM,Trainnumbers);
disp(flip(correspondencia_matrix'));
fprintf('Error de test de %f porciento. \n',100*error_test/length(clase_SOM_test));
fprintf('Error de train de %f porciento. \n',100*error_train/length(clase_SOM));

%% Plots

% Uncomment these lines to enable various plots.

%view(net)
%figure, plotsomtop(net)
%figure, plotsomnc(net)
%figure, plotsomnd(net)
%figure, plotsomplanes(net)
%figure, plotsomhits(net,inputs)
%figure, plotsompos(net,inputs)