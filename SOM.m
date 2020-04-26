%% Datos
clc
close all
clear all

load('Trainnumbers.mat');
if exist('reducedData','var') == 0
    load('reducedData.mat')
end
Trainnumbers.image=reducedData;

%% SOM, construcción y entrenamiento
inputs = Trainnumbers.image();
dimension1 = 50;
dimension2 = 50;
net = selforgmap([dimension1 dimension2]);
[net,tr] = train(net,inputs);
%load('netSOM.mat','net')
outputs_train = net(inputs);
clase_SOM = vec2ind(outputs_train);
save('netSOM.mat','net')

%% Correspondencia classes SOM (dim1*dim2) - clases de datos (nº clases)
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

correspondecia_matrix = reshape(correspondencia,[dimension1,dimension2]);
image(correspondecia_matrix,'CDataMapping','scaled')
colormap(hsv); colorbar;

%% Número de errores train
error_train=0;
for i=1:size(inputs,2)
    if correspondencia(clase_SOM(i))~=Trainnumbers.label(i)
        error_train=error_train+1;
    end
end
fprintf('Error de train de %f porciento. \n',100*error_train/size(inputs,2))
 
%% Clasificación de test y error de test
outputs = net(Trainnumbers.image);
classes_test = vec2ind(outputs);
error_test=0;
for i=1:length(Trainnumbers.label)
    if correspondencia(classes_test(i))~=Trainnumbers.label(i)
        error_test=error_test+1;
    end
end
fprintf('Error de test de %f porciento. \n',100*error_test/length(Trainnumbers.label))

%% Plots
% Uncomment these lines to enable various plots.
%view(net)
%figure, plotsomtop(net)
%figure, plotsomnc(net)
%figure, plotsomnd(net)
%figure, plotsomplanes(net)
%figure, plotsomhits(net,inputs)
%figure, plotsompos(net,inputs)

% e_train(cont)=100*error_train/length(p.clase);
% e_test(cont)=100*error_test/length(t.clase);
% cont=cont+1;
% %fprintf('%d datos de entrenamiento: ',a)
% fprintf('Error de entrenamiento de %f porciento. ',100*error_train/length(p.clase))
% fprintf('Error de test de %f porciento.\n',100*error_test/length(t.clase))
% 
% 
% % figure; surf(e_train); title('Train error')
% % figure; surf(e_test); title('Test error')
% 
% figure; plot(e_test); hold on; plot(e_train); legend('error de test','error de train')
% title('Error de test y de entrenamiento'); grid on