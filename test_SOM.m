clear all
close all

load('SOM_final.mat')
load('correspondencia_SOM_final.mat')
flag = 0; %true si testnumbers, false si trainnumbers

if flag
    load('Test_numbers_HW1.mat');
    dataset=Test_numbers;
else
    load('Trainnumbers.mat')
    dataset=Trainnumbers;
end

[reducedData,porcentaje]=function_pca(dataset,200);
dataset.reduced=reducedData;
outputs_train = net(dataset.reduced);
clase_SOM = vec2ind(outputs_train);
output=correspondencia(clase_SOM);