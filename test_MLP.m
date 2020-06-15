clear all
close all

load('MLP_final.mat')
flag = 1; %true si testnumbers, false si trainnumbers

if flag
    load('Test_numbers_HW1.mat');
    dataset=Test_numbers;
else
    load('Trainnumbers.mat')
    dataset=Trainnumbers;
end

[reducedData,porcentaje]=function_pca(dataset,200);
dataset.reduced=reducedData;
output_onehot=sim(net,dataset.reduced(:,:),'useGPU','yes','showResources','no');

for i=1:size(output_onehot,2) % Selection of most activated output.
    [M,I] = max(output_onehot(:,i));
    if I==10
        output(i)=0;
    else
        output(i)=I;
    end
end

if ~flag
    bien=0;
    for i=1:length(output) % Comporbación de aciertos.
        if  output(i)==Trainnumbers.label(i)
            bien=bien+1;
        end
    end
    acierto=bien/length(output);
end