clear all
close all

load('MLP_final.mat')
flag = 1; %true si testnumbers, false si trainnumbers

if flag
    load('Test_numbers_HW1.mat');
    load('Trainnumbers.mat')
    [Trainnumbers.reduced, transMatRed]=function_pca2(Trainnumbers,200);
    Test_numbers.reduced = (Test_numbers.image'*transMatRed')';
    dataset=Test_numbers;
else
    load('Trainnumbers.mat')
    [Trainnumbers.reduced, transMatRed]=function_pca2(Trainnumbers,200);
    dataset=Trainnumbers;
end

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