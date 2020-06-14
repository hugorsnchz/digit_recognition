load('Trainnumbers.mat')
load('MLP_final.mat')

[reducedData,porcentaje]=function_pca(Trainnumbers,200);
Trainnumbers.reduced=reducedData;
output_onehot=sim(net,Trainnumbers.reduced(:,:),'useGPU','yes','showResources','no');

for i=1:size(output_onehot,2) % Selection of most activated output.
    [M,I] = max(output_onehot(:,i));
    if I==10
        output(i)=0;
    else
        output(i)=I;
    end
end

bien=0;
for i=1:length(output) % Comporbación de aciertos.
    if  output(i)==Trainnumbers.label(i)
        bien=bien+1;
    end
end
 acierto=bien/length(output);