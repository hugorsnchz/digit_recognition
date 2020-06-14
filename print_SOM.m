function print_SOM(b,clase_SOM,Trainnumbers)

%b=10; %raiz del número de neuronas
n=1/b;
cont=1;

h1=figure();
daspect([1 1 1]);

for i=1:b
    for j=1:b
        ind=find(clase_SOM==cont);
        A(cont)=subplot('Position',[n*j-n n*i-n n n]); digitdisp(Trainnumbers.image(:,ind(1)));
        %set(A(i),'YTick',[],'XTick',[]);
        cont=cont+1;
    end
end
set(h1,'position',[500,200,700,700]);
end


