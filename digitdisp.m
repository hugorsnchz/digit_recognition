function [digito]=digitdisp(vector_imagen)

[a,~]=size(vector_imagen);

for i=1:sqrt(a)
    for j=1:sqrt(a)
        digito(i,j)=vector_imagen((i-1)*sqrt(a)+j);
    end
end

imshow(digito,[0,255]);

end