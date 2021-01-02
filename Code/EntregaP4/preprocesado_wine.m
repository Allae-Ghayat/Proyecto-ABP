function [x,y] = preprocesado_wine()
    load wine.data.txt;
    y = wine_data(:,1)'; %hay 178 patrones de 13 caracteristicas
    x = wine_data(:,2:end)';
    
    % Para redes neuronales al ser valores muy dispares es necesario normalizar 
   % xnorm= normalize(x); %normalizacion gaussiana
    
    %[W,f, eigval] = fisher(xnorm,y);
    
    %figure,bar(100*eigval./sum(eigval))
    %xlabel('Caracteristicas')
    %ylabel('Percentaje de varianza en los datos')
    %vemos que se puede reducir a dos
    
    %plotpat(xnorm,y),title("Original"),figure;
    W = fisher(xnorm,y,2);
    xfis = W*xnorm;
    x = xfis;
    %boxplot(xfis'); %Datos normalizados
    %figure,plotpat(xfis,y),title("Reducida");
end

