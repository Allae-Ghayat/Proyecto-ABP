% learning_rate lo especificamos nosotros, por ejemplo 0.1
function net = update_weights(net, row, learning_rate)
    for i=1:length(net.capa)        
        if(i == 1)
            inputs=row(1:length(row));
        else
            inputs = net.capa(i-1).output;
        end
        % Rellenamos inputs con un 1 para que coincida con los pesos, ya
        % que el último peso es el bias
        inputs = [inputs 1];
        for j=1:length(net.capa(i).neurona)
            pesos = net.capa(i).neurona(j).pesos;
            error = net.capa(i).neurona(j).error;
            for k=1:length(inputs)
                net.capa(i).neurona(j).pesos(k) = pesos(k) + learning_rate * error * inputs(k);
            end
        end
    end
end