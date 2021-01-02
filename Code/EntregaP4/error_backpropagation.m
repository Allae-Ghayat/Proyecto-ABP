% Calcula el error de las capas ocultas
function net = error_backpropagation(net, expected_output)
    % Situamos el contador en la última capa
    fin = length(net.capa);
    cont = fin;
    % Recorremos la red en sentido inverso
    while cont ~= 0
        errores = [];
        % Para las capas ocultas
        if(cont ~= fin)
            capa = net.capa(cont);
            % M: Matriz con los pesos de las neuronas de una determinada
            % capa
            M = [];
            n = length(net.capa(cont+1).neurona);
            p = length(net.capa(cont+1).neurona(1).pesos);
            for j=1:length(net.capa(cont+1).neurona)
                M = [M; net.capa(cont+1).neurona(j).pesos];                    
            end            
            for i=1:length(capa.neurona)
                M(:,i);
                error = M(:,i)'*errores_aux';
                errores = [errores error];
            end
        % Para la última capa
        else
            output = net.capa(cont).output;
            errores = (expected_output - output);
        end
        errores_aux = [];
        for j=1:length(net.capa(cont).neurona)
            derivada_aux = derivada(net.capa(cont).output);
            net.capa(cont).neurona(j).error = errores(j) * derivada_aux(j);
            errores_aux = [errores_aux net.capa(cont).neurona(j).error];        
        end
        cont = cont - 1;
    end
end