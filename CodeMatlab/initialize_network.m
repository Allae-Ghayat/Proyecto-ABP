% n_ocultas es un vector con el nº de neuronas en cada capa oculta
function net = initialize_network(n_inputs,n_ocultas,n_salidas)

    for i=1:length(n_ocultas)+1   %numero capas
        % Capas ocultas
        if(i < length(n_ocultas)+1)
            net.capa(i).output = zeros(1,n_ocultas);
            for j=1:n_ocultas(i)   %numero de neuronas en cada capa
                net.capa(i).neurona(j).pesos = rand(n_inputs+1,1)';
                net.capa(i).neurona(j).error = 0;
            end
        else
        % Para la última capa
            net.capa(i).output = zeros(1,n_salidas);
            for j=1:n_salidas
                net.capa(i).neurona(j).pesos = rand(n_ocultas(end)+1,1)';
                net.capa(i).neurona(j).error = 0;
            end            
        end
    end
end