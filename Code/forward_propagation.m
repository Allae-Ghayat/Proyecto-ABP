function [inputs,net] = forward_propagation(net, row)
    inputs = row;
    for i=1:length(net.capa)
        new_inputs = [];

        for j=1:length(net.capa(i).neurona)
            activation = activacion(net.capa(i).neurona(j).pesos, inputs);
            net.capa(i).output(j) = transfer(activation);
            new_inputs = [new_inputs net.capa(i).output(j)];
        end
        inputs = new_inputs;
    end
end