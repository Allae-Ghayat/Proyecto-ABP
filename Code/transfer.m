function t = transfer(activacion)
    %Funcion de activacion: sigmoide
    t = 1./(1 + exp(-activacion));
end