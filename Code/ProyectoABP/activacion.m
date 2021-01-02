function act = activacion(pesos,entrada)
    %calcula la activacion de una neurona para una entrada
    act = sum(pesos(1:end-1).*entrada)+pesos(end);
end