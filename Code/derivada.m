function derivada = derivada(output)
    derivada = output .* (1 - output);   % Derivada de la sigmoide
end