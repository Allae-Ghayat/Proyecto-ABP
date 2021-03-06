
    [xw, yw] = preprocesadobankbal();

    [n_inputs,~] = size(xw);    % tantas entradas como patrones
    n_outputs = max(yw);        % 3 clases de vino
    n_ocultas = 2;              % dos neuronas en la capa oculta
    network = initialize_network(n_inputs, n_ocultas, n_outputs);

    epoch = 100;
    error = 0;
    l_rate = 0.1;
    N = 20; %numero de veces que vamos a repetir el proceso
    P = 70; %porcentaje de datos de entrenamiento
    m = round(length(xw)*P/100); %frontera entre entrenamiento y test
    
    % División de datos para entrenamiento y validación
    [xw,yw] = shuffle(xw,yw);
    
    xtrn = xw(:,1:m);     ytrn = yw(1:m);
    xtst = xw(:,m+1:end); ytst = yw(m+1:end);

    xtrn = xtrn';   xtst = xtst';
    %tasa_acierto=zeros(1,100);
    for epoch=1:100
        
        %[xw,yw] = shuffle(xw,yw);

        %xtrn = xw(:,1:m);     ytrn = yw(1:m);
        %xtst = xw(:,m+1:end); ytst = yw(m+1:end);

        %xtrn = xtrn';   xtst = xtst';
    
    
        net = train_network(network, xtrn, ytrn, l_rate, epoch, n_outputs);
        %net,x,y,l_rate,epoch,n_outputs
        % Obtener las clases de los datos de validación
        y_predict = [];
        for i=1:length(ytst)
            row = xtst(i,:);
            output = forward_propagation(net, row);
            y_predict = [y_predict find(output == max(output))];
        end

        tasa_acierto(epoch)=sum(y_predict==ytst)/length(ytst)*100;
    end
    
    figure, plot(1:100,tasa_acierto,'r');
    ylabel('Tasa de acierto');
    xlabel('Numero de epochs');
    