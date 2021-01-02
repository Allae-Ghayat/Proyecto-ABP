% IMPORTANTE:
% En xw: tantas filas como muestras y tantas columnas como características
% Para ello trasponemos xtrn antes de llamar a train_network
% En yw: el rango tiene que ser de 1 a nº de clases

    [xw, yw] = preprocesadoElectricity("train.csv");
    %[xtst, ytst] = preprocesadoElectricity("test.csv");

    [n_inputs,~] = size(xw);    % tantas entradas como patrones
    n_outputs = 1;        % 3 clases de vino
    n_ocultas = 4;              % dos neuronas en la capa oculta
    network = initialize_network(n_inputs, n_ocultas, n_outputs);

    epoch = 100;
    error = 0;
    l_rate = 0.1;
    N = 20; %numero de veces que vamos a repetir el proceso
    P = 70;
    
    m = round(length(xw)*P/100); %frontera entre entrenamiento y test
    
    % División de datos para entrenamiento y validación
    [xw,yw] = shuffle(xw,yw);
    
    xtrn = xw(:,1:m);     ytrn = yw(1:m);
    xtst = xw(:,m+1:end); ytst = yw(m+1:end);

    xtrn = xtrn';   xtst = xtst';
    
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
            disp("------- "+i+" -------");
        end
        tasa_acierto(epoch)=sum(y_predict==ytst)/length(ytst)*100;
        disp("------- "+epoch+" ---- "+tasa_acierto(epoch)+" -------");
    end
    
    figure, plot(1:100,tasa_acierto,'r');
    ylabel('Tasa de acierto');
    xlabel('Numero de epochs');
    