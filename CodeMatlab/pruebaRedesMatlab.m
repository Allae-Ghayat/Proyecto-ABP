[xw, yw] = preprocesadoElectricity("train.csv");
%[xw,yw] = shuffle(xw,yw);

mean_xw = mean(xw');
std_xw = std(xw');

xw = (xw - mean_xw');
xw = xw./std_xw';

y_norm = (yw - min(yw))/(max(yw) - min(yw));

P = 70;
m = round(length(xw)*P/100); %frontera entre entrenamiento y test
xtrn = xw(:,1:m);     ytrn = y_norm(1:m);
xtst = xw(:,m+1:end); ytst = y_norm(m+1:end);

% 4 meses para entrenar y 4 para validar
% m = 2209;
% n = 4417;
% xtrn = xw(:,1:m);     ytrn = yw(1:m);
% xtst = xw(:,m+1:n);   ytst = yw(m+1:n);

mmax = minmax(xtrn);

%MODO ACTUAL
net = feedforwardnet([3],'traingd'); %solo numero de neuronas de capas ocultas
net.layers{1}.transferFcn = 'tansig'; %capa oculta
net.layers{2}.transferFcn = 'purelin'; %capa salida

net = configure(net,xtrn,ytrn); %Inicializa los pesos

view(net) 
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-3;
%net.trainParam.showWindow = 1;

net = train(net,xtrn,ytrn);
youtxtrn = sim(net,xtrn);
    
% net = newff(mmax,[5 1],{'purelin' 'purelin'},'trainlm');
% net.trainParam.epochs = 100;
% net.trainParam.goal = 0.0001;
% net = train(net,xtrn,ytrn);
% youtxtrn = sim(net,xtrn);
% 
%plot(xtrn,ytrn,'ob'), hold on, plot(xtrn,youtxtrn,'xr')
%title("Con entrenamiento: Entrenamiento o azul, Salida red x rojo");
youtxtst = sim(net,xtst);
%figure,plot(xtst,ytst,'.g'), plot(xtst,youtxtst,'xr');
%title("Con entrenamiento: Test verde puntos, Salida red x rojo");

youtxtrn = (youtxtrn * (max(yw) - min(yw))) + min(yw);
youtxtst = (youtxtst * (max(yw) - min(yw))) + min(yw);

%tasa_acierto = sum(youtxtst==ytst)/length(ytst)*100;