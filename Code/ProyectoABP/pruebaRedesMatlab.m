[xw, yw] = preprocesadoElectricity("train.csv");
[xw,yw] = shuffle(xw,yw);

P = 70;
m = round(length(xw)*P/100); %frontera entre entrenamiento y test
xtrn = xw(:,1:m);     ytrn = yw(1:m);
xtst = xw(:,m+1:end); ytst = yw(m+1:end);
% 
% xtrn = xtrn';
% ytrn = ytrn';
% xtst = xtst';
% ytst = ytst';

mmax = [min(xtrn(1,:)) max(xtrn(1,:)); min(xtrn(2,:)) max(xtrn(2,:)); min(xtrn(3,:)) max(xtrn(3,:));min(xtrn(4,:)) max(xtrn(4,:));min(xtrn(5,:)) max(xtrn(5,:))];
    
net = newff(mmax,[5 1],{'purelin' 'purelin'},'trainlm');
% traingd traingdm traingda trainlm
net.trainParam.epochs = 100;
net.trainParam.goal = 0.0001;
net = train(net,xtrn,ytrn);
youtxtrn = sim(net,xtrn);

plot(xtrn,ytrn,'ob'), hold on, plot(xtrn,youtxtrn,'xr')
title("Con entrenamiento: Entrenamiento o azul, Salida red x rojo");
youtxtst = sim(net,xtst);
figure,plot(xtst,ytst,'.g'), plot(xtst,youtxtst,'xr');
title("Con entrenamiento: Test verde puntos, Salida red x rojo");

tasa_acierto = sum(youtxtst==ytst)/length(ytst)*100;



