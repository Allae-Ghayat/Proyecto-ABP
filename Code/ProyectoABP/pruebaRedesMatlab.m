[xw, yw] = preprocesadoElectricity("train.csv");
[xw,yw] = shuffle(xw,yw);

P = 70;
    
m = round(length(xw)*P/100); %frontera entre entrenamiento y test
xtrn = xw(:,1:m);     ytrn = yw(1:m);
xtst = xw(:,m+1:end); ytst = yw(m+1:end);

    
net = newff(minmax(xtrn),[2 1],{'tansig' 'purelin'},'traingd');
% traingd traingdm traingda trainlm
net.trainParam.epochs = 100;
net.trainParam.goal = 0.01;
net = train(net,xtrn,ytrn);
y2 = sim(net,xtrn);
plot(xtrn,ytrn,'o'), hold on, plot(xtrn,y2,'xr')
y2 = sim(net,xtst);
figure, plot(xtrn,ytrn,'or'), hold on, plot(xtst,ytst,'.k'), plot(xtst,y2,'xr');

tasa_acierto = sum(y2==ytst)/length(ytst)*100;



