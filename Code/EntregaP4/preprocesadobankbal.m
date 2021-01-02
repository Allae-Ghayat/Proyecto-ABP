function [xred, y] = preprocesadobank()
    %cargamos los datos del excel
    bank = readtable('bank.csv');
    %asignamos valores numéricos a las variables que son cadenas
    bank.job = grp2idx(categorical(bank.job));
    bank.marital = grp2idx(categorical(bank.marital));
    bank.education = grp2idx(categorical(bank.education));
    bank.default = grp2idx(categorical(bank.default));
    bank.housing = grp2idx(categorical(bank.housing));
    bank.loan = grp2idx(categorical(bank.loan));
    bank.contact = grp2idx(categorical(bank.contact));
    bank.month = grp2idx(categorical(bank.month));
    bank.poutcome = grp2idx(categorical(bank.poutcome));
    bank.y = grp2idx(categorical(bank.y));
    %de tabla a matriz
    t = table2array(bank);
    x = t(:,1:end-1)';
    y = t(:,end)';
    %KMEANS

    indsi = find(y==2);
    indno = find(y==1);
    xsi = x(:,indsi);
    ysi = y(:,indsi);
    xpreno = x(:,indno);
    cent = kmeans(xpreno,700);

    x = [cent xsi];
    y = [ones(1,700) ysi];
    
    %FIN KMEANS
    
    xnorm = normalize(x);
    
    [W,~,~,~,explained,~] = pca(xnorm');
    sum(explained(1:13)); %97.8348
    W = W(1:13,:);
    xred = W*xnorm; %reduccion de dimensionalidad
    %xred = xred';
end 