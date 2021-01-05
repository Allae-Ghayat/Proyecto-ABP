function [x, y] = preprocesadoElectricity(archivo)
    %cargamos los datos del excel
    elect = readtable(archivo);
    
    %Separamos en un array de 2 posiciones la parte referente a fecha y
    %hora del campo datetime, realizamos esto para que la red pueda
    %aprender en base a la hora y fecha de forma separada, ya que aunque
    %sean fechas diferentes, a la misma hora puede haber consumos
    %parecidos.
     datearray = split(cellstr(elect.datetime)," ");
     %guardamos cada campo en un vector diferente
     day = datearray(:,1);
     fecha = split(day,"-");
     year = fecha(:,1);
     month = fecha(:,2);
     day = fecha(:,3);
     
     hour = datearray(:,2);
     hora = split(hour,":");
     hour = hora(:,1);
     min = hora(:,2);
     seg = hora(:,3);
     %aplicamos la transformación de datenum para transformar tanto la
     %fecha como la hora a valores numéricos
%     formatday = datenum(day,"yyyy-mm-dd");
%     formathour = datenum(hour,"HH:MM:SS");
     %debemos pasar también datetime a tipo numérico aunque no lo usemos ya
     %que sino table2array da fallo, al no ser todos los campos numéricos
     %(datetime es tipo date y todos los campos del array deben ser del
     %mismo tipo)
     
%     elect.datetime=datenum(cellstr(elect.datetime),"yyyy-mm-dd HH:MM:SS");
    
%     t = table2array(elect);
     %una vez tenemos el array con los datos añadimos los datos formateados
     %de fecha y hora, para esto pisamos el 1º campo que era ID por la
     %fecha y el 2º campo por la hora que era el anterior datetime(conjunto
     %fecha hora)
     
     t = zeros(size(day,1),8);
     
%     t(:,1) = formatday;
%     t(:,2) = formathour;
    year = str2double(year);
    month = str2double(month);
    day = str2double(day);
    hour = str2double(hour);
%    min = str2double(min);
%    seg = str2double(seg);

    t(:,1) = year;
    t(:,2) = month;
    t(:,3) = day;
    t(:,4) = hour;
%    t(:,5) = min;
%    t(:,6) = seg;
    t(:,5:8) = table2array(elect(:,3:end));
     
     %guardamos en x los campos: fecha, hora, temperatura,presion y
     %velocidad del viento
     x = t(:,1:end-1)';
     
     %guardamos en y el valor de la electricidad para esas características
     y = t(:,end)';
    
     %normalizamos las características, fecha y hora nno la normalizamos ya
     %que perderian su significado, ya que cada número representa
     %exactamente una fecha y hora
     
     %x(3:end,:)= normalize(x(3:end,:));

end 