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
     hour = datearray(:,2);
     %aplicamos la transformación de datenum para transformar tanto la
     %fecha como la hora a valores numéricos
     formatday = datenum(day,"yyyy-mm-dd");
     formathour = datenum(hour,"HH:MM:SS");
     %debemos pasar también datetime a tipo numérico aunque no lo usemos ya
     %que sino table2array da fallo, al no ser todos los campos numéricos
     %(datetime es tipo date y todos los campos del array deben ser del
     %mismo tipo)
     
     elect.datetime=datenum(cellstr(elect.datetime),"yyyy-mm-dd HH:MM:SS");
    
     t = table2array(elect);
     %una vez tenemos el array con los datos añadimos los datos formateados
     %de fecha y hora, para esto pisamos el 1º campo que era ID por la
     %fecha y el 2º campo por la hora que era el anterior datetime(conjunto
     %fecha hora)
     
     t(:,1) = formatday;
     t(:,2) = formathour;
     
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