function [x, y] = preprocesadoElectricity(archivo)
    %cargamos los datos del excel
    elect = readtable(archivo);
%     disp(string(elect.datetime(:)));
%     disp(size(elect.datetime(:)));
%     disp(datenum(string(elect.datetime(:)),"yyyy-mm-dd HH:MM:SS"));
     elect.datetime =datenum(cellstr(elect.datetime),"yyyy-mm-dd HH:MM:SS");
%     
     t = table2array(elect);
      
    x = t(:,2:end-1)';
    
    y = t(:,end)';
    
    x(2:end,:)= normalize(x(2:end,:));

end 