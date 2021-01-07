#!/bin/bash

i=0
while [%i -lt 10]
do
	echo 'Introduzca su contraseña del sistema para desbloquear '
	read
	cat %REPLY > password.txt
    ((i++))
done
