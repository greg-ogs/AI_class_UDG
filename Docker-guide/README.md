# Instrucciones par instalar contenedores en un equipo con sistema operativo windows
## Instalar wsl

Para instalar el subsistema de linux es necesario abrir la Terminbal de windows y en nella escribir el comandp para instalar WSL, dicho comando es:
```
wsl --install
```
Reiniciar el equipo una vez finalizada la instalacion y al terminar una ventana de terminal de windows donde solicitara crear un nobre de usuario y contrasena nuevos. Una vez creados se puede cerrar la ventana.
## Instalar aplicaciones 
### Instalar PyCharm
Pycharm es un potente IDE que se puede descargar tanto en su vercion comunitaria como de paga, puede descargarse en cualquiera de sus versiones en el siguiente [link](https://www.jetbrains.com/products/compare/?product=pycharm-ce&product=pycharm).

Ejecutar el archivo .exe  y acceptar las opciones por defecto hasta llegar donde permita marcar la casilla de **Add bin folder to the path** y **Create associations**.
Al finalizar el istalador se debe de reiniciar el equipo.

### Instalar docker runetime o docker desktop
El ejecutable de Docker desktop se puede descargar desde la [pagina oficial](https://www.docker.com/products/docker-desktop/).
Al ejecutar el archivo .exe se debe dejar todo predeterminado y reiniciar una vez terminada a instalacion.
Al volver a iniciar el equipo no es necesario crear una cuenta.

## Contenedores
El primer paso es crear un nuevo proyecto, para eso abrimos **Pycharm** y creamos un nuevo projecto.

<picture>
  <img alt="Pagina de inicio de PyCharm" src="Captura de pantalla 2024-06-14 152555.png" width="900" height="440">
</picture>

Crear un proyecto nuevo en **python puro** y nombrandolo en este caso proyecto_1 y seleccionando la ubicacion donde queremos guardarlo, fuinalmente hacemos clic en crear. 

<picture>
  <img alt="Pagina de inicio de PyCharm" src="Captura de pantalla 2024-06-14 154244.png" width="900" height="440">
</picture>

Dentro de la carpeta proyecto_1 creamos un archivo con el nombre de **__Dockerfile__**. En este archivo tenemos el codigo que indicara como se construira nuestro contenedor. El codigo siguiente muestra el que usaremos esta vez.
```
FROM tensorflow/tensorflow:latest

RUN pip install Pillow matplotlib tensorflow-hub seaborn pandas 

WORKDIR /app
```
Donde :
+ La primera linea que comienza con `FROM` nos indica de donde descargaremos la imagen previamente construida por google en este caso
+ La segunda linea que comienza con `RUN` ejecuta el comando de consola que coloquemos de forma posterior, en este caso son las librerias que deseamos y no estan preinstaladas.
+ La tercera linea que comienza por `WORKDIR` como su nombre lo sugiere es el directorio de trabajo, es la carpeta en la que entraremos desde el inicio para ejecutar nustros diferentes scrips.

<picture>
  <img alt="Pagina de inicio de PyCharm" src="Captura de pantalla 2024-06-14 155145.png" width="300" height="450">
</picture>

## Construir imagen
Para construir la nueva imagen abriremos la terminal de PowerShell, que en PyCharm se encuentra en la parte inferior izquierda 

<picture>
  <img alt="Pagina de inicio de PyCharm" src="Captura de pantalla 2024-06-14 155734.png" width="300" height="450">
</picture>

y una vez abierta escribiremos el comando:
```
docker build -t gregogs/cuda:super_resolution-2.16.1 .
```
Donde:
+ `docker build`: comando para construir una imagen
+ `-t usuario/nombre:tag`
+ `.` (punto): Es el contexto para construir la imagen, es el directorio donde estan los archivos necesarios para construir la imagen, en este caso solo se requier el archivo Dockerfile.

>Tag es el identificador de la imagen, se pueden tener varias imagenes con el mismo usuario y el mismo nombre pero el tag o identificador es lo que las diferencia.

>El punto se usa para hgacer referencia al directorio actual en el que nos encontramos el la terminal de powershell.

Al usar este comando docker iniciara la descarga y construccion de la imagen la cual la podemos ver en la aplicacion de Docker desctop en el apartado de imagenes.

## Crear contenedor
Para crear un contenedor por pirmera vez, en la terminal es necesario el comando `docker run` seguido de una serie de argumentos de tal forma que el comando queda asi:
```
docker run --gpus all -it --rm -v D:\dev\NeuralNet\Stable_difussion:/app gregogs/cuda:super_resolution-2.16.1
```
Donde:
+ `--gpus all` da acceso a todas las gpus o tarjetas graficas del equipo.
+ `-it` inicia el modo interactivo.
+ `--rm` elimina el contenedor al terminar.
+ `-v directorio/de/trabajo/en/windows:/WORKDIR` monta una carpeta de windows en la carpeta que usamos como directorio de trabajo dentro del contenedor
+  `gregogs/cuda:super_resolution-2.16.1` es la imagen que usaremos para crear el contenedor.

>El directorio WORKDIR dentro del comando anterior es la misma carpeta que elegimos en el aarchivo **Dockerfile**.

Al ejecutar el comando debera de iniciarce en la consola el contenedor y debe verse de forma similiar a la siguiente:

<picture>
  <img alt="Pagina de inicio de PyCharm" src="Captura de pantalla 2024-06-14 162705.png" width="900" height="350">
</picture>

## Ejecutando un archivo de python dentro del contenedor
Para ejecutar un archivo de python dentro de un contenedor escribimos el comando `python3 file.py`.
Para salir del contenedor escribimos `exit`
