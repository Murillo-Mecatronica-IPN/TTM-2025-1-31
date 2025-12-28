# Guía de Instalación: ROS Melodic y Dependencias

Esta guía detalla los pasos necesarios para instalar ROS Melodic y las librerías necesarias para el funcionamiento del dron y los sensores (Kinect y RTAB-MAP).

**Nota:** Todos los comandos deben ejecutarse en una terminal de Ubuntu.

## 1. Configuración Inicial

Para todos los siguientes procesos deben de ejecutarse los comandos en una terminal, tal como se escriben.

El primer paso para la instalación es configurar que la computadora pueda aceptar paquetes de ROS.org, esto se debe a que la versión ROS Melodic es una versión antigua, para lograr esta configuración se ejecuta el siguiente comando:

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```

Posteriormente, se configuran las llaves, las llaves sirven para poder instalar objetos desde GitHub. Para lograr esto se instala el comando \textit{curl} y se obtiene se instala el repositorio de GitHub con los siguientes comandos:

```bash
sudo apt install curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```

Antes de ejecutar el comando de instalación, se verifica que el índice de paquetes de *Debian* está actualizado, esto se logra con el siguiente comando:

```bash
sudo apt update
```

## 2. Instalación de ROS Melodic

Existen diversos modos de instalación para diferentes complementos y herramientas, sin embargo, se usa la instalación sugerida por el desarrollador, que instala las herramientas base y simuladores más usados. Se usa el siguiente comando:

```bash
sudo apt install ros-melodic-desktop-full
```

## 3. Configuración del Entorno

Es conveniente que las variables de entorno ROS se añadan automáticamente a tu sesión bash cada vez que se inicia una nueva terminal. Esto es fundamental para que ROS pueda ser ejecutado y usado.

```bash
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 4. Herramientas de Gestión y Dependencias

Para crear y gestionar los espacios de trabajo ROS, existen varias herramientas y requisitos que se distribuyen por separado. Para instalar esta herramienta y otras dependencias para la creación de paquetes ROS se ejecuta:

```bash
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
```

### Inicialización de rosdep

Antes de que se puedan utilizar muchas herramientas de ROS, se tiene que inicializar *rosdep*. La herramienta de rosdep permite instalar fácilmente las dependencias del sistema para la fuente que desea compilar y es necesario para ejecutar algunos componentes básicos.

Instalación:
```bash
sudo apt install python-rosdep
```

Inicialización:
```bash
sudo rosdep init
rosdep update
```

## 5. Instalación de Drivers (Kinect y RTAB-MAP)

### Librería Freenect (Kinect)
La librería requerida para que el Kinect pueda conectarse con un sistema operativo Linux es *Freenect*. Dicha librería es la encargada de que la conexión vía USB sea posible, además de que permite probar si el dispositivo envía los datos de forma correcta. La librería se instala con el siguiente comando (funciona en cualquier versión de ROS):

```bash
sudo apt install ros-$ROS_DISTRO-freenect-stack
```

Para verificar la instalación se ejecuta el siguiente comando:

```bash
roslaunch freenect_launch freenect.launch
```

### Librería RTAB-MAP
Para la instalación de la librería RTAB-MAP se ejecuta el siguiente comando:

```bash
sudo apt install ros-$ROS_DISTRO-rtabmap-ros
```
