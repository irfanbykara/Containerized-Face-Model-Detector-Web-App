Este es un demo de una aplicación para decidir si una imagen podría ser el de un modelo o no. Para ello, estamos utilizando la arquitectura efficient-net-b4, que está entrenada en un conjunto de datos personalizado durante menos de 50 épocas. Proporciona una buena precisión en un conjunto de datos en la naturaleza. Hay dos puntos de control del modelo para que los usuarios los prueben. Uno es extremadamente exigente, que solo considera a los mejores candidatos como modelos. 
Eso es una aplicacion de django con postgresql en backend. 
Se puede probar la aplicacion directamente en googlecolab tambien sin hacer nada de programacion.

[Open in Colab]([LINK_TO_YOUR_COLAB_NOTEBOOK](https://colab.research.google.com/drive/1108U-ThcWEJ3fQLDd3IAc17RB3Qtcuj7))


# Instrucciones para Ejecutar el Proyecto

Este repositorio contiene un proyecto Django. Siga los pasos a continuación para configurar el entorno y ejecutar el servidor.

## Crear y Activar un Entorno Virtual

1. Asegúrese de tener Python 3.10 instalado en su sistema.

2. Abra una terminal y navegue hasta el directorio del proyecto.

3. Cree un nuevo entorno virtual ejecutando el siguiente comando:
    ```bash
    python3.10 -m venv myenv
    ```

4. Active el entorno virtual recién creado:
    - En Linux o macOS:
        ```bash
        source myenv/bin/activate
        ```
    - En Windows:
        ```bash
        myenv\Scripts\activate
        ```

## Instalar los Requisitos del Proyecto

1. Una vez que el entorno virtual esté activado, instale las dependencias del proyecto desde el archivo `requirements.txt` ejecutando el siguiente comando:
    ```bash
    pip install -r requirements.txt
    ```

## Ejecutar el Servidor Django

1. Después de instalar los requisitos, asegúrese de estar en la raíz del proyecto donde se encuentra el archivo `manage.py`.

2. Ejecute el servidor Django con el siguiente comando:
    ```bash
    python manage.py runserver
    ```

3. Abra su navegador web y navegue a `http://localhost:8000` para ver la aplicación en funcionamiento.

¡Disfrute explorando el proyecto!


https://github.com/irfanbykara/Containerized-Face-Model-Detector-Web-App/assets/63783207/1a475e38-9ab1-498c-beb0-ec87fcf5298e

