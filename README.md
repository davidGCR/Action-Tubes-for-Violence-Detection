# Action Tubes for Violence Detection

Generacion de Action Tubes para deteccion espacio-temporal de acciones violentas en video. Método propuesto en [[1]](#1).

## Descripcion

 Un action tube es un conjunto de detecciones espaciales que delimitan espacial y temporalmente una accion humana como correr, pelear, etc. El presente repositorio genera action tubes a partir de detecciones espaciales usando imagenes dinamicas y un algoritmo de tracking capaz de construir multiples action tubes en paralelo. Los tubes generados pueden ser usados como propuestas para localizar acciones violentas en video. Más detalle sobre el método vea [[1]](#1).

![_q5Nwh4Z6ao_0](https://user-images.githubusercontent.com/18419040/175229468-ed49919f-767e-415d-add9-dc9708ed4f60.gif)
![6Rl7q_kXYbg_2](https://user-images.githubusercontent.com/18419040/175229585-9e2a6b68-3514-4096-842c-f291117462f9.gif)
![83d-16REC40_0](https://user-images.githubusercontent.com/18419040/175229977-e0098226-d509-4d75-ba25-ebe5630769b4.gif)
![EFv961C5RgY_0](https://user-images.githubusercontent.com/18419040/175231117-9c94e826-4ec4-4f09-8408-f0222ad0511e.gif)
![FXC43fACfPc_0](https://user-images.githubusercontent.com/18419040/175234530-2c1dd970-416a-4e97-bd85-ca4dff620693.gif)
![qaAclucigpY_3](https://user-images.githubusercontent.com/18419040/175236943-f791b7a5-03c9-47bb-87b3-458f0c870144.gif)


## Empezando

### Dependencias

* Windows 10 o Ubuntu 18+.
* Python3.
* OpenCV.
* Pytorch.

### Instalación

* Cree un virtual environment de python en la carpeta raiz del proyecto ejecutando el siguiente comando.
```
python -m venv .env
```

* Active el virtual environment creado en el paso anterior. En windows ejecute el siguiente comando
```
.\.env\Scripts\activate
```

* Para instalar las dependencias ejecute el siguiente comando.

```
pip install -r requirements.txt
```
## Ejecución
### 1. Base de Videos

* Descargue la base de datos RWF-2000 en formato zip. [[Link]](https://drive.google.com/file/d/1sJFv-A-mbUFCcNflgXeCYeDNgGQixUXC/view?usp=sharing)

* Descomprima el archivo.

* La base de datos esta compuesta de videos conteniendo acciones violentas y no violentas preprocesados como folders de imagenes/fotogramas.  

* Puede usar cualquier otro video para la extraccion de action tubes. Solo asegurese de preprocesar el video como un folder de frames.

### 2. Detección de Personas
* Para generar los action tubes es necesario extraer las regiones espaciales de personas en el video. Puede generarlas por su cuenta usando modelos como MaskRCNN, FastRCNN, etc. 

* Caso contrario, siga el [link](https://drive.google.com/file/d/180UhgMNggdnZKIzMpDMbyk5z6PQUmBr6/view?usp=sharing) para descargar las detecciones generadas para RWF-2000. Estas fueron extraidas usando el [código original](https://github.com/megvii-model/CrowdDetection.git) del detector propuesto en [[2]](#2).

### 2. Generacion de Action Tubes
Teniendo los videos y las detecciones descargadas, ejecute el siguiente comando para generar los action tubes.

```
python .\tools\demo.py --video_folder path/to/video/folder --pd_file path/to/person/detections/file.json --out_file results/tubes.json --plot True
```
*  --video_folder: Path a folder conteniendo los fotogramas
* --pd_file: Ruta de archivo JSON con las detecciones.
* --out_file: Ruta de archivo JSON para guardar los tubes generados.
* --plot: Flac para vizualizar los tubes. Puede ser True o False.


## Referencias

* <a id="1">[1]</a> Choqueluque-Roman, D.; Camara-Chavez, G. Weakly Supervised Violence Detection in Surveillance Video. Sensors 2022, 22, 4502. https://doi.org/10.3390/s22124502.

* <a id="2">[2]</a> 
Chu, X., Zheng, A., Zhang, X., & Sun, J. (2020). Detection in crowded scenes: One proposal, multiple predictions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 12214-12223).


