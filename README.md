# Skin Cancer Classifier

Este proyecto implementa un modelo de aprendizaje profundo (Deep Learning) para la clasificación binaria de imágenes de lesiones en la piel, identificándolas como benignas o malignas. El modelo está construido utilizando PyTorch y aplica técnicas de Transfer Learning sobre una arquitectura ResNet-18.

## Características Principales

* **Transfer Learning y Fine-Tuning:** Utiliza el modelo ResNet-18 preentrenado en el conjunto de datos ImageNet. Las primeras capas se mantienen congeladas para actuar como extractoras de características, mientras que la última capa convolucional (`layer4`) y la capa final de clasificación (`fc`) se entrenan específicamente para este problema médico.
* **Data Augmentation:** Se aplican transformaciones dinámicas a las imágenes de entrenamiento (rotaciones de 20 grados, volteos horizontales/verticales aleatorios y ajustes de brillo y contraste) para mitigar el sobreajuste y mejorar la capacidad de generalización del modelo.
* **Optimización orientada a la Sensibilidad:** Tratándose de un diagnóstico médico donde minimizar los falsos negativos es crítico, el modelo implementa un umbral de decisión personalizado (0.001) sobre la probabilidad obtenida mediante la función softmax. Esta configuración maximiza la sensibilidad (tasa de verdaderos positivos en casos malignos) a cambio de una reducción controlada en la especificidad.

## Descarga del Dataset

Debido a las restricciones de tamaño de almacenamiento de GitHub, las imágenes originales utilizadas para el entrenamiento y prueba de este modelo se encuentran alojadas de forma externa. 

Puede descargar el archivo `.zip` con el dataset completo desde el siguiente enlace:

[Descargar dataset de imágenes (Google Drive)](https://drive.google.com/file/d/1miWYd67BK6QyKpgthHTh9cM7d6m0FW9Q/view?usp=drive_link)

## Estructura de Directorios Esperada

Una vez descargado el dataset, descomprima el archivo y asegúrese de que la estructura de carpetas en la raíz del proyecto siga el siguiente esquema para la correcta ejecución del notebook:

```text
Images/
├── benign/
│   ├── 1274.jpg
│   └── ...
└── malignant/
    ├── 0076.jpg
    └── ...
