# Clasificador de Cáncer de Piel (Skin Cancer Classifier)

Este proyecto implementa un modelo de aprendizaje profundo (Deep Learning) para la clasificación binaria de imágenes de lesiones en la piel, identificándolas como benignas o malignas. El modelo está construido utilizando PyTorch y aplica técnicas de transferencia de aprendizaje (Transfer Learning) sobre una arquitectura ResNet-18.

## Descripción y Características Principales

* **Transfer Learning y Fine-Tuning:** Utiliza el modelo ResNet-18 preentrenado en el conjunto de datos ImageNet. Las capas convolucionales se reutilizan como extractoras de características, mientras que la capa final totalmente conectada se reemplaza por una nueva adaptada a este problema de clasificación binaria.
* **Aumento de Datos (Data Augmentation):** Se aplican transformaciones dinámicas a las imágenes de entrenamiento (rotaciones de 20 grados, volteos horizontales/verticales aleatorios y ajustes de brillo y contraste) para mitigar el sobreajuste y mejorar la capacidad de generalización.
* **Optimización orientada a la Sensibilidad:** Tratándose de un diagnóstico médico donde minimizar los falsos negativos es crítico, el modelo implementa un umbral de decisión personalizado (0.001) sobre la probabilidad obtenida de la clase "maligno" mediante la función softmax. 

## Flujo de Trabajo

El desarrollo y ejecución del proyecto sigue un proceso estructurado en cinco fases principales:

1. **Preprocesamiento de Imágenes:** Las imágenes se redimensionan a 224x224 píxeles, se convierten a tensores y se normalizan utilizando los valores estándar de media y desviación típica.
2. **Partición de Datos:** Se divide el dataset original en un 80% para entrenamiento y un 20% para evaluación. Se emplea una semilla fija (42) para garantizar la total reproducibilidad de la partición. El procesamiento por lotes se gestiona mediante la clase `DataLoader`.
3. **Configuración del Modelo:** Se carga la arquitectura ResNet-18. Las primeras capas se congelan para retener los pesos originales de ImageNet, habilitando el cálculo de gradientes únicamente para el bloque convolucional final (`layer4`) y la nueva capa de clasificación (`fc`).
4. **Entrenamiento:** El modelo se somete a entrenamiento durante 100 épocas utilizando la función de pérdida de entropía cruzada y el optimizador Adam con una tasa de aprendizaje de 0.0001. Durante este proceso, se monitoriza y registra la evolución de la pérdida media por época como indicador de convergencia.
5. **Evaluación:** Se evalúa el rendimiento sobre el conjunto de prueba. Las predicciones se generan aplicando el umbral de decisión sobre las probabilidades y se calculan las métricas finales (exactitud, sensibilidad, especificidad), visualizando los resultados a través de una matriz de confusión.

## Conclusiones

A partir del desarrollo y evaluación de este clasificador, se extraen las siguientes conclusiones fundamentales:

* **Priorización de la seguridad del paciente:** La decisión de establecer un umbral de clasificación deliberadamente bajo (0.001) resulta altamente efectiva para el contexto médico. Esta configuración favorece la sensibilidad, asegurando la recuperación de la mayor parte de los casos malignos a costa de la especificidad, lo cual es el comportamiento deseado para evitar diagnósticos falsos negativos.
* **Eficacia del Transfer Learning:** La reutilización de las capas convolucionales de ResNet-18 demuestra ser una estrategia eficiente para extraer características visuales complejas de las lesiones cutáneas, permitiendo entrenar un modelo funcional sin necesidad de inicializar una arquitectura desde cero.
* **Monitorización del aprendizaje:** La evolución de la pérdida a lo largo de las 100 épocas de entrenamiento confirma la correcta convergencia del modelo y la idoneidad de la tasa de aprendizaje (0.0001) seleccionada para el optimizador Adam.

## Descarga del Dataset

Debido a las restricciones de tamaño de almacenamiento de GitHub, las imágenes originales utilizadas para el entrenamiento y prueba de este modelo se encuentran alojadas de forma externa. Pertenece a un dataset público de Kaggle.

Puede descargar el archivo `.zip` con el dataset completo desde el siguiente enlace:

[Descargar dataset de imágenes (Google Drive)](https://drive.google.com/file/d/1miWYd67BK6QyKpgthHTh9cM7d6m0FW9Q/view?usp=drive_link)

## Estructura de Directorios Esperada

Una vez descargado el dataset, descomprima el archivo y asegúrese de que la estructura de carpetas en la raíz del proyecto siga el siguiente esquema para la correcta ejecución del notebook:

```text
Images/
├── benign/
│   ├── 0836.jpg
│   └── ...
└── malignant/
    ├── 1756.jpg
    └── ...
