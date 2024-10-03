# Proyecto de Práctica Laboral - 2do Año de Ciencias de la Computación

Este repositorio contiene el desarrollo y los resultados obtenidos como parte de la práctica laboral del segundo año de Ciencias de la Computación, realizado por el Grupo de Trabajo Científico Estudiantil de Inteligencia Artificial. El objetivo principal es la aplicación de técnicas de aprendizaje automático (Machine Learning) para abordar problemas de clasificación, regresión y clusterización, utilizando datasets clásicos y siguiendo buenas prácticas en el desarrollo de modelos.

## Resumen
El Machine Learning (ML) es una rama de la inteligencia artificial que permite a las máquinas aprender a partir de datos. Este trabajo se enfocó en la familiarización con conceptos clave y la implementación de modelos para resolver problemas prácticos de clasificación, regresión y clusterización, siguiendo un flujo de trabajo estructurado y logrando una organización eficiente del proceso, desde el preprocesamiento de datos hasta la evaluación de modelos. Se obtuvieron resultados satisfactorios para cada una de las tareas, con una evaluación crítica de los modelos para seleccionar las mejores técnicas según el problema planteado.

## Tabla de Contenidos
1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Tareas Realizadas](#tareas-realizadas)
   - [Clasificación sobre el Wine Dataset](#clasificación-sobre-el-wine-dataset)
   - [Regresión sobre el California Housing Dataset](#regresión-sobre-el-california-housing-dataset)
   - [Clusterización de Datos Bidimensionales](#clusterización-de-datos-bidimensionales)
3. [Flujo de Trabajo](#flujo-de-trabajo)
4. [Tecnologías Utilizadas](#tecnologías-utilizadas)
5. [Resultados y Conclusiones](#resultados-y-conclusiones)

## Descripción del Proyecto
El objetivo de este proyecto fue el desarrollo de modelos de Machine Learning para abordar tareas específicas de clasificación, regresión y clusterización, aplicando técnicas de análisis exploratorio, preprocesamiento de datos, entrenamiento y evaluación de modelos. Para este propósito, se utilizaron dos datasets clásicos (Wine y California Housing) y se generó un conjunto de datos artificial para la clusterización.

## Tareas Realizadas

### Clasificación sobre el Wine Dataset
El objetivo fue clasificar muestras de vino en una de tres categorías basadas en 13 características químicas. Para ello, se realizaron los siguientes pasos:
1. **Análisis Exploratorio de Datos (EDA)**: Se identificaron las relaciones entre características mediante visualizaciones y una matriz de correlación. Se encontraron variables altamente correlacionadas como *flavanoids* y *total_phenols*, lo que motivó la creación de nuevas variables y la eliminación de las redundantes para reducir la multicolinealidad.
2. **Preprocesamiento de Datos**:
   - **Ingeniería de Rasgos**: Se crearon las variables *flavanoid_ratio* y *nonflavanoid_ratio*, eliminando las características originales para reducir redundancias.
   - **Manejo de Outliers**: Aplicación de winsorización para limitar el impacto de los valores extremos.
   - **Escalado de Datos**: Estandarización de las características mediante *StandardScaler*.
   - **Reducción de Dimensionalidad**: Se utilizó PCA para reducir la dimensionalidad, aunque no se observó una mejora significativa en el rendimiento de los modelos.
3. **Selección y Entrenamiento de Modelos**: Se entrenaron tres modelos —Regresión Logística, Random Forest y SVM— aplicando validación cruzada y optimización de hiperparámetros mediante *GridSearchCV*.
4. **Evaluación de Modelos**: Se evaluaron los modelos en el conjunto de prueba, destacando Random Forest con una alta precisión, aunque mostrando signos de sobreajuste. Se analizaron las métricas *precision*, *recall*, *F1-score*, y *AUC-ROC* para determinar el rendimiento.

### Regresión sobre el California Housing Dataset
El objetivo fue predecir el valor medio de las casas en California en función de varias características.
1. **Análisis Exploratorio de Datos (EDA)**: Se identificaron patrones y relaciones entre variables. Por ejemplo, se observó que el ingreso medio tiene una fuerte correlación con el valor medio de las casas.
2. **Preprocesamiento de Datos**:
   - **Codificación de Variables Categóricas**: La variable *ocean_proximity* fue transformada mediante *One-Hot Encoding*.
   - **Escalado de Datos**: Las características fueron estandarizadas con *StandardScaler*.
   - **Manejo de Datos Faltantes**: Imputación de valores faltantes con la media.
3. **Entrenamiento de Modelos**: Se entrenaron Árboles de Decisión y Random Forest, optimizando hiperparámetros mediante *GridSearchCV*.
4. **Evaluación de Modelos**: Se utilizaron las métricas *RMSE* y *MAE*, observando que Random Forest tuvo un rendimiento significativamente mejor que Árboles de Decisión, aunque a un mayor costo computacional.

### Clusterización de Datos Bidimensionales
1. **Generación de Datos**: Se generaron 1000 puntos distribuidos alrededor de tres centros predefinidos (0,0), (1,2) y (2,0).
2. **Aplicación de Algoritmos de Clusterización**: Se aplicaron *K-Means* y *DBSCAN* para agrupar los puntos generados. Los resultados fueron visualizados y comparados.
3. **Análisis Crítico**:
   - **Eficacia y Precisión**: Ambos algoritmos mostraron una buena capacidad para agrupar los puntos, con un *V-Measure* de 1.0.
   - **Eficiencia**: *K-Means* fue más rápido que *DBSCAN*, pero *DBSCAN* manejó mejor los datos ruidosos sin necesidad de predefinir el número de clusters.

## Flujo de Trabajo
1. **Análisis Exploratorio de Datos (EDA)**: Visualización y análisis de características.
2. **Preprocesamiento de Datos**: Limpieza, transformación y normalización de datos.
3. **Selección y Entrenamiento de Modelos**: Entrenamiento, ajuste de hiperparámetros y validación cruzada.
4. **Evaluación y Análisis de Resultados**: Comparación de métricas de rendimiento y análisis crítico de los resultados.

## Tecnologías Utilizadas
- **Lenguaje**: Python
- **Bibliotecas**:
  - **NumPy**, **Pandas**: Manipulación y análisis de datos.
  - **Matplotlib**, **Seaborn**: Visualización de datos.
  - **Scikit-learn**: Modelos de Machine Learning y preprocesamiento.
  - **Jupyter Notebooks**: Desarrollo interactivo y documentación.

## Resultados y Conclusiones
- **Clasificación**: Random Forest mostró el mejor rendimiento en la clasificación del *Wine Dataset*, aunque el sobreajuste es un riesgo a considerar.
- **Regresión**: Random Forest superó a los Árboles de Decisión para el *California Housing Dataset*, pero a un costo computacional mayor.
- **Clusterización**: *DBSCAN* fue mejor para manejar ruido, mientras que *K-Means* fue más eficiente en términos de tiempo de entrenamiento.
- **Balance Eficiencia-Eficacia**: La elección de modelos depende del contexto del problema y los recursos disponibles.

