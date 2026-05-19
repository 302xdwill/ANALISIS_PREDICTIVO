# Sistema de Análisis Multivariado y Predicción 📊

Una plataforma web interactiva desarrollada en Python con **Streamlit** diseñada para realizar cálculos matriciales avanzados, pronósticos de series de tiempo y evaluación de clasificadores mediante conversión paramétrica de datos. Ideal para entornos académicos de Ingeniería de Sistemas y análisis estadístico profesional.

## 🚀 Características Principales

### 1. 🧮 Regresión Lineal Múltiple (Modo Operacional Matricial)
- **Cálculo Puro con Álgebra Matricial:** Resolución explícita de ecuaciones normales mediante la pseudo-inversa de Moore-Penrose: $\beta = (X^T X)^{-1} X^T y$.
- **Desglose de Operaciones Intermedias:** Visualización interactiva de las matrices intermedias ($X$, $X^T$, $X^T X$, $(X^T X)^{-1}$ y $X^T y$) replicando minuciosamente los flujos analíticos de Microsoft Excel.
- **Formateo Avanzado:** Impresión de coeficientes con precisión en punto flotante puro y estructuración dinámica de la ecuación general ajustada a 4 decimales.
- **Gráficas de Diagnóstico:** Inclusión de gráficos avanzados de *Residuos vs. Valores Predichos* (evaluación de homocedasticidad) e *Histograma de Distribución de Errores* (verificación de normalidad).

### 2. 📈 Pronóstico de Series de Tiempo (5 Métodos de Práctica Académica)
Implementación nativa y simulación unificada de los métodos clásicos de proyección temporal solicitados en los planes de ingeniería de sistemas:
- **Método Ingenuo (Naive Method):** Proyección constante basada rigurosamente en la última observación real.
- **Método de la Media (Mean Method):** Línea de pronóstico de comportamiento estable calculada sobre la media aritmética global histórica.
- **Método de la Media Móvil Simple (SMA):** Pronóstico dinámico e iterativo mediante ventanas móviles configurables.
- **Método de la Deriva (Drift Method):** Extrapolación lineal basada en la tasa de cambio neta entre el primer y último elemento registrado.
- **Método Ingenuo Estacional (Seasonal Naive):** Replicación exacta del patrón o comportamiento cíclico anterior ajustado a una longitud de temporada personalizable.

### 3. 🎯 Conversión Paramétrica y Matriz de Confusión
- **Transformación Cuantitativa a Cualitativa:** Segmentación dinámica de variables continuas en categorías booleanas (`Alto` y `Bajo`) mediante un control deslizante interactivo de umbral crítico ($\tau$).
- **Métricas de Eficiencia de Clasificación:** Cálculo automatizado de indicadores esenciales:
  - Exactitud (Accuracy)
  - Precisión (Precision)
  - Sensibilidad / Cobertura (Sensitivity / Recall)
  - Especificidad (Specificity)
- **Interfaz Adaptativa (Fix Dark Mode):** Mapa de calor interactivo para la Matriz de Confusión optimizado y testeado para la visualización tanto en **Modo Oscuro** (Dark Mode) como en **Modo Claro**, previniendo problemas de contraste u opacidad de texto.

## 🛠️ Tecnologías Utilizadas
- **Python 3**
- **Streamlit** - Framework interactivo para interfaces web de ciencia de datos.
- **NumPy** - Operaciones matriciales y álgebra lineal avanzada.
- **Pandas** - Estructuración, limpieza y manipulación de datasets en dataframes.
- **Plotly** - Renderizado de gráficas interactivas bidimensionales y mapas de calor.
- **Scikit-Learn** - Módulo auxiliar para la extracción automatizada de matrices de confusión.

## 📦 Instalación y Ejecución Local

Esta guía está optimizada para entornos **Linux (como Pop!_OS)** y desarrollo dentro de **Visual Studio Code**.

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/tu-usuario/nombre-del-repositorio.git](https://github.com/tu-usuario/nombre-del-repositorio.git)
   cd nombre-del-repositorio

2. **Instalar las dependencias obligatorias:** Asegúrate de contar con el gestor de paquetes de Python (pip) instalado y ejecuta:

    ```bash
    pip install streamlit pandas numpy scikit-learn plotly


3. **Lanzar la plataforma web:** Ejecuta el servidor web local de Streamlit desde la terminal integrada de VS Code:

    ```bash
    streamlit run app.py

Acceso al sistema:
Abre tu navegador web e ingresa a la dirección de red local predeterminada:
http://localhost:8501

**📖 Guía de Uso de la Interfaz**
**Carga de Datos:** Arrastra o sube cualquier archivo en formato .csv en la barra superior. Si no dispones de un archivo al momento, el sistema se iniciará automáticamente con el dataset predeterminado de la práctica (Usuarios, CPU, RAM, Tiempo).

**Manuales Integrados:** Cada componente de selección cuenta con un indicador visual explicativo (📖) que detalla de forma didáctica qué tipo de dato requiere la plataforma (variables continuas, series cronológicas, índices numéricos o factores causales).

**Análisis Dinámico:** Todas las interpretaciones de comportamiento multivariado y resúmenes estadísticos se redactan de manera inteligente absorbiendo directamente los nombres de las columnas del archivo cargado.

**Desarrollado como proyecto aplicativo para la Escuela de Ingeniería de Sistemas de la Universidad Peruana Unión (UPeU).**