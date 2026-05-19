import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# Configuración estética global adaptativa
st.set_page_config(page_title="Analizador Multivariado", layout="wide")

st.markdown("""
    <style>
    /* Diseño adaptativo de pestañas */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: rgba(128, 128, 128, 0.1); 
        border-radius: 4px 4px 0px 0px; 
        padding: 10px 20px; 
        font-weight: bold; 
    }
    .stTabs [aria-selected="true"] { 
        background-color: rgba(15, 44, 89, 0.8) !important; 
        color: white !important; 
        border-bottom: 3px solid #ff9f43;
    }
    
    /* Cajas contenedoras de métricas */
    div.stMetric { 
        background-color: rgba(128, 128, 128, 0.1); 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid rgba(128, 128, 128, 0.3); 
    }
    
    /* Texto instructivo del manual */
    .manual-text { 
        color: #4da6ff; 
        font-size: 0.9em; 
        margin-top: -10px; 
        margin-bottom: 15px; 
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Sistema de Análisis Multivariado y Predicción")
st.markdown("Plataforma interactiva para cálculos matriciales de regresión, proyecciones de series de tiempo y evaluación de clasificación.")

# --- MANEJO Y CARGA DE DATASETS ---
uploaded_file = st.file_uploader("Sube tu archivo de datos en formato (.csv)", type=["csv"])

default_data = {
    "Usuarios": [10, 20, 30, 40, 50],
    "CPU": [30, 45, 60, 70, 85],
    "RAM": [4, 6, 8, 10, 12],
    "Tiempo": [90, 135, 180, 230, 290]
}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("¡Archivo CSV cargado correctamente!")
else:
    st.info("💡 Utilizando el dataset predeterminado de la práctica de ingeniería. Sube tu archivo CSV para procesar datos personalizados.")
    df = pd.DataFrame(default_data)

with st.expander("🔍 Inspeccionar Matriz de Datos Activa"):
    st.dataframe(df, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["🧮 Regresión Múltiple (procedimiento)", "📈 Series de Tiempo", "🎯 Matriz de Confusión"])

# ==========================================
# PESTAÑA 1: REGRESIÓN LINEAL MÚLTIPLE MATRICIAL
# ==========================================
with tab1:
    st.header("📋 Desarrollo Operacional de Regresión Múltiple")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            col_y = st.selectbox("🎯 Variable Dependiente (y):", numeric_cols, index=len(numeric_cols)-1, key="reg_y")
            st.markdown("<p class='manual-text'>📖 Manual: DEBE ser un dato numérico continuo. Representa el valor objetivo o de salida a predecir.</p>", unsafe_allow_html=True)
        with col2:
            cols_x = st.multiselect("🔍 Variables Independientes (X):", [c for c in numeric_cols if c != col_y], default=[c for c in numeric_cols if c != col_y][:3], key="reg_x")
            st.markdown("<p class='manual-text'>📖 Manual: DEBEN ser datos numéricos. Son los factores del sistema que causan o influyen sobre 'y'.</p>", unsafe_allow_html=True)
        
        if col_y and cols_x:
            # --- ALGEBRA MATRICIAL COMPLETA ---
            y_arr = df[col_y].to_numpy()
            X_base = df[cols_x].to_numpy()
            ones_col = np.ones(len(df))
            X_mat = np.column_stack((ones_col, X_base))
            
            Xt = X_mat.T
            XtX = Xt @ X_mat
            XtX_inv = np.linalg.pinv(XtX)
            Xty = Xt @ y_arr
            B = XtX_inv @ Xty
            
            # --- DESPLIEGUE PASO A PASO (ESTILO EXCEL) ---
            st.subheader("1. Flujo Matricial del Procedimiento (Estilo Tablas de Excel)")
            
            with st.expander("Ver Matrices Intermedias del Cálculo", expanded=True):
                st.markdown("**Paso A: Matriz de Diseño $X$ (con columna base de 1s para el intercepto)**")
                st.dataframe(pd.DataFrame(X_mat, columns=["Intercepto (B0)"] + cols_x), use_container_width=True)
                
                st.markdown("**Paso B: Matriz Transpuesta $X^T$**")
                st.dataframe(pd.DataFrame(Xt, index=["Intercepto (B0)"] + cols_x), use_container_width=True)
                
                st.markdown("**Paso C: Producto Matricial $X^T X$**")
                nombres_mat = ["Intercepto"] + cols_x
                st.dataframe(pd.DataFrame(XtX, columns=nombres_mat, index=nombres_mat), use_container_width=True)
                
                st.markdown("**Paso D: Matriz Pseudo-Inversa $(X^T X)^{-1}$**")
                st.dataframe(pd.DataFrame(XtX_inv, columns=nombres_mat, index=nombres_mat), use_container_width=True)
                
                st.markdown("**Paso E: Vector Producto $X^T y$**")
                st.dataframe(pd.DataFrame(Xty, index=nombres_mat, columns=[f"Proyección {col_y}"]), use_container_width=True)

            # --- PARSEO DE TEXTO EXPLICATIVO EXACTO ---
            st.subheader("2. Resultados Analíticos Estrictos")
            
            # Formatear el bloque de coeficientes con precisión en punto flotante puro
            bloque_coeficientes = "Coeficientes del modelo:\n"
            bloque_coefficients_text = f"Intercepto (B0): {B[0]}\n"
            for idx, col_name in enumerate(cols_x):
                bloque_coefficients_text += f"{col_name} (B{idx+1}): {B[idx+1]}\n"
            
            st.code(bloque_coeficientes + bloque_coefficients_text, language="text")
            
            # Formatear la ecuación final a 4 decimales exactos
            bloque_ecuacion = "Ecuación del modelo:\n"
            ecuacion_formula = f"{col_y} = {B[0]:.4f}"
            for idx, col_name in enumerate(cols_x):
                ecuacion_formula += f" + {B[idx+1]:.4f}*{col_name}"
            
            st.code(bloque_ecuacion + ecuacion_formula, language="text")

            # Interpretación guiada según la naturaleza de la data cargada
            st.markdown("### 🧠 Interpretación del Comportamiento Multivariado")
            st.info(f"El punto de partida del sistema cuando todas las variables son cero es **{B[0]:.4f}**. A partir de ahí:")
            for idx, col_name in enumerate(cols_x):
                impacto = "incremento" if B[idx+1] >= 0 else "decremento"
                st.write(f"- Cada variación unitaria en la variable **'{col_name}'** genera un {impacto} directo y lineal de **{abs(B[idx+1]):.4f}** unidades sobre el parámetro **'{col_y}'**, asumiendo las demás variables constantes.")

            # --- GRÁFICAS DE DIAGNÓSTICO ESTADÍSTICO (DEL MANUAL) ---
            st.subheader("3. Gráficas de Diagnóstico y Dispersión Lineal")
            
            y_pred = X_mat @ B
            residuos = y_arr - y_pred
            
            g1, g2 = st.columns(2)
            with g1:
                # Gráfica de control de residuos: Evalúa homocedasticidad y estabilidad del error lineal
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(x=y_pred, y=residuos, mode='markers', marker=dict(color='#ff6b6b', size=10, line=dict(width=1, color='white')), name="Residuo"))
                fig_res.add_shape(type="line", x0=min(y_pred)*0.9, y0=0, x1=max(y_pred)*1.1, y1=0, line=dict(color="gray", dash="dash"))
                fig_res.update_layout(title="Gráfica de Residuos vs. Valores Predichos", xaxis_title="Valores Ajustados (Predicción)", yaxis_title="Residuos (Errores)", template="plotly_white")
                st.plotly_chart(fig_res, use_container_width=True)
                
            with g2:
                # Histograma analítico: Evalúa el supuesto estadístico de normalidad en las desviaciones
                fig_hist = px.histogram(x=residuos, nbins=8, title="Distribución de Frecuencia de los Residuos", labels={'x': 'Magnitud del Error'}, color_discrete_sequence=['#4da6ff'], template="plotly_white")
                fig_hist.update_layout(bargap=0.05)
                st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.error("El archivo cargado requiere un mínimo de dos columnas numéricas cuantitativas.")

# ==========================================
# PESTAÑA 2: SERIES DE TIEMPO (5 MÉTODOS DE LA PRÁCTICA)
# ==========================================
with tab2:
    st.header("📈 Proyección de Series de Tiempo")
    
    col1, col2 = st.columns(2)
    with col1:
        col_time = st.selectbox("📅 Eje Temporal o Secuencia (X):", df.columns, index=0, key="ts_time")
        st.markdown("<p class='manual-text'>📖 Manual: Selecciona la columna de orden cronológico o secuencias numéricas continuas.</p>", unsafe_allow_html=True)
    with col2:
        col_val = st.selectbox("📈 Variable a Pronosticar (Y):", numeric_cols, index=len(numeric_cols)-1, key="ts_val")
        st.markdown("<p class='manual-text'>📖 Manual: El parámetro continuo que deseas evaluar en el tiempo.</p>", unsafe_allow_html=True)
    
    if col_time and col_val:
        df_ts = df.dropna(subset=[col_time, col_val]).copy()
        y_hist = df_ts[col_val].to_numpy()
        fechas_hist = df_ts[col_time].astype(str).tolist()
        
        st.markdown("---")
        st.subheader("⚙️ Configuración del Pronóstico")
        
        c1, c2 = st.columns(2)
        with c1:
            metodo = st.selectbox("Selecciona el Método de Predicción:", [
                "Méto Ingenuo", 
                "Método de la Media", 
                "Método de la Media Móvil Simple", 
                "Método de la Deriva", 
                "Método Ingenuo Estacional"
            ])
        with c2:
            horizonte = st.slider("Horizonte de pronóstico (periodos a futuro):", 1, 20, 7, key="ts_horiz")
        
        ventana = 5
        estacionalidad = 7
        if metodo == "Método de la Media Móvil Simple":
            ventana = st.slider("Tamaño de la ventana móvil:", 2, len(y_hist)-1 if len(y_hist)>2 else 2, min(5, len(y_hist)-1))
        elif metodo == "Método Ingenuo Estacional":
            estacionalidad = st.slider("Longitud de la temporada:", 2, len(y_hist)//2 if len(y_hist)>3 else 2, min(7, max(2, len(y_hist)//2)))

        fechas_futuras = [f"Pronóstico +{i+1}" for i in range(horizonte)]
        fechas_total = fechas_hist + fechas_futuras
        
        predicciones = []
        descripcion_metodo = ""
        
        if metodo == "Méto Ingenuo":
            descripcion_metodo = "El método ingenuo predice fijando todo el horizonte futuro basándose en el último valor real registrado."
            predicciones = [y_hist[-1]] * horizonte
            
        elif metodo == "Método de la Media":
            descripcion_metodo = "Asigna a cada intervalo futuro la media aritmética histórica global de la variable bajo estudio."
            predicciones = [np.mean(y_hist)] * horizonte
            
        elif metodo == "Método de la Media Móvil Simple":
            descripcion_metodo = f"Establece estimaciones promediando dinámicamente las últimas {ventana} observaciones móviles."
            ventana_movil = y_hist[-ventana:].tolist()
            for _ in range(horizonte):
                nueva_media = np.mean(ventana_movil)
                predicciones.append(nueva_media)
                ventana_movil.append(nueva_media)
                ventana_movil.pop(0)
                
        elif metodo == "Método de la Deriva":
            descripcion_metodo = "Extrapola la tasa de cambio histórica media calculando la pendiente geométrica neta del dataset."
            primero, ultimo, obs = y_hist[0], y_hist[-1], len(y_hist)
            for h in range(1, horizonte + 1):
                predicciones.append(ultimo + h * ((ultimo - primero) / (obs - 1)))
                
        elif metodo == "Método Ingenuo Estacional":
            descripcion_metodo = f"Replica cíclicamente los últimos {estacionalidad} valores históricos asumiendo estacionalidad repetitiva."
            ciclo_anterior = y_hist[-estacionalidad:]
            for i in range(horizonte):
                predicciones.append(ciclo_anterior[i % estacionalidad])

        st.info(f"🧠 **Interpretación del Modelo:** {descripcion_metodo}")
        
        y_grafica_hist = y_hist.tolist() + [None] * horizonte
        y_grafica_pred = [None] * (len(y_hist) - 1) + [y_hist[-1]] + predicciones
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=fechas_total, y=y_grafica_hist, mode='lines+markers', name='Datos Históricos', line=dict(color='#b366ff', width=3), marker=dict(color='#b366ff', size=8)))
        fig_ts.add_trace(go.Scatter(x=fechas_total, y=y_grafica_pred, mode='lines+markers', name=metodo, line=dict(color='#00d2d3', dash='dot', width=3), marker=dict(symbol='diamond', size=8)))
        fig_ts.update_layout(title=f"Evolución y Pronóstico de '{col_val}' usando {metodo}", xaxis_title=col_time, yaxis_title=col_val, template="plotly_white")
        st.plotly_chart(fig_ts, use_container_width=True)

# ==========================================
# PESTAÑA 3: MATRIZ DE CONFUSIÓN Y CONVERSIÓN
# ==========================================
with tab3:
    st.header("🔲 Conversión y Matriz de Confusión")
    col_cuanti = st.selectbox("🔄 Columna Numérica a convertir (Cuantitativa -> Cualitativa):", numeric_cols, index=len(numeric_cols)-1, key="mc_cuanti")
    st.markdown("<p class='manual-text'>📖 Manual: Selecciona un parámetro continuo. Se transformará en categorías binarias ('Alto' y 'Bajo') basándose en el punto de corte.</p>", unsafe_allow_html=True)
    
    min_val, max_val, mean_val = float(df[col_cuanti].min()), float(df[col_cuanti].max()), float(df[col_cuanti].mean())
    
    umbral = st.slider(f"Ajustar Umbral de Corte para '{col_cuanti}':", min_val, max_val, mean_val, key="mc_slider")
    st.markdown("<p class='manual-text'>📖 Manual: Todo valor >= a este número será categorizado como 'Alto'. Todo valor inferior como 'Bajo'.</p>", unsafe_allow_html=True)
    
    df_eval = df.copy()
    df_eval['Clase_Real'] = np.where(df_eval[col_cuanti] >= umbral, 'Alto', 'Bajo')
    
    desviacion_modelo = st.slider("Simulador de ruido (Nivel de error del modelo predictor):", 0.0, 1.0, 0.2, step=0.1, key="mc_ruido")
    np.random.seed(42)
    valores_predichos = df_eval[col_cuanti] + np.random.normal(0, (max_val - min_val) * desviacion_modelo, len(df_eval))
    df_eval['Clase_Predicha'] = np.where(valores_predichos >= umbral, 'Alto', 'Bajo')
    
    cm = confusion_matrix(df_eval['Clase_Real'], df_eval['Clase_Predicha'], labels=['Alto', 'Bajo'])
    tp, fn, fp, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    exactitud = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    col_mat, col_met = st.columns([1, 1])
    with col_mat:
        fig_cm = px.imshow(cm, text_auto=True, x=['Alto (Predicho)', 'Bajo (Predicho)'], y=['Alto (Real)', 'Bajo (Real)'], color_continuous_scale='Blues', aspect="auto")
        fig_cm.update_layout(template="plotly_white")
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_met:
        st.metric("🎯 Exactitud (Accuracy)", f"{exactitud:.2%}")
        st.metric("🔍 Precisión (Precision)", f"{precision:.2%}")
        st.metric("⚡ Sensibilidad (Recall)", f"{sensibilidad:.2%}")
        st.metric("🛡️ Especificidad", f"{especificidad:.2%}")
        
    st.markdown("### 🧠 Interpretación de la Matriz")
    st.info(f"Con el punto de corte establecido en **{umbral:.2f}**, el modelo logra una **Exactitud del {exactitud:.2%}**. Esto significa que esa es la proporción de casos que el clasificador identificó correctamente como 'Altos' o 'Bajos' respecto al total de tu data evaluada.")