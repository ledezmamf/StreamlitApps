import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. Configuración de la página
st.set_page_config(page_title="Forecasting: Machine Learning", layout="wide")
st.title("📈 Machine Learning: Pronóstico de Demanda")
st.markdown("Algoritmo Holt-Winters (Suavización Exponencial Triple) para detección de patrones, tendencia y estacionalidad en la cadena de suministro.")

# 2. Panel Lateral: Parámetros del Mercado (Simulador)
st.sidebar.header("1. Simular Historial de Ventas")
st.sidebar.markdown("*(Simulamos 3 años de datos mensuales)*")
tendencia = st.sidebar.slider("Fuerza de la Tendencia (Crecimiento)", 0.0, 50.0, 10.0)
estacionalidad = st.sidebar.slider("Fuerza de la Estacionalidad (Picos)", 10, 200, 80)
ruido = st.sidebar.slider("Volatilidad (Ruido del mercado)", 5, 100, 25)

st.sidebar.header("2. Parámetros del Modelo de ML")
meses_pronostico = st.sidebar.slider("Horizonte de Pronóstico (Meses a futuro)", 3, 24, 12)

# 3. Motor de Datos Sintéticos (El reemplazo de la Base de Datos)
@st.cache_data
def generar_datos_historicos(trend, seasonal, noise):
    meses_historicos = 36 # 3 años
    tiempo = np.arange(meses_historicos)
    
    # Ecuación de demanda: Nivel base + Tendencia + Onda estacional + Ruido aleatorio
    nivel_base = 500
    componente_tendencia = tiempo * trend
    # Usamos la función seno para simular la estacionalidad anual (ciclos de 12 meses)
    componente_estacional = np.sin(2 * np.pi * tiempo / 12) * seasonal
    componente_ruido = np.random.normal(0, noise, meses_historicos)
    
    demanda = nivel_base + componente_tendencia + componente_estacional + componente_ruido
    
    # Crear un DataFrame con fechas reales
    fechas = pd.date_range(start='2021-01-01', periods=meses_historicos, freq='MS')
    df = pd.DataFrame({'Fecha': fechas, 'Ventas': demanda.astype(int)})
    return df

df_historico = generar_datos_historicos(tendencia, estacionalidad, ruido)

# 4. Motor de Machine Learning (Validación 80/20 y Pronóstico Final)

# --- FASE A: Validación (Split Cronológico) ---
# Separamos el 80% para entrenar y el 20% para testear
corte = int(len(df_historico) * 0.8)
train = df_historico.iloc[:corte]
test = df_historico.iloc[corte:]

# Entrenamos un modelo SOLO con el 80%
modelo_validacion = ExponentialSmoothing(
    train['Ventas'], trend='add', seasonal='add', seasonal_periods=12
).fit(optimized=True)

# Hacemos que prediga el 20% restante que le ocultamos
predicciones_test = modelo_validacion.forecast(len(test))

# Calculamos el error absoluto sobre datos "desconocidos" para el modelo
mae_real = np.mean(np.abs(test['Ventas'] - predicciones_test))
margen_error_pct = (mae_real / test['Ventas'].mean()) * 100

# --- FASE B: El Modelo Final (Para predecir el futuro) ---
# Entrenamos el modelo definitivo con el 100% de los datos
modelo_final = ExponentialSmoothing(
    df_historico['Ventas'], trend='add', seasonal='add', seasonal_periods=12
).fit(optimized=True)

ajuste_historico = modelo_final.fittedvalues

# Generamos el pronóstico real hacia el futuro
predicciones = modelo_final.forecast(meses_pronostico)
fechas_futuras = pd.date_range(
    start=df_historico['Fecha'].iloc[-1] + pd.DateOffset(months=1), 
    periods=meses_pronostico, 
    freq='MS'
)

# 5. Métricas de Negocio
st.markdown("---")
st.markdown("### 📊 Tablero de Precisión del Modelo (Out-of-Sample)")

col1, col2, col3 = st.columns(3)
col1.metric("Venta Promedio Histórica", f"{int(df_historico['Ventas'].mean())} unid/mes")
col2.metric("Pronóstico Mes 1 (Próximo pedido)", f"{int(predicciones.iloc[0])} unid.", "Demanda Proyectada")
col3.metric("Error Real de Validación (MAPE)", f"± {margen_error_pct:.1f}%", "Calculado sobre el 20% oculto", delta_color="inverse")

# 6. Gráfico Visual Premium (Plotly)
st.markdown("<br>", unsafe_allow_html=True)

fig = go.Figure()

# Línea Histórica Real
fig.add_trace(go.Scatter(
    x=df_historico['Fecha'], y=df_historico['Ventas'], 
    mode='lines+markers', name='Historial Real (Pasado)',
    line=dict(color='#FAFAFA', width=2, shape='spline'),
    marker=dict(size=6, color='#FAFAFA')
))

# Línea del Ajuste del Modelo (Cómo el modelo entendió el pasado)
fig.add_trace(go.Scatter(
    x=df_historico['Fecha'], y=ajuste_historico, 
    mode='lines', name='Aprendizaje del Modelo',
    line=dict(color='#95A5A6', width=1, dash='dot', shape='spline')
))

# Línea del Pronóstico (El Futuro)
fig.add_trace(go.Scatter(
    x=fechas_futuras, y=predicciones, 
    mode='lines+markers', name='Pronóstico (Futuro)',
    line=dict(color='#3498DB', width=3, shape='spline'),
    marker=dict(size=8, color='#3498DB'),
    fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.15)'
))

fig.update_layout(
    template="plotly_dark",
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis_title="Eje Temporal (Meses)",
    yaxis_title="Volumen de Ventas (Unidades)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# Insight Ejecutivo
st.info(f"**💡 Insight Prescriptivo:** El algoritmo detectó un patrón estacional claro cada 12 meses. Para el próximo trimestre, se proyecta una demanda acumulada de **{int(predicciones.iloc[:3].sum())} unidades**, con un margen de error histórico de **{margen_error_pct:.1f}%**. Sugerimos ajustar el MRP (Material Requirements Planning) a estos valores para evitar quiebres de stock.")