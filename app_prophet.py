import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from datetime import timedelta

# 1. Configuración B2B Refinada
st.set_page_config(page_title="Refined Advanced Forecasting: Meta Prophet PRO", layout="wide")
st.title("🧠 Inteligencia Artificial: Pronóstico Avanzado de Demanda PRO (Meta Prophet)")
st.markdown("Implementación corporativa sintonizable de Prophet para series temporales. Permite la calibración manual de parámetros avanzados para optimizar el ajuste y reducir el riesgo de sobreajuste.")

# 2. Panel Lateral de Configuración Profesional
st.sidebar.header("1. Origen de Datos")
archivo_cargado = st.sidebar.file_uploader("Subir dataset propio (CSV o Excel)", type=["csv", "xlsx"])
st.sidebar.markdown("*Nota: El archivo debe tener columnas 'Fecha' y 'Ventas' con frecuencia diaria.*")

st.sidebar.header("2. Parámetros del Algoritmo (Sintonía Fina)")
horizonte_dias = st.sidebar.slider("Horizonte de Pronóstico (Días)", 30, 365, 90)

# --- Controles Avanzados de Prophet ---
st.sidebar.markdown("---")
st.sidebar.subheader("Calibración del Modelo")

# Parámetro Multiplicativo vs Aditivo
modo_estacionalidad = st.sidebar.radio(
    "Modo de Estacionalidad", 
    ['additive', 'multiplicative'],
    help="Aditiva: El efecto estacional es constante. Multiplicativa: El efecto crece/decrece proporcionalmente a la tendencia."
)

# Rango de detección de Changepoints (Tu pedido)
rango_changepoints = st.sidebar.slider(
    "Rango de detección de Cambios (Changepoint Range)",
    0.5, 1.0, 0.8,
    help="Proporción del historial (desde el inicio) donde se buscan cambios de tendencia. El valor por defecto de Prophet es 0.8."
)

# Flexibilidad de Tendencia
flexibilidad_tendencia = st.sidebar.select_slider(
    "Flexibilidad de Tendencia (Changepoint Prior)",
    options=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    value=0.05,
    help="Valores altos permiten que el modelo cambie de tendencia más bruscamente. Valores bajos fuerzan líneas más rectas."
)

# Controles de Estacionalidad Integrados
incluir_feriados = st.sidebar.toggle("Incluir Efecto Feriados (Argentina)", value=True)
if incluir_feriados:
    fuerza_feriados = st.sidebar.slider(
        "Fuerza de Feriados (Holidays Prior)", 0.1, 50.0, 10.0,
        help="Controla la importancia que el modelo le da a los días festivos (Argentina)."
    )

# 3. Motor de Datos Refinado (Sandbox Mode "Difícil")
@st.cache_data
def generar_datos_sinteticos_diarios_complejos():
    """Genera datos diarios con un quiebre de tendencia y estacionalidad ruidosa"""
    dias_historicos = 365 * 2 # 2 años de datos diarios
    fechas = pd.date_range(start='2022-01-01', periods=dias_historicos, freq='D')
    
    # 1. Tendencia con Quiebre Brusco ( Changepoint)
    # Los primeros 18 meses sube. Los últimos 6 meses colapsa.
    corte_quiebre = int(dias_historicos * 0.75)
    tendencia_1 = np.linspace(100, 300, corte_quiebre)
    tendencia_2 = np.linspace(300, 150, dias_historicos - corte_quiebre)
    tendencia = np.concatenate([tendencia_1, tendencia_2])
    
    # 2. Estacionalidad Ruidosa (Fourier + Sinusoidal)
    estacionalidad_anual = np.sin(2 * np.pi * np.arange(dias_historicos) / 365) * 50
    estacionalidad_semanal = np.where(fechas.dayofweek == 6, -40, 10) # Cae fuerte los domingos
    
    # 3. Ruido Importante
    ruido = np.random.normal(0, 30, dias_historicos) # Ruido duplicado vs anterior
    
    ventas = tendencia + estacionalidad_anual + estacionalidad_semanal + ruido
    
    # 4. Picos de Black Friday y Eventos Excepcionales
    ventas[fechas.month == 11] += np.random.uniform(50, 100, (fechas.month == 11).sum()) 
    ventas[dias_historicos - 40 : dias_historicos - 30] += 120 # Un pico aleatorio cerca del final
    
    df = pd.DataFrame({'Fecha': fechas, 'Ventas': np.maximum(ventas, 0).astype(int)})
    return df

if archivo_cargado is not None:
    try:
        if archivo_cargado.name.endswith('.csv'):
            df_crudo = pd.read_csv(archivo_cargado)
        else:
            df_crudo = pd.read_excel(archivo_cargado)
        
        # Validar columnas
        columnas_req = ['Fecha', 'Ventas']
        if not all(col in df_crudo.columns for col in columnas_req):
            st.error(f"El archivo debe contener las columnas exactas: {', '.join(columnas_req)}")
            st.stop()
            
        df_crudo['Fecha'] = pd.to_datetime(df_crudo['Fecha'])
        df_historico = df_crudo.sort_values('Fecha')
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        st.stop()
else:
    # Usamos los datos "difíciles" sintéticos
    df_historico = generar_datos_sinteticos_diarios_complejos()
    st.info("Mostrando datos sintéticos diarios 'complejos' (Sandbox Mode con Changepoint brusco). Sube tu propio archivo para analizar datos reales.")

# 4. Preparación para Prophet
df_prophet = df_historico.rename(columns={'Fecha': 'ds', 'Ventas': 'y'})

# --- FASE DE MODELADO (Validación Cronológica) ---
corte = int(len(df_prophet) * 0.8)
train = df_prophet.iloc[:corte]
test = df_prophet.iloc[corte:]

# Instanciación del Modelo con TODOS los Parámetros Profesionales Calibrados
modelo_val = Prophet(
    changepoint_prior_scale=flexibilidad_tendencia,
    changepoint_range=rango_changepoints,
    seasonality_mode=modo_estacionalidad,
    yearly_seasonality=True, 
    weekly_seasonality=True
)

if incluir_feriados:
    modelo_val.add_country_holidays(country_name='AR')
    # Nota técnica: 'holidays_prior_scale' se puede pasar al inicializar o a posteriori. 
    # Aquí lo pasamos al inicializar para simplificar el flujo si se pre-definen feriados.
    # Pero como lo activamos dinámicamente, Prophet requiere definirlo al inicio.
    # Pequeño fix: reiniciamos el modelo incluyendo holidays_prior_scale si está activo.
    
    modelo_val = Prophet(
        changepoint_prior_scale=flexibilidad_tendencia,
        changepoint_range=rango_changepoints,
        seasonality_mode=modo_estacionalidad,
        holidays_prior_scale=fuerza_feriados, # <- Agregado aquí
        yearly_seasonality=True, 
        weekly_seasonality=True
    )
    modelo_val.add_country_holidays(country_name='AR')

modelo_val.fit(train)

# Predecir sobre el test set para obtener el MAPE (Falsamente positivo al principio)
futuro_val = modelo_val.make_future_dataframe(periods=len(test), freq='D')
predicciones_val = modelo_val.predict(futuro_val)
pred_test_solo = predicciones_val.iloc[corte:]['yhat'].values

mae = np.mean(np.abs(test['y'].values - pred_test_solo))
mape = (mae / np.mean(test['y'].values)) * 100

# --- FASE DE PRODUCCIÓN (Modelo Final al 100%) ---
modelo_final = Prophet(
    changepoint_prior_scale=flexibilidad_tendencia,
    changepoint_range=rango_changepoints,
    seasonality_mode=modo_estacionalidad,
    yearly_seasonality=True, 
    weekly_seasonality=True
)
if incluir_feriados:
    modelo_final = Prophet(
        changepoint_prior_scale=flexibilidad_tendencia,
        changepoint_range=rango_changepoints,
        seasonality_mode=modo_estacionalidad,
        holidays_prior_scale=fuerza_feriados, # <- Agregado aquí
        yearly_seasonality=True, 
        weekly_seasonality=True
    )
    modelo_final.add_country_holidays(country_name='AR')

modelo_final.fit(df_prophet)

futuro_final = modelo_final.make_future_dataframe(periods=horizonte_dias, freq='D')
forecast = modelo_final.predict(futuro_final)

# 5. Dashboard Ejecutivo de KPIs Refinado
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

volumen_proyectado = forecast.iloc[-horizonte_dias:]['yhat'].sum()
pico_maximo = forecast.iloc[-horizonte_dias:]['yhat_upper'].max()

col1.metric("Registros Analizados", f"{len(df_historico)} días")
col2.metric(f"Demanda Proyectada ({horizonte_dias} días)", f"{int(volumen_proyectado):,} unid.")
col3.metric("Pico Máximo Esperado (Riesgo)", f"{int(pico_maximo)} unid/día", delta_color="inverse")
col4.metric("Precisión Histórica (MAPE OOS)", f"± {mape:.1f}%", help="Calculado sobre el último 20% que el modelo no vio.", delta_color="inverse")

# 6. Visualización Interactiva con Changepoints
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Proyección de Demanda, Intervalos de Incertidumbre y Cambios de Tendencia")
st.markdown(f"**Insight Profesional:** El rango de detección de changepoints ({rango_changepoints}) indica hasta qué fecha el modelo busca cambios de tendencia históricos. Los últimos {int((1-rango_changepoints)*100)}% de los datos se ignoran para estabilizar la proyección futura.")

fig = go.Figure()

# Puntos Históricos Reales (Ruidosos ahora)
fig.add_trace(go.Scatter(
    x=df_prophet['ds'], y=df_prophet['y'], 
    mode='markers', name='Datos Reales',
    marker=dict(color='#FAFAFA', size=3, opacity=0.4)
))

# Línea de Predicción (yhat)
fig.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat'], 
    mode='lines', name='Curva de Predicción',
    line=dict(color='#3498DB', width=2)
))

# Banda Superior/Inferior de Confianza
fig.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat_upper'], 
    mode='lines', marker=dict(color="#444"), line=dict(width=0),
    name='Límite Superior', showlegend=False
))
fig.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat_lower'], 
    mode='lines', marker=dict(color="#444"), line=dict(width=0),
    name='Banda de Incertidumbre', fill='tonexty', fillcolor='rgba(52, 152, 219, 0.2)'
))

# --- NUEVO: Visualización de Changepoints Históricos (Sintonía Fina) ---
changepoints = modelo_final.changepoints
# Tomamos solo los 20 changepoints más significativos para no saturar
magnitudes = np.abs(modelo_final.params['delta'][0])
top_cp_indices = magnitudes.argsort()[-20:][::-1]
significant_cp_dates = changepoints.iloc[top_cp_indices]

for cp_date in significant_cp_dates:
    fig.add_vline(x=cp_date, line_width=1, line_dash="dash", line_color="rgba(231, 76, 60, 0.4)") # Rojo translúcido

fig.update_layout(
    template="plotly_dark",
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis_title="Fecha",
    yaxis_title="Unidades"
)

st.plotly_chart(fig, use_container_width=True)

# 7. Descomposición de la Demanda (Insights)
st.markdown("---")
st.markdown("### 🔍 Descomposición Estructural de la Demanda (Impacto de Parámetros)")
st.markdown(f"Insight: Al cambiar a modo '{modo_estacionalidad}', Prophet modela la tendencia y estacionalidad de forma aditiva o multiplicativa.")

# Usamos la función nativa de Prophet para Plotly
fig_comp = plot_components_plotly(modelo_final, forecast)
fig_comp.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=800)
st.plotly_chart(fig_comp, use_container_width=True)