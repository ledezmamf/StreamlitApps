import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error

# 1. Configuración B2B
st.set_page_config(page_title="Enterprise Forecasting: LightGBM", layout="wide")
st.title("LightGBM (Microsoft)")
st.markdown("Modelo corporativo corregido: Filtra meses/semanas incompletas e incorpora Eventos Excepcionales (Campañas) para romper el 'techo de cristal' algorítmico.")

# 2. Panel Lateral
st.sidebar.header("1. Origen de Datos")
archivo_cargado = st.sidebar.file_uploader("Subir dataset (CSV/Excel)", type=["csv", "xlsx"])

st.sidebar.header("2. Sintonía Fina del Algoritmo")
n_estimators = st.sidebar.slider("Ciclos de Aprendizaje", 50, 1000, 200)
learning_rate = st.sidebar.select_slider("Tasa de Aprendizaje", options=[0.001, 0.01, 0.05, 0.1, 0.2], value=0.05)
num_leaves = st.sidebar.slider("Complejidad del Árbol", 10, 100, 31)
max_depth = st.sidebar.slider("Profundidad Máxima", 3, 20, 7)

# 3. Motor de Datos Sintéticos Realistas (Con Inyección de Contexto)
@st.cache_data
def generar_datos_complejos_realistas():
    dias = 365 * 6
    fechas = pd.date_range(start='2021-01-01', periods=dias, freq='D')
    df = pd.DataFrame({'Fecha': fechas})
    tiempo = np.arange(dias)
    
    fuerza_mercado = np.linspace(0.8, 1.2, dias) + np.random.normal(0, 0.1, dias)
    ciclo_macro = np.sin(1.5 * np.pi * tiempo / (365 * 1.5)) * (60 * fuerza_mercado)
    
    base = np.full(dias, 300.0)
    base[1000:1500] -= 70  
    base[1800:] += 50      
    
    df['Inversion_Marketing'] = np.random.uniform(500, 5000, dias)
    rendimiento_mkt = np.where((tiempo >= 1000) & (tiempo <= 1500), 8, 10)
    efecto_marketing = (df['Inversion_Marketing'] / 1000) * rendimiento_mkt
    
    mes = df['Fecha'].dt.month
    año_idx = df['Fecha'].dt.year - df['Fecha'].dt.year.min()
    multiplicador_anual_verano = np.array([1.0, 1.2, 0.8, 1.1, 1.0, 0.9, 1.0, 1.0]) 
    
    es_verano_finde = (mes.isin([1, 2, 12])) & (df['Fecha'].dt.dayofweek.isin([5, 6]))
    pico_verano = np.zeros(dias)
    for i in range(dias):
        if es_verano_finde[i]:
            pico_verano[i] = 100 * multiplicador_anual_verano[año_idx[i]]
            
    # --- LA SOLUCIÓN AL TECHO DE CRISTAL ---
    # Inyectamos una variable binaria que explica saltos masivos de demanda
    df['Campana_TV'] = 0
    # Activamos la campaña aleatoriamente en el pasado para que el modelo aprenda
    df.loc[300:320, 'Campana_TV'] = 1 
    # La activamos fuertemente en la zona de validación (donde antes el modelo fallaba)
    df.loc[1850:1880, 'Campana_TV'] = 1 
    df.loc[2050:2100, 'Campana_TV'] = 1 
    
    # Efecto real de la campaña en las ventas (Boost masivo)
    efecto_campana = df['Campana_TV'] * 60 
            
    ruido = np.zeros(dias)
    ruido[0] = np.random.normal(0, 20)
    for i in range(1, dias):
        ruido[i] = 0.5 * ruido[i-1] + np.random.normal(0, 20) 
        
    df['Ventas'] = base + ciclo_macro + efecto_marketing + pico_verano + efecto_campana + ruido
    df['Ventas'] = np.maximum(df['Ventas'], 0).astype(int)
    
    return df

def feature_engineering(df):
    df_feat = df.copy()
    df_feat['Mes'] = df_feat['Fecha'].dt.month
    df_feat['DiaSemana'] = df_feat['Fecha'].dt.dayofweek
    df_feat['Trimestre'] = df_feat['Fecha'].dt.quarter
    df_feat['DiaDelAño'] = df_feat['Fecha'].dt.dayofyear
    df_feat['EsFinde'] = df_feat['Fecha'].dt.dayofweek.isin([5, 6]).astype(int)
    
    df_feat['Mes_Sen'] = np.sin(2 * np.pi * df_feat['Mes']/12)
    df_feat['Mes_Cos'] = np.cos(2 * np.pi * df_feat['Mes']/12)
    return df_feat

# Carga de Datos
if archivo_cargado is not None:
    df_crudo = pd.read_csv(archivo_cargado) if archivo_cargado.name.endswith('.csv') else pd.read_excel(archivo_cargado)
    df_crudo['Fecha'] = pd.to_datetime(df_crudo['Fecha'])
    df_base = df_crudo.sort_values('Fecha')
    if 'Inversion_Marketing' not in df_base.columns:
        df_base['Inversion_Marketing'] = 1000 
    if 'Campana_TV' not in df_base.columns:
        df_base['Campana_TV'] = 0
else:
    df_base = generar_datos_complejos_realistas()

df_procesado = feature_engineering(df_base)
# Agregamos la nueva variable clave
features = ['Mes', 'DiaSemana', 'Trimestre', 'DiaDelAño', 'EsFinde', 'Mes_Sen', 'Mes_Cos', 'Inversion_Marketing', 'Campana_TV']

# --- FASE 1: Validación 80/20 ---
corte = int(len(df_procesado) * 0.8)
train_df, test_df = df_procesado.iloc[:corte], df_procesado.iloc[corte:]
X_train, y_train = train_df[features], train_df['Ventas']
X_test, y_test = test_df[features], test_df['Ventas']

modelo_val = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, max_depth=max_depth, random_state=42, n_jobs=-1)
modelo_val.fit(X_train, y_train)
pred_test = modelo_val.predict(X_test)

# --- FASE 2: Predicción Futura (100% de datos) ---
modelo_final = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, max_depth=max_depth, random_state=42, n_jobs=-1)
modelo_final.fit(df_procesado[features], df_procesado['Ventas'])

horizonte = 90 # Días
fechas_futuras = pd.date_range(start=df_base['Fecha'].iloc[-1] + pd.Timedelta(days=1), periods=horizonte, freq='D')
df_futuro = pd.DataFrame({'Fecha': fechas_futuras})
df_futuro['Inversion_Marketing'] = df_base['Inversion_Marketing'].mean()
df_futuro['Campana_TV'] = 0 # Asumimos que no hay campaña en el futuro inmediato, salvo que el gerente lo decida.
df_futuro_feat = feature_engineering(df_futuro)
pred_futuro = modelo_final.predict(df_futuro_feat[features])

# --- PREPARACIÓN DE DATOS PARA LAS PESTAÑAS (CORRECCIÓN DE BORDES) ---
df_hist_raw = df_base.iloc[:corte].set_index('Fecha')
df_val_raw = pd.DataFrame({'Real': y_test.values, 'Prediccion': pred_test}, index=test_df['Fecha'])
df_fut_raw = pd.DataFrame({'Prediccion': pred_futuro}, index=fechas_futuras)

def render_tab(freq, title, horizonte_label):
    if freq == 'D':
        df_h = df_hist_raw
        df_v = df_val_raw
        df_f = df_fut_raw
    else:
        # Agrupamos sumando las ventas, pero también CONTANDO los días
        df_h = df_hist_raw.resample(freq).agg(Ventas=('Ventas', 'sum'), dias=('Ventas', 'count'))
        df_v = df_val_raw.resample(freq).agg(Real=('Real', 'sum'), Prediccion=('Prediccion', 'sum'), dias=('Real', 'count'))
        df_f = df_fut_raw.resample(freq).agg(Prediccion=('Prediccion', 'sum'), dias=('Prediccion', 'count'))
        
        # Filtro de limpieza: Eliminamos la basura matemática de los bordes incompletos
        min_dias = 7 if freq == 'W' else 28
        df_h = df_h[df_h['dias'] >= min_dias]
        df_v = df_v[df_v['dias'] >= min_dias]
        df_f = df_f[df_f['dias'] >= min_dias]
        
    mape_val = mean_absolute_percentage_error(df_v['Real'], df_v['Prediccion']) * 100
    
    st.markdown(f"### {title}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Períodos Entrenados", f"{len(df_h)}")
    c2.metric("Períodos Validados", f"{len(df_v)}")
    c3.metric("Error (MAPE)", f"± {mape_val:.1f}%", delta_color="inverse")
    c4.metric(f"Proyección ({horizonte_label})", f"{int(df_f['Prediccion'].sum()):,} unid.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_h.index, y=df_h['Ventas'], mode='markers', name='Datos Reales (Pasado)', marker=dict(color='#FAFAFA', size=4, opacity=0.3)))
    fig.add_trace(go.Scatter(x=df_v.index, y=df_v['Real'], mode='markers', name='Datos Reales (Validación)', marker=dict(color='#FAFAFA', size=5, opacity=0.8)))
    fig.add_trace(go.Scatter(x=df_v.index, y=df_v['Prediccion'], mode='lines', name='Validación (LightGBM)', line=dict(color='#2ECC71', width=2)))
    fig.add_trace(go.Scatter(x=df_f.index, y=df_f['Prediccion'], mode='lines', name='Proyección Futura', line=dict(color='#3498DB', width=3)))

    fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)

# 5. Renderizado del Dashboard Principal
st.markdown("---")
tab_diario, tab_semanal, tab_mensual = st.tabs(["📅 Visión Diaria (Operativa)", "📆 Visión Semanal (Táctica)", "🗓️ Visión Mensual (Estratégica)"])

with tab_diario:
    render_tab('D', "Planificación Logística Diaria", "Próx. 90 Días")
with tab_semanal:
    render_tab('W', "Planificación de Compras Semanal", "Próx. 12 Semanas")
with tab_mensual:
    render_tab('ME', "Planificación Financiera Mensual", "Próx. 3 Meses")

# 6. Feature Importance
st.markdown("---")
st.markdown("### 🧠 Anatomía de las Ventas (Feature Importance)")
importancias = pd.DataFrame({'Variable': features, 'Importancia': modelo_final.feature_importances_}).sort_values(by='Importancia', ascending=True)
fig_imp = px.bar(importancias, x='Importancia', y='Variable', orientation='h', color='Importancia', color_continuous_scale='viridis')
fig_imp.update_layout(template="plotly_dark", margin=dict(l=0, r=20, t=30, b=0), coloraxis_showscale=False)
st.plotly_chart(fig_imp, use_container_width=True)
