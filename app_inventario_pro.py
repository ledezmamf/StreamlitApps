import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone
from supabase import create_client

# --- Persistencia compartida entre todas tus herramientas (visitas + valoraciones) ---
# Requiere agregar a requirements.txt: supabase
# y cargar en Streamlit Secrets: supabase_url y supabase_key (ver instrucciones aparte)
NOMBRE_APP = "inventario_pro"

@st.cache_resource
def conectar_supabase():
    return create_client(st.secrets["supabase_url"], st.secrets["supabase_key"])

def registrar_evento(tipo_evento, valor=None):
    try:
        supabase = conectar_supabase()
        supabase.table("eventos_herramientas").insert({
            "app_nombre": NOMBRE_APP,
            "tipo_evento": tipo_evento,
            "valor": valor,
            "creado_en": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception:
        pass  # si falla el registro, no se interrumpe la experiencia del visitante

if "visita_registrada" not in st.session_state:
    st.session_state.visita_registrada = True
    registrar_evento("visita")

# 1. Configuración de la página
st.set_page_config(page_title="Optimización de Inventario Financiero", layout="wide")
st.title("Inventario Avanzado: Optimización Estocástica y Financiera")
st.markdown("Simulación Monte Carlo que encuentra el Punto de Pedido Óptimo minimizando el costo total (Almacenamiento vs. Quiebre de Stock).")

# 2. Panel lateral (Inputs del usuario)
st.sidebar.header("1. Comportamiento Operativo")
demanda_media = st.sidebar.number_input("Demanda Promedio (unidades/día)", value=120)
demanda_std = st.sidebar.number_input("Variación día a día (+/- unidades)", value=20)
lead_time_medio = st.sidebar.number_input("Lead Time Promedio (días)", value=90.0, step=1.0)
lead_time_std = st.sidebar.number_input("Variación Lead Time (+/- días)", value=30.0, step=1.0)

st.sidebar.header("2. Restricciones Logísticas")
punto_pedido_actual = st.sidebar.number_input("Punto de Pedido Actual", value=12000, step=50)
tamano_lote = st.sidebar.number_input("Múltiplo de Compra (Tamaño Pallet)", min_value=1, value=100, step=10)
cantidad_a_pedir = st.sidebar.number_input(
    "Cantidad a pedir cada vez (unidades)", min_value=tamano_lote, value=50000, step=tamano_lote,
    help="Cuánto pedís en cada orden de compra. La frecuencia de pedidos se calcula sola a partir de esto y de tu demanda, no se ingresa por separado."
)

st.sidebar.header("3. Variables Financieras (USD)")
costo_unitario = st.sidebar.number_input("Costo Unitario del Producto ($)", value=50.0, step=1.0)
tasa_mantenimiento = st.sidebar.slider("Costo Anual de Almacenamiento (%)", 1, 50, 20) / 100.0
costo_quiebre = st.sidebar.number_input("Lucro Cesante por Unidad Faltante ($)", value=120.0, step=1.0)

# --- CÁLCULOS CONTABLES ANUALIZADOS ---
cantidad_a_pedir_ajustada = np.ceil(cantidad_a_pedir / tamano_lote) * tamano_lote
ciclos_por_ano = (demanda_media * 365) / cantidad_a_pedir_ajustada
dias_entre_pedidos_implicito = 365 / ciclos_por_ano
costo_mant_anual = costo_unitario * tasa_mantenimiento # Ej: $10 por unidad al año

# 3. Motor de Simulación
if st.sidebar.button("Ejecutar Optimización", type="primary"):
    
    simulaciones = 10000
    np.random.seed(42)
    
    # Simulación de Demanda durante el Lead Time
    tiempos_entrega = np.maximum(0.1, np.random.normal(lead_time_medio, lead_time_std, simulaciones))
    demanda_espera = np.maximum(0, np.random.normal(
        loc=tiempos_entrega * demanda_media, 
        scale=np.sqrt(tiempos_entrega) * demanda_std
    ))
    
    demanda_media_total = np.mean(demanda_espera)
    
    # Evaluar la situación ACTUAL
    quiebres_actuales = np.sum(demanda_espera > punto_pedido_actual)
    prob_quiebre_actual = (quiebres_actuales / simulaciones) * 100
    
    # Buscar el Punto de Pedido (ROP) Óptimo
    # Limitamos la búsqueda entre la media y el extremo superior para no ensuciar el gráfico
    rop_candidatos = np.linspace(demanda_media_total * 0.5, np.max(demanda_espera), 200)
    costos_totales = []
    costos_mantenimiento = []
    costos_quiebre_lista = []
    
    for rop in rop_candidatos:
        faltante_esperado = np.mean(np.maximum(0, demanda_espera - rop))
        stock_seguridad = max(0, rop - demanda_media_total)
        
        # Costos anualizados reales
        c_mantenimiento = stock_seguridad * costo_mant_anual
        c_quiebre = faltante_esperado * costo_quiebre * ciclos_por_ano
        c_total = c_mantenimiento + c_quiebre
        
        costos_mantenimiento.append(c_mantenimiento)
        costos_quiebre_lista.append(c_quiebre)
        costos_totales.append(c_total)
        
    # Identificar el óptimo
    indice_optimo = np.argmin(costos_totales)
    rop_teorico = rop_candidatos[indice_optimo]
    
    # Ajustar por la restricción de Pallet (Tamaño de Lote)
    rop_real_pallet = np.ceil(rop_teorico / tamano_lote) * tamano_lote

    # Costo recalculado exactamente en el ROP ajustado a pallet (no el vecino más
    # cercano de la grilla de 200 candidatos, que era una aproximación innecesaria)
    faltante_esperado_pallet = np.mean(np.maximum(0, demanda_espera - rop_real_pallet))
    stock_seguridad_pallet = max(0, rop_real_pallet - demanda_media_total)
    costo_minimo_real = (
        stock_seguridad_pallet * costo_mant_anual
        + faltante_esperado_pallet * costo_quiebre * ciclos_por_ano
    )
    
    # Recalcular el riesgo con el ROP Ajustado
    quiebres_optimos = np.sum(demanda_espera > rop_real_pallet)
    prob_quiebre_optima = (quiebres_optimos / simulaciones) * 100

    # 4. Tablero de Resultados Visuales
    st.markdown("---")
    st.markdown("### Tablero de Decisión Financiera (Impacto Anual)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Punto Actual", f"{int(punto_pedido_actual)} unid.", f"Riesgo de quiebre: {prob_quiebre_actual:.1f}%", delta_color="blue", delta_arrow="off")
    col2.metric("Punto Sugerido", f"{int(rop_real_pallet)} unid.", f"Riesgo residual: {prob_quiebre_optima:.1f}%", delta_color="blue", delta_arrow="off")
    col3.metric("Stock de Seguridad (Inmovilizado)", f"{int(rop_real_pallet - demanda_media_total)} unid.")
    col4.metric("Costo de Riesgo Total (Anual)", f"${costo_minimo_real:,.2f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_graf1, col_graf2 = st.columns(2)
    
    # --- GRÁFICO 1: PLOTLY INTERACTIVO (La Curva "U") ---
    with col_graf1:
        st.markdown("**Trade-Off Financiero Anualizado**")
        
        fig1 = go.Figure()
        
        # Línea de Costo de Almacenamiento
        fig1.add_trace(go.Scatter(x=rop_candidatos, y=costos_mantenimiento, mode='lines', 
                                  name='Costo Almacenamiento', line=dict(color='#2ECC71', dash='dash')))
        
        # Línea de Costo de Faltante
        fig1.add_trace(go.Scatter(x=rop_candidatos, y=costos_quiebre_lista, mode='lines', 
                                  name='Costo de Faltante', line=dict(color='#E74C3C', dash='dash')))
        
        # Línea de Costo Total (La Curva U)
        fig1.add_trace(go.Scatter(x=rop_candidatos, y=costos_totales, mode='lines', 
                                  name='Costo Total', line=dict(color='#3498DB', width=3)))
        
        # Líneas verticales de referencia
        fig1.add_vline(x=punto_pedido_actual, line_width=2, line_dash="dash", line_color="#95A5A6", 
                       annotation_text="Actual", annotation_position="top left")
        fig1.add_vline(x=rop_real_pallet, line_width=2, line_dash="solid", line_color="#F1C40F", 
                       annotation_text="Sugerido (Pallet)", annotation_position="top right")

        # Diseño del gráfico
        fig1.update_layout(
            template="plotly_dark", # Cambialo a "plotly_white" si preferís fondo blanco
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            xaxis_title="Punto de Pedido Evaluado (Unidades)",
            yaxis_title="Costo Anual USD",
            hovermode="x unified" # Muestra un tooltip interactivo para todas las líneas a la vez
        )
        
        st.plotly_chart(fig1, use_container_width=True)

    # --- GRÁFICO 2: PLOTLY INTERACTIVO (Histograma) ---
    with col_graf2:
        st.markdown("**Distribución de Riesgo: Actual vs. Sugerido**")
        
        fig2 = go.Figure()
        
        # Histograma
        fig2.add_trace(go.Histogram(x=demanda_espera, nbinsx=40, marker_color="#34DBCD", 
                                    opacity=0.75, name="Escenarios Simulados"))
        
        # Líneas verticales de referencia
        fig2.add_vline(x=punto_pedido_actual, line_width=2, line_dash="dash", line_color="#95A5A6", 
                       annotation_text="Actual", annotation_position="top left")
        fig2.add_vline(x=rop_real_pallet, line_width=2, line_dash="solid", line_color="#F1C40F", 
                       annotation_text="Sugerido", annotation_position="top right")
        
        # Diseño del gráfico
        fig2.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            xaxis_title="Demanda Total durante la Espera",
            yaxis_title="Frecuencia (Escenarios)",
            bargap=0.05,
            hovermode="x"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    st.info(f"**💡 Insight Prescriptivo:** El algoritmo encontró que el balance financiero perfecto es pedir a las **{int(rop_teorico)}** unidades. Sin embargo, debido a que su proveedor exige comprar en múltiplos de **{int(tamano_lote)}**, el punto de pedido oficial debe fijarse en **{int(rop_real_pallet)}** unidades. Pidiendo **{int(cantidad_a_pedir_ajustada)}** unidades cada vez, esto implica aproximadamente **{ciclos_por_ano:.1f} pedidos por año** (uno cada **{dias_entre_pedidos_implicito:.0f} días**), con un costo de riesgo validado de **${costo_minimo_real:,.2f}**.")

# --- Valoración del visitante (siempre visible, haya corrido la simulación o no) ---
st.markdown("---")
st.markdown("#### ¿Te resultó útil esta herramienta?")
calificacion = st.feedback("stars")
if calificacion is not None:
    registrar_evento("valoracion", calificacion + 1)
    st.success("¡Gracias por tu opinión!")
