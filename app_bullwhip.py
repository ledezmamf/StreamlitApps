import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 1. Configuración de la página
st.set_page_config(page_title="Dinámica de Sistemas: Efecto Látigo", layout="wide")
st.title("Dinámica de Sistemas: El Efecto Látigo (Bullwhip Effect)")
st.markdown("Simulación multicapa de una cadena de suministro para medir la amplificación de la demanda y el impacto del Lead Time.")

# 2. Panel Lateral de Parámetros
st.sidebar.header("1. Shock de Demanda (Mercado)")
demanda_base = 100
incremento_demanda = st.sidebar.slider("Incremento repentino de Demanda (%)", 0, 100, 20)
semana_shock = st.sidebar.number_input("Semana del impacto", min_value=1, value=5)

st.sidebar.header("2. Tiempos Logísticos (Lead Time)")
lead_time = st.sidebar.slider("Semanas de entrega entre eslabones", 1, 4, 2)

st.sidebar.header("3. Política de Gerenciamiento")
cobertura = st.sidebar.number_input("Semanas de Cobertura Deseada (Stock Objetivo)", value=3)
tiempo_ajuste = st.sidebar.slider("Agresividad de compra (Semanas para ajustar stock)", 1, 4, 2)
visibilidad_compartida = st.sidebar.toggle("Compartir pronóstico (Visibilidad Total)")

# 3. Motor de Simulación (Dinámica de Sistemas)
if st.sidebar.button("Ejecutar Simulación", type="primary"):
    semanas_simulacion = 30
    
    # Inicialización de la Demanda del Cliente Final
    demanda_real = np.full(semanas_simulacion, demanda_base)
    demanda_real[semana_shock:] = demanda_base * (1 + incremento_demanda / 100.0)
    
    # Estructura de datos para los actores
    actores = ['Minorista', 'Distribuidor', 'Fábrica']
    resultados = {actor: {'pedidos': np.zeros(semanas_simulacion), 'inventario': np.zeros(semanas_simulacion)} for actor in actores}
    
    # Clase simplificada y corregida para cada nodo de la cadena
    class NodoSupplyChain:
        def __init__(self):
            self.inventario = demanda_base * cobertura
            self.pedidos_en_transito = [demanda_base] * lead_time
            self.pronostico = demanda_base
            self.pedidos_historicos = []

        def procesar_semana(self, demanda_entrante, info_pos=None):
            # 1. Recibir mercadería
            mercaderia_recibida = self.pedidos_en_transito.pop(0)
            self.inventario += mercaderia_recibida
            
            # 2. Entregar mercadería
            ventas = min(self.inventario, demanda_entrante)
            self.inventario -= ventas
            
            # 3. Pronosticar (Suavización exponencial)
            if info_pos is not None:
                # Usa la venta real del cajero del supermercado para pronosticar
                self.pronostico = (0.5 * info_pos) + (0.5 * self.pronostico)
            else:
                # Efecto Látigo Clásico: usa el pedido distorsionado del eslabón anterior
                self.pronostico = (0.5 * demanda_entrante) + (0.5 * self.pronostico)
            
            # 4. Calcular pedido (Política Order-Up-To)
            inventario_objetivo = self.pronostico * cobertura
            pipeline_objetivo = self.pronostico * (lead_time - 1)
            posicion_objetivo = inventario_objetivo + pipeline_objetivo
            
            pipeline_actual = sum(self.pedidos_en_transito)
            posicion_actual = self.inventario + pipeline_actual
            
            # Compras por pánico + reposición
            ajuste = (posicion_objetivo - posicion_actual) / tiempo_ajuste
            nuevo_pedido = max(0, self.pronostico + ajuste)
            
            # 5. Emitir pedido al proveedor
            self.pedidos_en_transito.append(nuevo_pedido)
            self.pedidos_historicos.append(nuevo_pedido)
            
            return nuevo_pedido, self.inventario

    # Instanciar actores
    minorista = NodoSupplyChain()
    distribuidor = NodoSupplyChain()
    fabrica = NodoSupplyChain()
    
    # Bucle temporal de la simulación
    for t in range(semanas_simulacion):
        demanda_t = demanda_real[t]
        # El dato POS es la venta real del cliente final
        dato_pos = demanda_t if visibilidad_compartida else None
        
        # Flujo de información (El minorista siempre recibe su demanda y el dato POS coincide)
        ped_min, inv_min = minorista.procesar_semana(demanda_t, dato_pos)
        ped_dist, inv_dist = distribuidor.procesar_semana(ped_min, dato_pos)
        ped_fab, inv_fab = fabrica.procesar_semana(ped_dist, dato_pos)
        
        # Guardar resultados
        resultados['Minorista']['pedidos'][t] = ped_min
        resultados['Distribuidor']['pedidos'][t] = ped_dist
        resultados['Fábrica']['pedidos'][t] = ped_fab

    # 4. Tablero de Resultados (Visualización Plotly)
    st.markdown("---")
    st.markdown("### Amplificación de la Demanda (Efecto Látigo)")
    
    # Calcular métricas de amplificación (Varianzas)
    var_demanda = np.var(demanda_real)
    var_fabrica = np.var(resultados['Fábrica']['pedidos'])
    indice_latigo = var_fabrica / var_demanda if var_demanda > 0 else 1
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Shock Inicial (Cliente)", f"+{incremento_demanda}%", "Demanda estable", delta_arrow="off")
    col2.metric("Pico de Producción (Fábrica)", f"+{int(((max(resultados['Fábrica']['pedidos']) / demanda_base) - 1) * 100)}%", "Reacción al shock", delta_color="inverse")
    col3.metric("Índice de Bullwhip", f"{indice_latigo:.2f}x", "Amplificación de varianza", delta_color="inverse")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráfico Principal: Las Ondas del Efecto Látigo (Con Áreas Sombreadas)
    fig = go.Figure()
    
    eje_x = list(range(1, semanas_simulacion + 1))
    
    fig.add_trace(go.Scatter(x=eje_x, y=demanda_real, mode='lines', 
                             name='1. Cliente (Real)', 
                             line=dict(color='#FAFAFA', width=3, dash='dot', shape='spline')))
                             # fill='tozeroy', fillcolor='rgba(250, 250, 250, 0.05)')) # 5% de opacidad
    
    fig.add_trace(go.Scatter(x=eje_x, y=resultados['Minorista']['pedidos'], mode='lines', 
                             name='2. Minorista', 
                             line=dict(color="#EEFF00", width=2.5, shape='spline'),
                             fill='tozeroy', fillcolor='rgba(238, 255, 0, 0.10)')) # 15% de opacidad
    
    fig.add_trace(go.Scatter(x=eje_x, y=resultados['Distribuidor']['pedidos'], mode='lines', 
                             name='3. Distribuidor', 
                             line=dict(color='#F7930A', width=2.5, shape='spline'),
                             fill='tozeroy', fillcolor='rgba(247, 147, 10, 0.10)'))
                             
    fig.add_trace(go.Scatter(x=eje_x, y=resultados['Fábrica']['pedidos'], mode='lines', 
                             name='4. Fábrica', 
                             line=dict(color='#C74281', width=2.5, shape='spline'),
                             fill='tozeroy', fillcolor='rgba(199, 66, 129, 0.10)'))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis_title="Semanas de Simulación",
        yaxis_title="Cantidad de Pedidos (Unidades)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insight de Consultoría Automático
    if visibilidad_compartida:
        st.success("**💡 Insight Prescriptivo:** Al habilitar la 'Visibilidad Compartida', el índice de amplificación colapsa. Al compartir el dato de venta real (POS) en tiempo real con la fábrica, eliminás el ruido sistémico y evitás sobreproducción.")
    else:
        st.error(f"**⚠️ Alerta Sistémica:** Un incremento de solo **{incremento_demanda}%** en las tiendas provocó que la fábrica aumente su producción en un **{int(((max(resultados['Fábrica']['pedidos']) / demanda_base) - 1) * 100)}%**. Las demoras logísticas ({lead_time} semanas) y las compras por pánico desestabilizaron toda la red.")