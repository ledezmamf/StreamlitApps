import streamlit as st
import simpy
import random
import numpy as np
import plotly.graph_objects as go

# 1. Configuración de la página
st.set_page_config(page_title="Gemelo Digital Modular", layout="wide")
st.title("Gemelo Digital Modular: Línea de Producción Dinámica")
st.markdown("Agregá, eliminá y customizá cada estación de trabajo para construir tu propia fábrica.")

# --- INICIALIZAR ESTADO DINÁMICO (SESSION STATE) ---
if 'maquinas' not in st.session_state:
    # Máquinas por defecto la primera vez que se abre la app
    st.session_state.maquinas = [
        {'nombre': 'Corte', 'automatizada': True, 'capacidad': 2, 'tiempo_proceso': 4.0, 'prob_falla': 0.05, 'tiempo_reparacion': 15},
        {'nombre': 'Ensamblaje', 'automatizada': False, 'capacidad': 3, 'tiempo_proceso': 7.0, 'prob_falla': 0.10, 'tiempo_reparacion': 30}
    ]

# --- PANEL LATERAL DE CONFIGURACIÓN ---
st.sidebar.header("1. Horarios y Turnos")
dias_simulacion = st.sidebar.number_input("Días de Simulación", min_value=1, value=5, step=1)
horas_turno = st.sidebar.number_input("Horas laborables por día", min_value=1, max_value=24, value=8, step=1)
minutos_totales = dias_simulacion * horas_turno * 60

st.sidebar.header("2. Recursos Humanos (RRHH)")
plantilla_total = st.sidebar.number_input("Total de Operarios Contratados", min_value=1, value=3, step=1)
ausentismo = st.sidebar.slider("Tasa de Ausentismo (%)", 0, 100, 10) / 100.0
operarios_presentes = max(1, round(plantilla_total * (1 - ausentismo)))

st.sidebar.header("3. Materiales")
llegada_media = st.sidebar.number_input("Llegada de Material (Minutos entre piezas)", min_value=1.0, value=5.0, step=0.5)

# --- PANEL DINÁMICO DE MÁQUINAS ---
st.sidebar.header("4. Línea de Producción")

# Iterar sobre las máquinas creadas para mostrarlas
for i, maq in enumerate(st.session_state.maquinas):
    with st.sidebar.expander(f"⚙️ Estación: {maq['nombre']}", expanded=False):
        maq['nombre'] = st.text_input("Nombre", maq['nombre'], key=f"nom_{i}")
        maq['automatizada'] = st.toggle("Automática (No usa RRHH)", value=maq['automatizada'], key=f"aut_{i}")
        maq['capacidad'] = st.number_input("Cant. de Máquinas", min_value=1, value=maq['capacidad'], key=f"cap_{i}")
        maq['tiempo_proceso'] = st.number_input("T. Proceso (min)", min_value=0.5, value=float(maq['tiempo_proceso']), step=0.5, key=f"tpro_{i}")
        maq['prob_falla'] = st.slider("Prob. Falla (%)", 0, 100, int(maq['prob_falla']*100), key=f"falla_{i}") / 100.0
        maq['tiempo_reparacion'] = st.number_input("T. Reparación (min)", min_value=1, value=int(maq['tiempo_reparacion']), key=f"trep_{i}")
        
        # Botón para eliminar esta máquina específica
        if st.button("🗑️ Eliminar Estación", key=f"del_{i}"):
            st.session_state.maquinas.pop(i)
            st.rerun()

# Botón para agregar una nueva máquina al final de la línea
if st.sidebar.button("➕ Agregar Nueva Estación"):
    nueva_maq = {
        'nombre': f'Estación {len(st.session_state.maquinas)+1}', 
        'automatizada': True, 
        'capacidad': 1, 
        'tiempo_proceso': 5.0, 
        'prob_falla': 0.0, 
        'tiempo_reparacion': 10
    }
    st.session_state.maquinas.append(nueva_maq)
    st.rerun()

# --- EJECUCIÓN DEL GEMELO DIGITAL ---
if st.sidebar.button("Ejecutar Gemelo Digital", type="primary"):
    
    if len(st.session_state.maquinas) == 0:
        st.error("⚠️ Debes agregar al menos una estación de trabajo a la línea de producción.")
    else:
        # Diccionarios dinámicos basados en la cantidad de máquinas creadas
        tiempos_espera = {maq['nombre']: [] for maq in st.session_state.maquinas}
        estadisticas = {'piezas_terminadas': 0, 'roturas_sufridas': 0, 'tiempo_perdido_roturas': 0}
    
        def generador_piezas(env, recursos_maq, rrhh):
            pieza_id = 0
            while True: 
                pieza_id += 1
                env.process(pieza(env, f'Pieza_{pieza_id}', recursos_maq, rrhh))
                yield env.timeout(random.expovariate(1.0 / llegada_media))
    
        def pieza(env, nombre, recursos_maq, rrhh):
            # La pieza viaja por TODAS las máquinas de la lista en orden secuencial
            for i, config_maq in enumerate(st.session_state.maquinas):
                llegada = env.now
                
                # Solicita la máquina correspondiente
                with recursos_maq[i].request() as req_maq:
                    yield req_maq
                    
                    # Si NO es automática, tiene que pedir un operario humano extra
                    if not config_maq['automatizada']:
                        with rrhh.request() as req_rrhh:
                            yield req_rrhh
                            tiempos_espera[config_maq['nombre']].append(env.now - llegada)
                            yield env.timeout(max(0.5, np.random.normal(config_maq['tiempo_proceso'], config_maq['tiempo_proceso'] * 0.2)))
                            
                            # Probabilidad de Falla
                            if random.random() < config_maq['prob_falla']:
                                estadisticas['roturas_sufridas'] += 1
                                t_rep = max(1, np.random.normal(config_maq['tiempo_reparacion'], config_maq['tiempo_reparacion'] * 0.2))
                                estadisticas['tiempo_perdido_roturas'] += t_rep
                                yield env.timeout(t_rep)
                                
                    # Si ES automática, avanza directo
                    else:
                        tiempos_espera[config_maq['nombre']].append(env.now - llegada)
                        yield env.timeout(max(0.5, np.random.normal(config_maq['tiempo_proceso'], config_maq['tiempo_proceso'] * 0.2)))
                        
                        # Probabilidad de Falla
                        if random.random() < config_maq['prob_falla']:
                            estadisticas['roturas_sufridas'] += 1
                            t_rep = max(1, np.random.normal(config_maq['tiempo_reparacion'], config_maq['tiempo_reparacion'] * 0.2))
                            estadisticas['tiempo_perdido_roturas'] += t_rep
                            yield env.timeout(t_rep)
                            
            # Si logró pasar por todas las máquinas, la pieza está terminada
            estadisticas['piezas_terminadas'] += 1
    
        # Iniciar Entorno
        env = simpy.Environment()
        
        # Crear los recursos físicos dinámicamente
        recursos_maquinas = [simpy.Resource(env, capacity=int(maq['capacidad'])) for maq in st.session_state.maquinas]
        rrhh = simpy.Resource(env, capacity=operarios_presentes)
        
        env.process(generador_piezas(env, recursos_maquinas, rrhh))
        
        # Correr hasta finalizar el turno
        env.run(until=minutos_totales)
    
        # --- RENDERIZADO VISUAL EN STREAMLIT ---
        st.markdown("---")
        st.markdown("### Tablero de Operaciones de la Simulación")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⏱️ Tiempo Simulado", f"{dias_simulacion} días ({horas_turno}hs/día)")
        col2.metric("👷 Operarios Activos", f"{operarios_presentes} de {plantilla_total}", f"{plantilla_total - operarios_presentes} ausentes", delta_color="red", delta_arrow="off")
        col3.metric("📦 Producción Final", f"{estadisticas['piezas_terminadas']} piezas")
        col4.metric("🔧 Fallas Mecánicas", f"{estadisticas['roturas_sufridas']} paradas", f"{int(estadisticas['tiempo_perdido_roturas']/60)} hrs perdidas", delta_color="red", delta_arrow="off")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_graf1, col_graf2 = st.columns(2)
        
        # --- GRÁFICO 1: PLOTLY WATERFALL (Cascada Continua con Truco de Espaciado) ---
        with col_graf1:
            st.markdown("**Acumulación de Tiempo de Espera en Cola (WIP)**")
            
            promedios = {k: (np.mean(v) if len(v) > 0 else 0) for k, v in tiempos_espera.items()}
            
            # TRUCO: Inyectamos un dato fantasma [' '] con valor 0 para separar el total
            estaciones = list(promedios.keys()) + [' '] + ['Total Acumulado']
            valores_y = list(promedios.values()) + [0] + [sum(promedios.values())]
            medidas = ['relative'] * len(promedios) + ['relative'] + ['total']
            textos = [f"+{v:.1f}m" for v in promedios.values()] + [""] + [f"{sum(promedios.values()):.1f}m"]
            
            fig1 = go.Figure(go.Waterfall(
                name="WIP",
                orientation="v",
                measure=medidas,
                x=estaciones,
                y=valores_y,
                textposition="outside",
                text=textos,
                connector={"line": {"color": "rgba(255,255,255,0)"}}, # Ocultamos el conector por defecto
                decreasing={"marker": {"color": "#E74C3C"}},
                increasing={"marker": {"color": "#626363", "line": {"color": "#0E1117", "width": 1}}}, # Borde sutil para distinguir barras juntas
                totals={"marker": {"color": "#C22E2E"}},
                outsidetextfont=dict(size=14)
            ))
            
            fig1.update_layout(
                template="plotly_dark",
                margin=dict(l=0, r=0, t=40, b=0),
                waterfallgap=0, # 0 espacio: obliga a las barras de proceso a tocarse
                xaxis_title="Flujo de Producción",
                yaxis_title="Minutos Acumulados",
                showlegend=False
            )

            # Forzar el rango del eje X para dar margen a los costados
            fig1.update_xaxes(range=[-1, len(estaciones) + 0.1])

            st.plotly_chart(fig1, use_container_width=True)
            
        # --- GRÁFICO 2: PLOTLY DONA MEJORADA (Líneas conectoras externas) ---
        with col_graf2:
            st.markdown("**Impacto de la Disponibilidad (OEE Simplificado)**")
            
            capacidad_total_planta = sum([m['capacidad'] for m in st.session_state.maquinas])
            t_total_maquinas = minutos_totales * capacidad_total_planta
            
            t_perdido = estadisticas['tiempo_perdido_roturas']
            t_util = t_total_maquinas - t_perdido
            
            if t_total_maquinas > 0:
                fig2 = go.Figure(go.Pie(
                    labels=['Tiempo Productivo', 'Freno por Rotura'], 
                    values=[t_util, t_perdido],
                    hole=0.65, 
                    marker=dict(colors=["#2E8DCC", "#073775"]),
                    textinfo='label+percent',
                    textposition='outside', # Genera la línea conectora hacia afuera
                    outsidetextfont=dict(color='#FAFAFA', size=14), # Contraste alto simulando el globo
                ))
                
                fig2.update_layout(
                    template="plotly_dark",
                    margin=dict(l=80, r=80, t=40, b=40), # Mucho margen para que las líneas y textos respiren
                    showlegend=False
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("No hay capacidad instalada para graficar.")
