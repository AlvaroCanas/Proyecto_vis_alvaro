'''
Para realizar este proyecto he leído y seguido los siguientes recursos:

    -   https://docs.streamlit.io/get-started/tutorials (de aquí he aprendido sobre todo del tutorial básico de Streamlit 'First Steps')

    -   https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started (para aprender a usar Streamlit Cloud y publicar mi proyecto)

Me he enfocado en con el menor código y la mayor simpleza posible dar el mejor resultado, espero que os guste.
'''
import streamlit as st
import pandas as pd
import plotly.express as px

# hacemos la config basica de la pagina, layout wide para que ocupe todo el ancho
st.set_page_config(page_title="Dashboard Ventas", layout="wide")

# aplicamos un css custom para las tarjetas de metricas (afecta a toda la página, al final no lo uso mucho pero queda bien)
st.markdown("""
<style>
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 20px; border-radius: 10px; color: white; text-align: center; margin: 5px;}
    .metric-card h2 {font-size: 2.5em; margin: 0;}
    .metric-card p {margin: 5px 0 0 0; opacity: 0.9;}
</style>
""", unsafe_allow_html=True)

# creamos la funcion para cargar los datos, mejor usar el decorador para el cache para que no recargue de nuevo (es muy lento si no)
@st.cache_data
def load_data():
    df = pd.concat([pd.read_csv("parte_1.csv.gz"), pd.read_csv("parte_2.csv.gz")], ignore_index=True)   # juntamos los dos csv porque el original era muy grande
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

st.title("Dashboard de Ventas - IMAT Alimentación (🛒)")

# las 4 pestañas principales
tab1, tab2, tab3, tab4 = st.tabs(["Visión Global (📊)", "Por Tienda (🏪)", "Por Estado (📍)", "Extra (✨)"])

#  PESTAÑA 1: vision general de todo
with tab1:
    st.header("Visión Global de Ventas")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nº Tiendas", df['store_nbr'].nunique())
    c2.metric("Nº Productos", df['family'].nunique())
    c3.metric("Nº Estados", df['state'].nunique())
    c4.metric("Nº Meses registrados", df[['year','month']].drop_duplicates().shape[0])
    
    st.divider()  # linea separadora
    
    col1, col2 = st.columns(2)
    
    with col1:
        # top productos, barras horizontales ordenadas
        st.subheader("Top 10 Productos Más Vendidos (🏆)")

        top_prod = df.groupby('family')['sales'].sum().nlargest(10).reset_index()
        fig = px.bar(top_prod, x='sales', y='family', orientation='h', color='sales', color_continuous_scale='Viridis')

        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)

        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # treemap de tiendas, se ve mejor que un pie chart jeje
        st.subheader("Distribución Ventas por Tienda - de la 1 a la 54 (🏪)")

        ventas_tienda = df.groupby('store_nbr')['sales'].sum().reset_index()
        fig = px.treemap(ventas_tienda, path=['store_nbr'], values='sales', color='sales', color_continuous_scale='Blues')

        st.plotly_chart(fig, use_container_width=True)

    
    # Y creamos la otra fila de graficos
    col3, col4 = st.columns(2)
    
    with col3:
        # tiendas que mas venden en promo
        st.subheader("Top 10 Tiendas en Promociones (🎯)")


        promo = df[df['onpromotion'] > 0].groupby('store_nbr')['sales'].sum().nlargest(10).reset_index()
        promo = promo.sort_values('sales', ascending=True)  # ordenamos pñara que el mayor quede arriba
        promo['store_nbr'] = 'Tienda ' + promo['store_nbr'].astype(str)

        fig = px.bar(promo, x='sales', y='store_nbr', orientation='h', color='sales', color_continuous_scale='Plasma')
        fig.update_layout(yaxis={'categoryorder':'array', 'categoryarray': promo['store_nbr'].tolist()})


        st.plotly_chart(fig, use_container_width=True)


    with col4:
        # ventas por dia, ordenamos los dias manualmente
        st.subheader("Ventas por Día de la Semana (📆)")

        dia = df.groupby('day_of_week')['sales'].mean().reset_index()
        orden = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dia['day_of_week'] = pd.Categorical(dia['day_of_week'], categories=orden, ordered=True)
        dia = dia.sort_values('day_of_week')
        fig = px.bar(dia, x='day_of_week', y='sales', color='sales', color_continuous_scale='Turbo')

        st.plotly_chart(fig, use_container_width=True)
    

    # parte de la estacionalidad
    st.subheader("Estacionalidad de Ventas (📈)")
    col5, col6 = st.columns(2)
    

    with col5:
        # ventas por semana del año
        semana = df.groupby('week')['sales'].mean().reset_index()
        fig = px.line(semana, x='week', y='sales', markers=True, title="Ventas Medias por Semana")
        st.plotly_chart(fig, use_container_width=True)
    
    with col6:
        # ventas por mes
        mes = df.groupby('month')['sales'].mean().reset_index()
        fig = px.line(mes, x='month', y='sales', markers=True, title="Ventas Medias por Mes")
        st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 2: analisis individual por tienda
with tab2:
    st.header("Análisis por Tienda")
    tienda = st.selectbox("Selecciona una tienda:", sorted(df['store_nbr'].unique()))
    df_tienda = df[df['store_nbr'] == tienda]  # filtramos los datos
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # evolucion anual de esa tienda
        st.subheader("Ventas por Año (📅)")
        anual = df_tienda.groupby('year')['sales'].sum().reset_index().sort_values('year')
        fig = px.bar(anual, x='year', y='sales', color='sales', color_continuous_scale='Blues')

        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # que productos vende mas
        st.subheader("Productos Vendidos (📦)")
        prod = df_tienda.groupby('family')['sales'].sum().nlargest(10).reset_index()
        fig = px.bar(prod, x='sales', y='family', orientation='h', color='sales')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # productos en promocion de esa tienda
        st.subheader("Productos en Promoción (🎯)")
        promo = df_tienda[df_tienda['onpromotion'] > 0].groupby('family')['sales'].sum().nlargest(10).reset_index()
        fig = px.bar(promo, x='sales', y='family', orientation='h', color='sales', color_continuous_scale='Oranges')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})

        st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 3: analisis por estado/region
with tab3:
    st.header("Análisis por Estado")
    estado = st.selectbox("Selecciona un estado:", sorted(df['state'].unique()))
    df_estado = df[df['state'] == estado]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # transacciones por año en ese estado
        st.subheader("Transacciones por Año (📅)")

        trans = df_estado.groupby('year')['transactions'].sum().reset_index().sort_values('year')
        fig = px.bar(trans, x='year', y='transactions', color='transactions', color_continuous_scale='Greens')

        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ranking de tiendas del estado
        st.subheader("Tiendas con Más Ventas (🏪)")

        tiendas = df_estado.groupby('store_nbr')['sales'].sum().nlargest(10).reset_index()
        fig = px.bar(tiendas, x='store_nbr', y='sales', color='sales', color_continuous_scale='Purples')

        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # producto top del estado
        st.subheader("Producto Más Vendido (🏆)")

        prod = df_estado.groupby('family')['sales'].sum().nlargest(10).reset_index()
        fig = px.bar(prod, x='sales', y='family', orientation='h', color='sales', color_continuous_scale='Reds')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})

        st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 4: graficos extras (pata ver mejor que pasa)
with tab4:
    st.header("Estadísticas Adicionales - Mejorar la comprensión (✨)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # evolucion temporal con area (queda bien)
        st.subheader("Evolución Mensual de Ventas (📈)")

        evol = df.groupby(['year', 'month'])['sales'].sum().reset_index()
        evol['fecha'] = pd.to_datetime(evol[['year', 'month']].assign(day=1))
        fig = px.area(evol, x='fecha', y='sales', color_discrete_sequence=['#667eea'])

        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # top ciudades
        st.subheader("Ventas por Ciudad (🌆)")

        ciudad = df.groupby('city')['sales'].sum().nlargest(10).reset_index()
        fig = px.bar(ciudad, x='sales', y='city', orientation='h', color='sales', color_continuous_scale='Teal')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)

        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # como afectan los festivos a las ventas
        st.subheader("Impacto de Festivos en Ventas (🎉)")
        festivo = df.groupby('holiday_type')['sales'].mean().reset_index().sort_values('sales')
        fig = px.bar(festivo, x='sales', y='holiday_type', orientation='h', color='holiday_type', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(showlegend=False)

        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # tipos de tienda (A, B, C, D, E)
        st.subheader("Ventas por Tipo de Tienda (🏪)")

        tipo = df.groupby('store_type')['sales'].sum().reset_index().sort_values('sales')
        fig = px.bar(tipo, x='store_type', y='sales', color='store_type', color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False)

        st.plotly_chart(fig, use_container_width=True)
    

    # treemap grande de estados, con selector de año
    st.subheader("Ventas por Estado (📅)")

    año_sel = st.selectbox("Selecciona año:", sorted(df['year'].unique()), key='año_treemap')
    treemap = df[df['year'] == año_sel].groupby('state')['sales'].sum().reset_index()
    fig = px.treemap(treemap, path=['state'], values='sales', color='sales', color_continuous_scale='RdYlGn')
    
    st.plotly_chart(fig, use_container_width=True)