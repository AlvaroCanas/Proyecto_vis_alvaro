import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gzip
import shutil
import os

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Ventas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la estética
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 24px;
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Carga y combina los datos de los archivos comprimidos"""
    try:
        # Intentar leer directamente los archivos .gz
        df1 = pd.read_csv('parte_1.csv.gz', compression='gzip')
        df2 = pd.read_csv('parte_2.csv.gz', compression='gzip')
        
        # Combinar los dataframes
        df = pd.concat([df1, df2], ignore_index=True)
        
        # Convertir la columna date a datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Asegurar que day_of_week sea string
        if 'day_of_week' in df.columns:
            df['day_of_week'] = df['day_of_week'].astype(str)
        
        return df
    
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        st.info("Verifica que los archivos parte_1.csv.gz y parte_2.csv.gz estén en el directorio correcto")
        return None

# Cargar datos
with st.spinner('Cargando datos...'):
    df = load_data()

if df is None:
    st.stop()

# Título principal
st.title("📊 Dashboard de Ventas - Análisis Anual")
st.markdown("---")

# Crear pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Visión Global", 
    "🏪 Análisis por Tienda", 
    "📍 Análisis por Estado", 
    "💡 Insights Avanzados"
])

# ==================== PESTAÑA 1: VISIÓN GLOBAL ====================
with tab1:
    st.header("Visión Global de Ventas")
    
    # Conteo general
    st.subheader("📈 Métricas Generales")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_stores = df['store_nbr'].nunique()
        st.metric("Total de Tiendas", f"{total_stores:,}")
    
    with col2:
        total_products = df['family'].nunique()
        st.metric("Productos Vendidos", f"{total_products:,}")
    
    with col3:
        total_states = df['state'].nunique()
        st.metric("Estados", f"{total_states:,}")
    
    with col4:
        total_months = df['month'].nunique()
        st.metric("Meses de Datos", f"{total_months:,}")
    
    st.markdown("---")
    
    # Análisis en términos medios
    st.subheader("📊 Análisis de Ventas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 productos más vendidos
        st.markdown("**Top 10 Productos Más Vendidos**")
        top_products = df.groupby('family')['sales'].sum().sort_values(ascending=False).head(10)
        fig_top_products = px.bar(
            x=top_products.values,
            y=top_products.index,
            orientation='h',
            labels={'x': 'Ventas Totales', 'y': 'Producto'},
            color=top_products.values,
            color_continuous_scale='Blues'
        )
        fig_top_products.update_layout(
            showlegend=False,
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_top_products, use_container_width=True)
    
    with col2:
        # Distribución de ventas por tiendas
        st.markdown("**Distribución de Ventas por Tiendas**")
        sales_by_store = df.groupby('store_nbr')['sales'].sum().sort_values(ascending=False)
        fig_store_dist = px.histogram(
            x=sales_by_store.values,
            nbins=30,
            labels={'x': 'Ventas Totales', 'y': 'Número de Tiendas'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_store_dist.update_layout(
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_store_dist, use_container_width=True)
    
    # Top 10 tiendas con ventas en promoción
    st.markdown("**Top 10 Tiendas con Más Ventas en Promoción**")
    promo_sales = df[df['onpromotion'] > 0].groupby('store_nbr')['sales'].sum().sort_values(ascending=False).head(10)
    fig_promo = px.bar(
        x=promo_sales.index.astype(str),
        y=promo_sales.values,
        labels={'x': 'Tienda', 'y': 'Ventas en Promoción'},
        color=promo_sales.values,
        color_continuous_scale='Greens'
    )
    fig_promo.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_promo, use_container_width=True)
    
    st.markdown("---")
    
    # Análisis de estacionalidad
    st.subheader("📅 Análisis de Estacionalidad")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Ventas por día de la semana
        st.markdown("**Ventas por Día de la Semana**")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        sales_by_day = df.groupby('day_of_week')['sales'].mean().reindex(day_order)
        fig_day = px.bar(
            x=sales_by_day.index,
            y=sales_by_day.values,
            labels={'x': 'Día', 'y': 'Ventas Promedio'},
            color=sales_by_day.values,
            color_continuous_scale='Oranges'
        )
        fig_day.update_layout(showlegend=False, height=350)
        fig_day.update_xaxis(tickangle=-45)
        st.plotly_chart(fig_day, use_container_width=True)
    
    with col2:
        # Ventas por semana del año
        st.markdown("**Ventas por Semana del Año**")
        sales_by_week = df.groupby('week')['sales'].mean().sort_index()
        fig_week = px.line(
            x=sales_by_week.index,
            y=sales_by_week.values,
            labels={'x': 'Semana del Año', 'y': 'Ventas Promedio'},
            markers=True
        )
        fig_week.update_traces(line_color='#ff7f0e', marker=dict(size=4))
        fig_week.update_layout(height=350)
        st.plotly_chart(fig_week, use_container_width=True)
    
    with col3:
        # Ventas por mes
        st.markdown("**Ventas por Mes**")
        month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                       'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        sales_by_month = df.groupby('month')['sales'].mean().sort_index()
        fig_month = px.bar(
            x=[month_names[i-1] for i in sales_by_month.index],
            y=sales_by_month.values,
            labels={'x': 'Mes', 'y': 'Ventas Promedio'},
            color=sales_by_month.values,
            color_continuous_scale='Purples'
        )
        fig_month.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_month, use_container_width=True)

# ==================== PESTAÑA 2: ANÁLISIS POR TIENDA ====================
with tab2:
    st.header("Análisis por Tienda")
    
    # Selector de tienda
    stores = sorted(df['store_nbr'].unique())
    selected_store = st.selectbox("Selecciona una tienda:", stores, key='store_selector')
    
    # Filtrar datos por tienda
    store_data = df[df['store_nbr'] == selected_store]
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Ventas totales por año
        st.markdown("**Ventas Totales por Año**")
        sales_by_year = store_data.groupby('year')['sales'].sum().sort_index()
        fig_year = px.bar(
            x=sales_by_year.index.astype(str),
            y=sales_by_year.values,
            labels={'x': 'Año', 'y': 'Ventas Totales'},
            color=sales_by_year.values,
            color_continuous_scale='Blues'
        )
        fig_year.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_year, use_container_width=True)
        
        total_sales = store_data['sales'].sum()
        st.metric("Ventas Totales", f"${total_sales:,.2f}")
    
    with col2:
        # Productos vendidos
        st.markdown("**Total de Productos Vendidos**")
        products_sold = store_data.groupby('family')['sales'].sum().sort_values(ascending=False)
        fig_products = px.pie(
            values=products_sold.values[:10],
            names=products_sold.index[:10],
            hole=0.4
        )
        fig_products.update_layout(height=400)
        st.plotly_chart(fig_products, use_container_width=True)
        
        total_products = len(products_sold)
        st.metric("Categorías de Productos", f"{total_products:,}")
    
    with col3:
        # Productos en promoción
        st.markdown("**Productos Vendidos en Promoción**")
        promo_data = store_data[store_data['onpromotion'] > 0]
        promo_by_family = promo_data.groupby('family')['sales'].sum().sort_values(ascending=False).head(10)
        fig_promo = px.bar(
            y=promo_by_family.index,
            x=promo_by_family.values,
            orientation='h',
            labels={'x': 'Ventas', 'y': 'Producto'},
            color=promo_by_family.values,
            color_continuous_scale='Greens'
        )
        fig_promo.update_layout(
            showlegend=False,
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_promo, use_container_width=True)
        
        total_promo_sales = promo_data['sales'].sum()
        st.metric("Ventas en Promoción", f"${total_promo_sales:,.2f}")

# ==================== PESTAÑA 3: ANÁLISIS POR ESTADO ====================
with tab3:
    st.header("Análisis por Estado")
    
    # Selector de estado
    states = sorted(df['state'].unique())
    selected_state = st.selectbox("Selecciona un estado:", states, key='state_selector')
    
    # Filtrar datos por estado
    state_data = df[df['state'] == selected_state]
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Transacciones por año
        st.markdown("**Transacciones Totales por Año**")
        trans_by_year = state_data.groupby('year')['transactions'].sum().sort_index()
        fig_trans = px.area(
            x=trans_by_year.index.astype(str),
            y=trans_by_year.values,
            labels={'x': 'Año', 'y': 'Transacciones'},
            color_discrete_sequence=['#2ca02c']
        )
        fig_trans.update_layout(height=400)
        st.plotly_chart(fig_trans, use_container_width=True)
        
        total_trans = state_data['transactions'].sum()
        st.metric("Transacciones Totales", f"{total_trans:,.0f}")
    
    with col2:
        # Ranking de tiendas
        st.markdown("**Ranking de Tiendas con Más Ventas**")
        store_ranking = state_data.groupby('store_nbr')['sales'].sum().sort_values(ascending=False).head(10)
        fig_ranking = px.bar(
            x=store_ranking.index.astype(str),
            y=store_ranking.values,
            labels={'x': 'Tienda', 'y': 'Ventas Totales'},
            color=store_ranking.values,
            color_continuous_scale='Reds'
        )
        fig_ranking.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_ranking, use_container_width=True)
    
    # Producto más vendido por tienda
    st.markdown("**Producto Más Vendido por Tienda en el Estado**")
    top_product_per_store = state_data.groupby(['store_nbr', 'family'])['sales'].sum().reset_index()
    top_product_per_store = top_product_per_store.loc[
        top_product_per_store.groupby('store_nbr')['sales'].idxmax()
    ]
    
    fig_top_prod = px.bar(
        top_product_per_store,
        x='store_nbr',
        y='sales',
        color='family',
        labels={'store_nbr': 'Tienda', 'sales': 'Ventas', 'family': 'Producto'},
        hover_data=['family']
    )
    fig_top_prod.update_layout(height=400)
    st.plotly_chart(fig_top_prod, use_container_width=True)

# ==================== PESTAÑA 4: INSIGHTS AVANZADOS ====================
with tab4:
    st.header("💡 Insights Avanzados y Análisis Predictivo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Análisis de correlación entre promociones y ventas
        st.markdown("**Impacto de Promociones en Ventas**")
        promo_impact = df.groupby('onpromotion')['sales'].mean()
        fig_promo_impact = px.bar(
            x=['Sin Promoción', 'Con Promoción'],
            y=promo_impact.values,
            labels={'x': 'Estado', 'y': 'Venta Promedio'},
            color=promo_impact.values,
            color_continuous_scale='Teal'
        )
        fig_promo_impact.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_promo_impact, use_container_width=True)
        
        increase = ((promo_impact[1] - promo_impact[0]) / promo_impact[0] * 100)
        st.metric("Incremento con Promoción", f"+{increase:.1f}%")
    
    with col2:
        # Ventas por tipo de tienda
        st.markdown("**Ventas por Tipo de Tienda**")
        sales_by_type = df.groupby('store_type')['sales'].sum()
        fig_type = px.pie(
            values=sales_by_type.values,
            names=sales_by_type.index,
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_type.update_layout(height=350)
        st.plotly_chart(fig_type, use_container_width=True)
    
    # Heatmap de ventas por mes y año
    st.markdown("**Mapa de Calor: Ventas por Mes y Año**")
    heatmap_data = df.groupby(['year', 'month'])['sales'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='month', columns='year', values='sales')
    
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Año", y="Mes", color="Ventas"),
        x=heatmap_pivot.columns,
        y=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'][:len(heatmap_pivot)],
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Análisis de clusters
    st.markdown("**Rendimiento por Cluster de Tiendas**")
    cluster_performance = df.groupby('cluster').agg({
        'sales': 'sum',
        'transactions': 'sum',
        'store_nbr': 'nunique'
    }).reset_index()
    
    fig_cluster = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Ventas por Cluster', 'Transacciones por Cluster')
    )
    
    fig_cluster.add_trace(
        go.Bar(x=cluster_performance['cluster'], y=cluster_performance['sales'], 
               name='Ventas', marker_color='indianred'),
        row=1, col=1
    )
    
    fig_cluster.add_trace(
        go.Bar(x=cluster_performance['cluster'], y=cluster_performance['transactions'], 
               name='Transacciones', marker_color='lightsalmon'),
        row=1, col=2
    )
    
    fig_cluster.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Top insights
    st.markdown("---")
    st.subheader("🎯 Conclusiones Clave")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_product = df.groupby('family')['sales'].sum().idxmax()
        st.success(f"**Producto Estrella:** {best_product}")
    
    with col2:
        best_store = df.groupby('store_nbr')['sales'].sum().idxmax()
        st.info(f"**Mejor Tienda:** #{best_store}")
    
    with col3:
        best_month = df.groupby('month')['sales'].sum().idxmax()
        month_names = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio',
                      7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
        st.warning(f"**Mejor Mes:** {month_names.get(best_month, best_month)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p>Dashboard de Ventas | Desarrollado con Streamlit y Plotly</p>
    </div>
    """,
    unsafe_allow_html=True
)