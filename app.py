import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Dashboard Ventas", page_icon="📊", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FILE_1 = DATA_DIR / "parte_1.csv.gz"
FILE_2 = DATA_DIR / "parte_2.csv.gz"
FILES = [FILE_1, FILE_2]


# ----------------------------
# Robust file waiting + reading
# ----------------------------
def wait_for_files(paths: list[Path], timeout_s: float = 35.0, poll_s: float = 0.5) -> None:
    """
    En Streamlit Cloud a veces el repo/data tarda unos segundos en estar accesible (cold start).
    Espera hasta que existan los ficheros o lanza un error con info de diagnóstico.
    """
    t0 = time.time()
    while True:
        missing = [p for p in paths if not p.exists()]
        if not missing:
            return
        if time.time() - t0 > timeout_s:
            # Diagnóstico útil
            data_ls = []
            try:
                if DATA_DIR.exists():
                    data_ls = sorted([x.name for x in DATA_DIR.iterdir()])
            except Exception:
                pass

            raise FileNotFoundError(
                "No han aparecido los ficheros a tiempo.\n"
                f"Missing: {[m.as_posix() for m in missing]}\n"
                f"DATA_DIR exists: {DATA_DIR.exists()} ({DATA_DIR.as_posix()})\n"
                f"Contenido de data/: {data_ls}\n"
                f"BASE_DIR: {BASE_DIR.as_posix()}"
            )
        time.sleep(poll_s)


def read_csv_gz(path: Path) -> pd.DataFrame:
    # low_memory=False como pediste
    return pd.read_csv(path, compression="gzip", low_memory=False)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    # Espera a que el filesystem esté listo
    wait_for_files(FILES)

    df1 = read_csv_gz(FILE_1)
    df2 = read_csv_gz(FILE_2)

    # Concatenación (mismas columnas)
    df = pd.concat([df1, df2], ignore_index=True)

    # Tipos
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Derivadas por si acaso
    if "date" in df.columns and df["date"].notna().any():
        if "year" not in df.columns:
            df["year"] = df["date"].dt.year
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month
        if "week" not in df.columns:
            df["week"] = df["date"].dt.isocalendar().week.astype("Int64")
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["date"].dt.day_name()

    for col in ["sales", "transactions", "onpromotion", "dcoilwtico"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ----------------------------
# Plotly defaults (bonito)
# ----------------------------
px.defaults.template = "plotly_white"
px.defaults.width = None
px.defaults.height = 420


def metric_card(col, title, value):
    col.metric(title, value)


def nice_bar(df, x, y, title, color=None):
    fig = px.bar(df, x=x, y=y, title=title, color=color, text_auto=".2s")
    fig.update_layout(title_x=0.0, margin=dict(l=20, r=20, t=60, b=20))
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def nice_line(df, x, y, title):
    fig = px.line(df, x=x, y=y, title=title, markers=True)
    fig.update_layout(title_x=0.0, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def nice_hist(series, title, nbins=40):
    fig = px.histogram(series, nbins=nbins, title=title)
    fig.update_layout(title_x=0.0, margin=dict(l=20, r=20, t=60, b=20))
    return fig


# ----------------------------
# App
# ----------------------------
st.title("📊 Dashboard de Ventas")

with st.spinner("Cargando datos..."):
    try:
        df = load_data()
    except Exception as e:
        st.error("La app ha fallado al cargar datos. Te dejo diagnóstico para arreglarlo en 1 golpe:")
        st.exception(e)
        st.stop()

# Validación suave
required_cols = {"store_nbr", "family", "sales", "onpromotion", "state", "transactions", "year", "month", "week", "day_of_week"}
missing = sorted(list(required_cols - set(df.columns)))
if missing:
    st.warning(f"Faltan columnas esperadas: {missing}. Algunas gráficas no se mostrarán.")

tab1, tab2, tab3, tab4 = st.tabs(["1) Global", "2) Por tienda", "3) Por estado", "4) Extra"])


# ----------------------------
# TAB 1 - Global
# ----------------------------
with tab1:
    st.subheader("📌 Visión global")

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "🏬 Nº total de tiendas", int(df["store_nbr"].nunique()) if "store_nbr" in df.columns else 0)
    metric_card(c2, "🧺 Nº total de productos (families)", int(df["family"].nunique()) if "family" in df.columns else 0)
    metric_card(c3, "🗺️ Nº de estados", int(df["state"].nunique()) if "state" in df.columns else 0)

    if "year" in df.columns and "month" in df.columns:
        meses = df[["year", "month"]].dropna().drop_duplicates()
        metric_card(c4, "🗓️ Meses con datos", int(len(meses)))
    else:
        metric_card(c4, "🗓️ Meses con datos", 0)

    st.divider()
    left, right = st.columns(2)

    # Top 10 productos
    if {"family", "sales"}.issubset(df.columns):
        top_products = (
            df.groupby("family", dropna=False)["sales"].sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig = nice_bar(top_products, x="sales", y="family", title="Top 10 productos más vendidos (ventas totales)")
        fig.update_layout(yaxis_title="", xaxis_title="Ventas")
        left.plotly_chart(fig, width="stretch")
    else:
        left.info("No puedo calcular Top productos: faltan columnas.")

    # Distribución por tienda
    if {"store_nbr", "sales"}.issubset(df.columns):
        store_sales = df.groupby("store_nbr")["sales"].sum()
        fig = nice_hist(store_sales.dropna(), title="Distribución de ventas totales por tienda", nbins=45)
        fig.update_layout(xaxis_title="Ventas totales por tienda", yaxis_title="Frecuencia")
        right.plotly_chart(fig, width="stretch")
    else:
        right.info("No puedo calcular distribución por tienda: faltan columnas.")

    st.divider()
    st.subheader("🎯 Promociones")

    if {"store_nbr", "sales", "onpromotion"}.issubset(df.columns):
        promo_df = df[df["onpromotion"].fillna(0) > 0]
        promo_rank = (
            promo_df.groupby("store_nbr")["sales"].sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="sales_promo")
        )
        fig = nice_bar(promo_rank, x="store_nbr", y="sales_promo", title="Top 10 tiendas por ventas en promoción")
        fig.update_layout(xaxis_title="Tienda", yaxis_title="Ventas en promo")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No puedo calcular promociones: faltan columnas.")

    st.divider()
    st.subheader("🌦️ Estacionalidad (ventas medias)")

    r1, r2, r3 = st.columns(3)

    # Día semana
    if {"day_of_week", "sales"}.issubset(df.columns):
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = df.groupby("day_of_week")["sales"].mean().reset_index(name="sales_mean")
        dow["day_of_week"] = pd.Categorical(dow["day_of_week"], categories=order, ordered=True)
        dow = dow.sort_values("day_of_week")
        fig = nice_bar(dow, x="day_of_week", y="sales_mean", title="Ventas medias por día de la semana")
        fig.update_layout(xaxis_title="Día", yaxis_title="Ventas medias")
        r1.plotly_chart(fig, width="stretch")
    else:
        r1.info("Faltan columnas para día de la semana.")

    # Semana
    if {"week", "sales"}.issubset(df.columns):
        wk = df.dropna(subset=["week"]).groupby("week")["sales"].mean().reset_index(name="sales_mean")
        fig = nice_line(wk, x="week", y="sales_mean", title="Ventas medias por semana del año")
        fig.update_layout(xaxis_title="Semana", yaxis_title="Ventas medias")
        r2.plotly_chart(fig, width="stretch")
    else:
        r2.info("Faltan columnas para semana del año.")

    # Mes
    if {"month", "sales"}.issubset(df.columns):
        mo = df.dropna(subset=["month"]).groupby("month")["sales"].mean().reset_index(name="sales_mean")
        fig = nice_line(mo, x="month", y="sales_mean", title="Ventas medias por mes")
        fig.update_layout(xaxis_title="Mes", yaxis_title="Ventas medias")
        r3.plotly_chart(fig, width="stretch")
    else:
        r3.info("Faltan columnas para mes.")


# ----------------------------
# TAB 2 - Por tienda
# ----------------------------
with tab2:
    st.subheader("🏬 Análisis por tienda")

    if "store_nbr" not in df.columns:
        st.error("No existe `store_nbr`.")
        st.stop()

    stores = sorted(df["store_nbr"].dropna().unique().tolist())
    store_sel = st.selectbox("Selecciona tienda", stores, index=0)
    dstore = df[df["store_nbr"] == store_sel].copy()

    a, b, c = st.columns(3)
    a.metric("📦 Ventas totales", float(dstore["sales"].sum(skipna=True)) if "sales" in dstore.columns else 0.0)
    b.metric("🧺 Productos vendidos (unidades)", float(dstore["sales"].sum(skipna=True)) if "sales" in dstore.columns else 0.0)

    if {"sales", "onpromotion"}.issubset(dstore.columns):
        promo_units = dstore.loc[dstore["onpromotion"].fillna(0) > 0, "sales"].sum(skipna=True)
        c.metric("🏷️ Vendidos en promoción", float(promo_units))
    else:
        c.metric("🏷️ Vendidos en promoción", 0.0)

    st.divider()
    left, right = st.columns(2)

    if {"year", "sales"}.issubset(dstore.columns):
        by_year = dstore.groupby("year")["sales"].sum().reset_index(name="sales_total").sort_values("year")
        fig = nice_line(by_year, x="year", y="sales_total", title="Ventas totales por año (tienda)")
        fig.update_layout(xaxis_title="Año", yaxis_title="Ventas")
        left.plotly_chart(fig, width="stretch")
    else:
        left.info("No puedo dibujar ventas por año: faltan columnas.")

    if {"family", "sales"}.issubset(dstore.columns):
        top_store_prod = (
            dstore.groupby("family")["sales"].sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig = nice_bar(top_store_prod, x="sales", y="family", title="Top 10 productos de la tienda")
        fig.update_layout(yaxis_title="", xaxis_title="Ventas")
        right.plotly_chart(fig, width="stretch")
    else:
        right.info("No puedo mostrar top productos: faltan columnas.")


# ----------------------------
# TAB 3 - Por estado
# ----------------------------
with tab3:
    st.subheader("🗺️ Análisis por estado")

    if "state" not in df.columns:
        st.error("No existe `state`.")
        st.stop()

    states = sorted(df["state"].dropna().unique().tolist())
    state_sel = st.selectbox("Selecciona estado", states, index=0)
    dstate = df[df["state"] == state_sel].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("🧾 Transacciones totales", float(dstate["transactions"].sum(skipna=True)) if "transactions" in dstate.columns else 0.0)
    col2.metric("🏬 Nº tiendas en el estado", int(dstate["store_nbr"].nunique()) if "store_nbr" in dstate.columns else 0)

    if {"family", "sales"}.issubset(dstate.columns) and len(dstate):
        best_prod = dstate.groupby("family")["sales"].sum().sort_values(ascending=False).head(1)
        col3.metric("🥇 Producto más vendido", str(best_prod.index[0]) if len(best_prod) else "N/A")
    else:
        col3.metric("🥇 Producto más vendido", "N/A")

    st.divider()
    l, r = st.columns(2)

    if {"year", "transactions"}.issubset(dstate.columns):
        tx_year = dstate.groupby("year")["transactions"].sum().reset_index(name="transactions_total").sort_values("year")
        fig = nice_line(tx_year, x="year", y="transactions_total", title="Transacciones por año (estado)")
        fig.update_layout(xaxis_title="Año", yaxis_title="Transacciones")
        l.plotly_chart(fig, width="stretch")
    else:
        l.info("No puedo dibujar transacciones por año: faltan columnas.")

    if {"store_nbr", "sales"}.issubset(dstate.columns):
        rank_stores = (
            dstate.groupby("store_nbr")["sales"].sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="sales_total")
        )
        fig = nice_bar(rank_stores, x="store_nbr", y="sales_total", title="Top 10 tiendas por ventas (estado)")
        fig.update_layout(xaxis_title="Tienda", yaxis_title="Ventas")
        r.plotly_chart(fig, width="stretch")
    else:
        r.info("No puedo dibujar ranking de tiendas: faltan columnas.")


# ----------------------------
# TAB 4 - Extra
# ----------------------------
with tab4:
    st.subheader("✨ Extra (sin pie charts)")

    c1, c2 = st.columns(2)

    # Promos vs ventas
    if {"store_nbr", "sales", "onpromotion"}.issubset(df.columns):
        by_store = df.groupby("store_nbr").agg(
            sales_total=("sales", "sum"),
            promo_sales=("sales", lambda s: s[df.loc[s.index, "onpromotion"].fillna(0) > 0].sum()),
        ).reset_index()
        by_store["promo_ratio"] = np.where(by_store["sales_total"] > 0, by_store["promo_sales"] / by_store["sales_total"], 0.0)

        fig = px.scatter(
            by_store,
            x="promo_ratio",
            y="sales_total",
            hover_data=["store_nbr", "promo_sales"],
            title="Dependencia de promociones: ratio promo vs ventas totales",
            trendline="ols",
        )
        fig.update_layout(xaxis_title="Ratio ventas en promo", yaxis_title="Ventas totales", title_x=0.0)
        c1.plotly_chart(fig, width="stretch")
    else:
        c1.info("No puedo calcular promos vs ventas: faltan columnas.")

    # Segmentación por store_type (si existe)
    if {"store_type", "sales"}.issubset(df.columns):
        seg = df.groupby("store_type")["sales"].sum().sort_values(ascending=False).reset_index(name="sales_total")
        fig = nice_bar(seg, x="store_type", y="sales_total", title="Ventas totales por tipo de tienda")
        fig.update_layout(xaxis_title="Tipo de tienda", yaxis_title="Ventas")
        c2.plotly_chart(fig, width="stretch")
    else:
        c2.info("No existe `store_type` o faltan columnas para segmentación.")

    st.divider()
    with st.expander("📌 Diagnóstico cold start"):
        st.write(
            "Si antes te salía 'Oh no' y con reboot iba, casi seguro era porque el script leía `data/*.csv.gz` "
            "antes de que el repo/data estuviera disponible.\n\n"
            "Este app.py lo evita esperando (con timeout) a que existan los ficheros y usando rutas absolutas "
            "basadas en `__file__`."
        )
