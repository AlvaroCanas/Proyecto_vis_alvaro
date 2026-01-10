import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Dashboard Ventas - Práctica Streamlit",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path("data")
FILES = [DATA_DIR / "parte_1.csv.gz", DATA_DIR / "parte_2.csv.gz"]


# ----------------------------
# Robust loading (cold start)
# ----------------------------
def _safe_read_csv_gz(path: Path, retries: int = 8, sleep_s: float = 0.6) -> pd.DataFrame:
    """
    Lectura robusta con reintentos para mitigar arranques fríos en Streamlit Cloud.
    Además fuerza low_memory=False para evitar problemas de inferencia de tipos.
    """
    last_err = None
    for _ in range(retries):
        try:
            if not path.exists():
                raise FileNotFoundError(f"No existe el fichero: {path.as_posix()}")
            return pd.read_csv(path, compression="gzip", low_memory=False)
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise last_err


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    dfs = [_safe_read_csv_gz(p) for p in FILES]
    df = pd.concat(dfs, ignore_index=True)

    # Tipos y columnas derivadas
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

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
# Plot helpers (Matplotlib)
# ----------------------------
def fig_bar(x, y, title, xlabel="", ylabel="", rotate_xticks=False):
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotate_xticks:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def fig_line(x, y, title, xlabel="", ylabel=""):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def fig_hist(values, title, xlabel="", bins=30):
    fig, ax = plt.subplots()
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    return fig


def section_title(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


# ----------------------------
# App
# ----------------------------
st.title("📊 Dashboard de Ventas (Streamlit)")
st.caption(
    "Lee `data/parte_1.csv.gz` y `data/parte_2.csv.gz`, concatena y muestra análisis global, por tienda, por estado y un extra."
)

with st.spinner("Cargando datos..."):
    try:
        df = load_data()
    except Exception as e:
        st.error(
            "No he podido cargar los datos. En Streamlit Cloud a veces el primer acceso coincide con un arranque en frío "
            "y los ficheros tardan unos segundos en estar disponibles.\n\n"
            f"**Error:** {type(e).__name__}: {e}\n\n"
            "Comprueba que existen:\n"
            "- `data/parte_1.csv.gz`\n"
            "- `data/parte_2.csv.gz`"
        )
        st.stop()

# Validación mínima (sin parar la app)
required_cols = {
    "store_nbr", "family", "sales", "onpromotion",
    "state", "transactions", "year", "month", "week", "day_of_week"
}
missing = sorted(list(required_cols - set(df.columns)))
if missing:
    st.warning(f"Faltan columnas esperadas: {missing}. Algunas visualizaciones pueden no mostrarse.")

tab1, tab2, tab3, tab4 = st.tabs(["1) Global", "2) Por tienda", "3) Por estado", "4) Extra"])

# ----------------------------
# TAB 1 - Global
# ----------------------------
with tab1:
    section_title("📌 Visión global", "KPIs y análisis agregados del periodo disponible.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏬 Nº total de tiendas", int(df["store_nbr"].nunique()) if "store_nbr" in df.columns else 0)
    c2.metric("🧺 Nº total de productos (families)", int(df["family"].nunique()) if "family" in df.columns else 0)
    c3.metric("🗺️ Estados", int(df["state"].nunique()) if "state" in df.columns else 0)

    if "year" in df.columns and "month" in df.columns:
        meses = df[["year", "month"]].dropna().drop_duplicates()
        c4.metric("🗓️ Meses con datos", int(len(meses)))
    else:
        c4.metric("🗓️ Meses con datos", 0)

    st.divider()
    colA, colB = st.columns(2)

    # Top 10 productos más vendidos
    if {"family", "sales"}.issubset(df.columns):
        top_products = (
            df.groupby("family")["sales"].sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        colA.pyplot(
            fig_bar(
                x=top_products["family"].astype(str).tolist(),
                y=top_products["sales"].tolist(),
                title="Top 10 productos más vendidos (ventas totales)",
                xlabel="Producto",
                ylabel="Ventas",
                rotate_xticks=True,
            ),
            use_container_width=True
        )
    else:
        colA.info("No puedo calcular Top productos: faltan columnas.")

    # Distribución de ventas totales por tienda (hist)
    if {"store_nbr", "sales"}.issubset(df.columns):
        store_sales = df.groupby("store_nbr")["sales"].sum()
        colB.pyplot(
            fig_hist(
                values=store_sales.dropna().values,
                title="Distribución de ventas totales por tienda",
                xlabel="Ventas totales por tienda",
                bins=35,
            ),
            use_container_width=True
        )
    else:
        colB.info("No puedo calcular distribución por tienda: faltan columnas.")

    st.divider()
    section_title("🎯 Promociones", "Ranking de tiendas con ventas en productos en promoción.")

    if {"store_nbr", "sales", "onpromotion"}.issubset(df.columns):
        promo_df = df[df["onpromotion"].fillna(0) > 0]
        promo_rank = (
            promo_df.groupby("store_nbr")["sales"].sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="sales_promo")
        )
        st.pyplot(
            fig_bar(
                x=promo_rank["store_nbr"].astype(str).tolist(),
                y=promo_rank["sales_promo"].tolist(),
                title="Top 10 tiendas por ventas en promoción",
                xlabel="Tienda",
                ylabel="Ventas en promo",
            ),
            use_container_width=True
        )
    else:
        st.info("No puedo calcular ranking de promociones: faltan columnas.")

    st.divider()
    section_title("🌦️ Estacionalidad", "Ventas medias por día de la semana, semana del año y mes.")
    r1, r2, r3 = st.columns(3)

    # Día semana (media)
    if {"day_of_week", "sales"}.issubset(df.columns):
        dow = df.groupby("day_of_week")["sales"].mean().reset_index(name="sales_mean")
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow["day_of_week"] = pd.Categorical(dow["day_of_week"], categories=order, ordered=True)
        dow = dow.sort_values("day_of_week")
        r1.pyplot(
            fig_bar(
                x=dow["day_of_week"].astype(str).tolist(),
                y=dow["sales_mean"].tolist(),
                title="Ventas medias por día de la semana",
                xlabel="Día",
                ylabel="Ventas medias",
                rotate_xticks=True,
            ),
            use_container_width=True
        )
    else:
        r1.info("Faltan columnas para día de la semana.")

    # Semana (media)
    if {"week", "sales"}.issubset(df.columns):
        wk = df.dropna(subset=["week"]).groupby("week")["sales"].mean().reset_index(name="sales_mean")
        r2.pyplot(
            fig_line(
                x=wk["week"].astype(int).tolist(),
                y=wk["sales_mean"].tolist(),
                title="Ventas medias por semana del año",
                xlabel="Semana",
                ylabel="Ventas medias",
            ),
            use_container_width=True
        )
    else:
        r2.info("Faltan columnas para semana del año.")

    # Mes (media)
    if {"month", "sales"}.issubset(df.columns):
        mo = df.dropna(subset=["month"]).groupby("month")["sales"].mean().reset_index(name="sales_mean")
        r3.pyplot(
            fig_line(
                x=mo["month"].astype(int).tolist(),
                y=mo["sales_mean"].tolist(),
                title="Ventas medias por mes",
                xlabel="Mes",
                ylabel="Ventas medias",
            ),
            use_container_width=True
        )
    else:
        r3.info("Faltan columnas para mes.")

# ----------------------------
# TAB 2 - Por tienda
# ----------------------------
with tab2:
    section_title("🏬 Análisis por tienda", "Selecciona una tienda para ver KPIs y evolución.")

    if "store_nbr" not in df.columns:
        st.error("No existe la columna `store_nbr`.")
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

    # Ventas por año
    if {"year", "sales"}.issubset(dstore.columns):
        by_year = dstore.groupby("year")["sales"].sum().reset_index(name="sales_total").sort_values("year")
        left.pyplot(
            fig_bar(
                x=by_year["year"].astype(int).astype(str).tolist(),
                y=by_year["sales_total"].tolist(),
                title="Ventas totales por año (tienda)",
                xlabel="Año",
                ylabel="Ventas",
            ),
            use_container_width=True
        )
    else:
        left.info("No puedo dibujar ventas por año (faltan columnas).")

    # Top productos tienda
    if {"family", "sales"}.issubset(dstore.columns):
        top_store_prod = (
            dstore.groupby("family")["sales"].sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        right.pyplot(
            fig_bar(
                x=top_store_prod["family"].astype(str).tolist(),
                y=top_store_prod["sales"].tolist(),
                title="Top 10 productos en la tienda",
                xlabel="Producto",
                ylabel="Ventas",
                rotate_xticks=True,
            ),
            use_container_width=True
        )
    else:
        right.info("No puedo mostrar top productos (faltan columnas).")

# ----------------------------
# TAB 3 - Por estado
# ----------------------------
with tab3:
    section_title("🗺️ Análisis por estado", "Transacciones por año, ranking de tiendas y producto líder.")

    if "state" not in df.columns:
        st.error("No existe la columna `state`.")
        st.stop()

    states = sorted(df["state"].dropna().unique().tolist())
    state_sel = st.selectbox("Selecciona estado", states, index=0)
    dstate = df[df["state"] == state_sel].copy()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🧾 Transacciones totales",
        float(dstate["transactions"].sum(skipna=True)) if "transactions" in dstate.columns else 0.0
    )
    col2.metric("🏬 Nº tiendas en el estado", int(dstate["store_nbr"].nunique()) if "store_nbr" in dstate.columns else 0)

    if {"family", "sales"}.issubset(dstate.columns) and len(dstate) > 0:
        best_prod = dstate.groupby("family")["sales"].sum().sort_values(ascending=False).head(1)
        col3.metric("🥇 Producto más vendido", str(best_prod.index[0]) if len(best_prod) else "N/A")
    else:
        col3.metric("🥇 Producto más vendido", "N/A")

    st.divider()
    l, r = st.columns(2)

    # Transacciones por año
    if {"year", "transactions"}.issubset(dstate.columns):
        tx_year = dstate.groupby("year")["transactions"].sum().reset_index(name="transactions_total").sort_values("year")
        l.pyplot(
            fig_line(
                x=tx_year["year"].astype(int).tolist(),
                y=tx_year["transactions_total"].tolist(),
                title="Transacciones por año (estado)",
                xlabel="Año",
                ylabel="Transacciones",
            ),
            use_container_width=True
        )
    else:
        l.info("No puedo dibujar transacciones por año (faltan columnas).")

    # Top 10 tiendas por ventas
    if {"store_nbr", "sales"}.issubset(dstate.columns):
        rank_stores = (
            dstate.groupby("store_nbr")["sales"].sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="sales_total")
        )
        r.pyplot(
            fig_bar(
                x=rank_stores["store_nbr"].astype(str).tolist(),
                y=rank_stores["sales_total"].tolist(),
                title="Top 10 tiendas por ventas (estado)",
                xlabel="Tienda",
                ylabel="Ventas",
            ),
            use_container_width=True
        )
    else:
        r.info("No puedo dibujar ranking de tiendas (faltan columnas).")

# ----------------------------
# TAB 4 - Extra
# ----------------------------
with tab4:
    section_title("✨ Extra", "Promos vs ventas por tienda (y segmentación si existe store_type).")

    c1, c2 = st.columns(2)

    # Scatter promos vs ventas (sin Altair: usamos matplotlib)
    if {"store_nbr", "sales", "onpromotion"}.issubset(df.columns):
        by_store = df.groupby("store_nbr").agg(
            sales_total=("sales", "sum"),
            promo_sales=("sales", lambda s: s[df.loc[s.index, "onpromotion"].fillna(0) > 0].sum()),
        ).reset_index()
        by_store["promo_ratio"] = np.where(by_store["sales_total"] > 0, by_store["promo_sales"] / by_store["sales_total"], 0.0)

        fig, ax = plt.subplots()
        ax.scatter(by_store["promo_ratio"], by_store["sales_total"])
        ax.set_title("Promoción vs ventas totales por tienda")
        ax.set_xlabel("Ratio ventas en promoción")
        ax.set_ylabel("Ventas totales")
        fig.tight_layout()
        c1.pyplot(fig, use_container_width=True)
    else:
        c1.info("No puedo calcular promos vs ventas: faltan columnas.")

    # Segmentación por store_type (si existe)
    if {"store_type", "sales"}.issubset(df.columns):
        seg = df.groupby("store_type")["sales"].sum().sort_values(ascending=False).reset_index(name="sales_total")
        c2.pyplot(
            fig_bar(
                x=seg["store_type"].astype(str).tolist(),
                y=seg["sales_total"].tolist(),
                title="Ventas totales por tipo de tienda",
                xlabel="Tipo de tienda",
                ylabel="Ventas",
                rotate_xticks=True,
            ),
            use_container_width=True
        )
    else:
        c2.info("No existe `store_type` o faltan columnas para segmentación.")

    st.divider()
    with st.expander("📌 Por qué falla al abrir el link y tras reboot funciona"):
        st.write(
            "En Streamlit Cloud puede pasar que el **primer acceso** coincida con un **arranque en frío**: "
            "el contenedor está iniciando, montando el repo o calentando el runtime. "
            "Si tu app intenta leer ficheros justo en ese instante, puede fallar; "
            "tras reboot ya está todo inicializado.\n\n"
            "✅ Aquí lo mitigamos con:\n"
            "- Lectura con **reintentos** (espera unos segundos si el fichero no aparece todavía)\n"
            "- **cache_data** para no recalcular en cada rerun\n"
            "- `low_memory=False` para evitar problemas de tipos al cargar CSV grandes"
        )
