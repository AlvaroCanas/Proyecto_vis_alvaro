import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Dashboard Ventas | Empresa Alimentación",
    page_icon="📈",
    layout="wide",
)

# Un poco de CSS para “look & feel”
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; }
      [data-testid="stMetricValue"] { font-size: 1.55rem; }
      [data-testid="stMetricLabel"] { opacity: 0.85; }
      .stTabs [data-baseweb="tab"] { font-size: 1rem; padding: 0.6rem 1rem; }
      .stTabs [aria-selected="true"] { font-weight: 700; }
      .small-note { opacity: .75; font-size: .9rem; }
      .kpi-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
                  padding: 12px 14px; border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Data loading
# -----------------------------
DATA_DIR = Path("data")
FILES = [DATA_DIR / "parte_1.csv.gz", DATA_DIR / "parte_2.csv.gz"]

REQUIRED_COLS = [
    "id", "date", "store_nbr", "family", "sales", "onpromotion",
    "holiday_type", "locale", "locale_name", "description", "transferred",
    "dcoilwtico", "city", "state", "store_type", "cluster", "transactions",
    "year", "month", "week", "quarter", "day_of_week"
]


@st.cache_data(show_spinner=False)
def load_and_prepare(files: list[Path]) -> pd.DataFrame:
    dfs = []
    missing = []
    for fp in files:
        if not fp.exists():
            missing.append(str(fp))
            continue
        df = pd.read_csv(fp, compression="gzip", low_memory=False)
        dfs.append(df)

    if missing:
        raise FileNotFoundError(
            "No encuentro estos ficheros en /data:\n- " + "\n- ".join(missing)
        )
    if not dfs:
        raise FileNotFoundError("No se han podido cargar datasets.")

    df = pd.concat(dfs, ignore_index=True)

    # Normalizaciones y robustez
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Asegurar columnas clave (sin reventar si alguna falta)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Tipos y limpieza básica
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce")
    df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").fillna(0)

    # Si faltan year/month/week/day_of_week, los recalculamos desde date
    if df["date"].notna().any():
        if df["year"].isna().all():
            df["year"] = df["date"].dt.year
        if df["month"].isna().all():
            df["month"] = df["date"].dt.month
        if df["quarter"].isna().all():
            df["quarter"] = df["date"].dt.quarter
        if df["week"].isna().all():
            # ISO week
            df["week"] = df["date"].dt.isocalendar().week.astype("Int64")
        if df["day_of_week"].isna().all():
            df["day_of_week"] = df["date"].dt.day_name()

    # Etiquetas útiles
    df["is_promo"] = df["onpromotion"] > 0
    df["has_holiday"] = df["holiday_type"].fillna("").astype(str).str.lower().ne("none") & df["holiday_type"].notna()

    # Orden de días (para gráficos)
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=dow_order, ordered=True)

    return df


def fmt_int(x) -> str:
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return "—"


def fmt_float(x) -> str:
    try:
        return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "—"


def top_n_table(df: pd.DataFrame, group_col: str, value_col: str, n: int = 10, title: str = ""):
    t = (
        df.groupby(group_col, dropna=False)[value_col]
        .sum(min_count=1)
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    if title:
        st.subheader(title)
    st.dataframe(
        t,
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("⚙️ Filtros globales")

    df = load_and_prepare(FILES)

    # Rango temporal
    min_date = df["date"].min()
    max_date = df["date"].max()

    if pd.isna(min_date) or pd.isna(max_date):
        st.warning("No hay fechas válidas en la columna `date`.")
        date_range = None
    else:
        date_range = st.date_input(
            "Rango de fechas",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

    years_available = sorted([y for y in df["year"].dropna().unique().tolist() if str(y) != "nan"])
    if years_available:
        years_sel = st.multiselect("Años", years_available, default=years_available)
    else:
        years_sel = []

    states_available = sorted(df["state"].dropna().astype(str).unique().tolist())
    states_sel = st.multiselect("Estados", states_available, default=states_available[: min(6, len(states_available))])

    family_available = sorted(df["family"].dropna().astype(str).unique().tolist())
    family_sel = st.multiselect("Familias (productos)", family_available, default=[])

    st.markdown("---")
    st.caption("Tip: deja Familias vacío para ver todo el catálogo.")

# Aplicar filtros globales
df_f = df.copy()

if date_range and len(date_range) == 2:
    d0 = pd.to_datetime(date_range[0])
    d1 = pd.to_datetime(date_range[1])
    df_f = df_f[(df_f["date"] >= d0) & (df_f["date"] <= d1)]

if years_sel:
    df_f = df_f[df_f["year"].isin(years_sel)]

if states_sel:
    df_f = df_f[df_f["state"].astype(str).isin(states_sel)]

if family_sel:
    df_f = df_f[df_f["family"].astype(str).isin(family_sel)]


# -----------------------------
# Header
# -----------------------------
st.title("📊 Dashboard de Ventas — Cierre de Año")
st.markdown(
    f"<div class='small-note'>Datos combinados: <b>parte_1</b> + <b>parte_2</b> · Registros: <b>{fmt_int(len(df_f))}</b></div>",
    unsafe_allow_html=True,
)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["1) Visión global", "2) Por tienda", "3) Por estado", "4) Insights (extra)"]
)

# -----------------------------
# TAB 1: Global
# -----------------------------
with tab1:
    st.header("📌 Visión global del periodo")

    # KPIs solicitados
    col1, col2, col3, col4 = st.columns(4)

    total_stores = df_f["store_nbr"].nunique(dropna=True)
    total_products = df_f["family"].nunique(dropna=True)
    total_states = df_f["state"].nunique(dropna=True)

    # Meses disponibles (conteo de meses únicos del dataset filtrado)
    months_count = df_f[["year", "month"]].dropna().drop_duplicates().shape[0]

    col1.metric("🏬 Nº total de tiendas", fmt_int(total_stores))
    col2.metric("🧺 Nº total de productos", fmt_int(total_products))
    col3.metric("🗺️ Estados (state)", fmt_int(total_states))
    col4.metric("🗓️ Meses con datos", fmt_int(months_count))

    st.markdown("---")

    # b) Análisis en términos medios
    cA, cB = st.columns((1.05, 0.95), gap="large")

    with cA:
        st.subheader("🏆 Top 10 productos más vendidos (ventas totales)")
        prod_sales = (
            df_f.groupby("family", dropna=False)["sales"]
            .sum(min_count=1)
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig = px.bar(prod_sales, x="sales", y="family", orientation="h")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with cB:
        st.subheader("🏪 Distribución de ventas por tienda")
        store_sales = (
            df_f.groupby("store_nbr", dropna=False)["sales"]
            .sum(min_count=1)
            .reset_index()
            .rename(columns={"sales": "sales_total"})
        )
        # Box plot + puntos para que se entienda bien la distribución
        fig2 = px.box(store_sales, y="sales_total", points="outliers")
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("🔥 Top 10 tiendas con ventas en productos en promoción")
    promo_store = df_f[df_f["is_promo"]].copy()
    promo_by_store = (
        promo_store.groupby("store_nbr", dropna=False)["sales"]
        .sum(min_count=1)
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"sales": "promo_sales"})
    )
    fig3 = px.bar(promo_by_store, x="store_nbr", y="promo_sales")
    fig3.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # c) Estacionalidad
    st.header("📈 Estacionalidad de las ventas")

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.subheader("📅 Día de la semana con más ventas (media)")
        dow = (
            df_f.groupby("day_of_week", dropna=False)["sales"]
            .mean()
            .reset_index()
            .sort_values("day_of_week")
        )
        fig = px.bar(dow, x="day_of_week", y="sales")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("🗓️ Ventas medias por semana (promedio entre años)")
        weekly = (
            df_f.dropna(subset=["year", "week"])
            .groupby(["year", "week"])["sales"]
            .sum(min_count=1)
            .reset_index()
        )
        weekly_mean = (
            weekly.groupby("week")["sales"]
            .mean()
            .reset_index()
            .sort_values("week")
        )
        fig = px.line(weekly_mean, x="week", y="sales", markers=True)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.subheader("📆 Ventas medias por mes (promedio entre años)")
        monthly = (
            df_f.dropna(subset=["year", "month"])
            .groupby(["year", "month"])["sales"]
            .sum(min_count=1)
            .reset_index()
        )
        monthly_mean = (
            monthly.groupby("month")["sales"]
            .mean()
            .reset_index()
            .sort_values("month")
        )
        fig = px.line(monthly_mean, x="month", y="sales", markers=True)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Nota: `sales` se interpreta como volumen vendido (agregado). Si tu práctica define otra interpretación, dímelo y lo adapto.")


# -----------------------------
# TAB 2: Por tienda
# -----------------------------
with tab2:
    st.header("🏬 Análisis por tienda (store_nbr)")

    stores = sorted(df_f["store_nbr"].dropna().unique().tolist())
    if not stores:
        st.info("No hay tiendas disponibles con los filtros actuales.")
    else:
        left, right = st.columns([0.35, 0.65], gap="large")

        with left:
            store_sel = st.selectbox("Selecciona una tienda", stores, index=0)
            df_s = df_f[df_f["store_nbr"] == store_sel].copy()

            total_sales_store = df_s["sales"].sum(skipna=True)
            total_products_sold = df_s["sales"].sum(skipna=True)  # mismo KPI, pero lo mostramos como “productos” en términos de volumen
            promo_products_sold = df_s.loc[df_s["is_promo"], "sales"].sum(skipna=True)

            st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
            st.metric("📦 Ventas totales (volumen)", fmt_float(total_sales_store))
            st.metric("🧺 Productos vendidos (volumen)", fmt_float(total_products_sold))
            st.metric("🏷️ Productos vendidos en promo (volumen)", fmt_float(promo_products_sold))
            st.markdown("</div>", unsafe_allow_html=True)

            # extra “útil”
            st.markdown("")
            st.subheader("🧩 Mix de producto (Top 8)")
            top_fam = (
                df_s.groupby("family")["sales"].sum(min_count=1).sort_values(ascending=False).head(8).reset_index()
            )
            fig = px.pie(top_fam, names="family", values="sales", hole=0.45)
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("📅 Ventas por año (de más antiguo a más reciente)")
            by_year = (
                df_s.groupby("year")["sales"].sum(min_count=1).reset_index().sort_values("year")
            )
            fig = px.bar(by_year, x="year", y="sales")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📈 Serie temporal mensual (ventas)")
            by_month = (
                df_s.dropna(subset=["date"])
                .set_index("date")["sales"]
                .resample("MS")
                .sum(min_count=1)
                .reset_index()
                .rename(columns={"sales": "sales_month"})
            )
            fig2 = px.line(by_month, x="date", y="sales_month", markers=True)
            fig2.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("🏷️ Promoción vs No promoción (ventas)")
            promo_cmp = (
                df_s.assign(promo=lambda x: np.where(x["is_promo"], "Promo", "No promo"))
                .groupby("promo")["sales"]
                .sum(min_count=1)
                .reset_index()
            )
            fig3 = px.bar(promo_cmp, x="promo", y="sales")
            fig3.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig3, use_container_width=True)


# -----------------------------
# TAB 3: Por estado
# -----------------------------
with tab3:
    st.header("🗺️ Análisis por estado (state)")

    states = sorted(df_f["state"].dropna().astype(str).unique().tolist())
    if not states:
        st.info("No hay estados disponibles con los filtros actuales.")
    else:
        cL, cR = st.columns([0.35, 0.65], gap="large")

        with cL:
            state_sel = st.selectbox("Selecciona un estado", states, index=0)
            df_st = df_f[df_f["state"].astype(str) == state_sel].copy()

            st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
            st.metric("🏬 Tiendas en el estado", fmt_int(df_st["store_nbr"].nunique()))
            st.metric("🧺 Familias vendidas", fmt_int(df_st["family"].nunique()))
            st.metric("📦 Ventas totales (volumen)", fmt_float(df_st["sales"].sum(skipna=True)))
            st.metric("🧾 Transacciones totales", fmt_float(df_st["transactions"].sum(skipna=True)))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("")
            st.subheader("🥇 Producto líder del estado")
            top_prod_state = (
                df_st.groupby("family")["sales"].sum(min_count=1).sort_values(ascending=False).head(1)
            )
            if len(top_prod_state) == 0:
                st.write("—")
            else:
                fam_name = top_prod_state.index[0]
                fam_sales = top_prod_state.iloc[0]
                st.success(f"**{fam_name}** · ventas: **{fmt_float(fam_sales)}**")

        with cR:
            st.subheader("📆 Nº total de transacciones por año")
            tx_year = (
                df_st.groupby("year")["transactions"]
                .sum(min_count=1)
                .reset_index()
                .sort_values("year")
            )
            fig = px.bar(tx_year, x="year", y="transactions")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🏆 Ranking de tiendas con más ventas (Top 10)")
            rank_store = (
                df_st.groupby("store_nbr")["sales"]
                .sum(min_count=1)
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            fig2 = px.bar(rank_store, x="store_nbr", y="sales")
            fig2.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("🧺 Producto más vendido por tienda (en este estado)")
            # Para cada tienda, el producto (family) con mayor venta
            by_store_family = (
                df_st.groupby(["store_nbr", "family"])["sales"]
                .sum(min_count=1)
                .reset_index()
            )
            idx = by_store_family.groupby("store_nbr")["sales"].idxmax()
            best_by_store = by_store_family.loc[idx].sort_values("sales", ascending=False)

            # Mostramos top 15 para que sea legible
            st.dataframe(
                best_by_store.head(15).rename(columns={"family": "top_family", "sales": "top_family_sales"}),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Tabla: para cada tienda del estado, su familia más vendida.")


# -----------------------------
# TAB 4: Extra (para sorprender)
# -----------------------------
with tab4:
    st.header("✨ Insights extra para decisiones rápidas")

    st.markdown(
        "Esta pestaña añade análisis que suele acelerar conclusiones en comité: **efecto de promociones** y **impacto de festivos**."
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader("🏷️ Eficiencia de promoción: ventas medias con vs sin promo")
        promo_effect = (
            df_f.assign(promo=lambda x: np.where(x["is_promo"], "Promo", "No promo"))
            .groupby("promo")["sales"]
            .mean()
            .reset_index()
        )
        fig = px.bar(promo_effect, x="promo", y="sales")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Uplift “simple”
        if set(promo_effect["promo"]) == {"Promo", "No promo"}:
            s_p = float(promo_effect.loc[promo_effect["promo"] == "Promo", "sales"].iloc[0])
            s_n = float(promo_effect.loc[promo_effect["promo"] == "No promo", "sales"].iloc[0])
            uplift = (s_p / s_n - 1.0) * 100 if s_n not in (0, np.nan) else np.nan
            st.info(f"Uplift aproximado de ventas medias en promo: **{fmt_float(uplift)}%**")

    with c2:
        st.subheader("🎉 Festivos vs no festivos (ventas medias)")
        holiday_cmp = (
            df_f.assign(holiday=lambda x: np.where(x["has_holiday"], "Festivo", "No festivo"))
            .groupby("holiday")["sales"]
            .mean()
            .reset_index()
        )
        fig = px.bar(holiday_cmp, x="holiday", y="sales")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Si existe dcoilwtico, ver relación simple
        if df_f["dcoilwtico"].notna().any():
            st.subheader("🛢️ Relación precio petróleo vs ventas (muestra)")
            sample = df_f.dropna(subset=["dcoilwtico", "sales"]).sample(
                n=min(5000, df_f.dropna(subset=["dcoilwtico", "sales"]).shape[0]),
                random_state=42
            )
            fig2 = px.scatter(sample, x="dcoilwtico", y="sales", trendline="ols")
            fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📌 Resumen ejecutivo (auto)")
    # Un resumen breve basado en los filtros actuales
    top_store = (
        df_f.groupby("store_nbr")["sales"].sum(min_count=1).sort_values(ascending=False).head(1)
    )
    top_family = (
        df_f.groupby("family")["sales"].sum(min_count=1).sort_values(ascending=False).head(1)
    )
    msg = []
    if len(top_store) > 0:
        msg.append(f"• La tienda con mayor volumen vendido es **{top_store.index[0]}**.")
    if len(top_family) > 0:
        msg.append(f"• El producto/familia líder es **{top_family.index[0]}**.")
    if df_f["is_promo"].any():
        promo_share = 100 * df_f.loc[df_f["is_promo"], "sales"].sum(skipna=True) / max(df_f["sales"].sum(skipna=True), 1e-9)
        msg.append(f"• Aproximadamente **{fmt_float(promo_share)}%** del volumen vendido ocurre con promoción.")
    st.write("\n".join(msg) if msg else "No hay suficiente información con los filtros actuales.")


# Footer
st.markdown("---")
st.caption("Dashboard en Streamlit · Diseñado para lectura rápida de KPIs + exploración interactiva.")
