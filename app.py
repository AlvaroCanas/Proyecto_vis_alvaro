from __future__ import annotations

from pathlib import Path
import time
import gzip
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# -----------------------------
# Page config + light styling
# -----------------------------
st.set_page_config(
    page_title="Proyecto VIS | Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
      h1, h2, h3 { letter-spacing: -0.2px; }
      [data-testid="stMetricValue"] { font-size: 1.55rem; }
      .small-note { opacity: .75; font-size: .9rem; }
      .card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.03);
      }
      .stTabs [data-baseweb="tab"] { font-size: 1rem; padding: 0.6rem 1rem; }
      .stTabs [aria-selected="true"] { font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path("data")
FILES = [DATA_DIR / "parte_1.csv.gz", DATA_DIR / "parte_2.csv.gz"]


# -----------------------------
# Helpers
# -----------------------------
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


def safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except Exception:
        return 0


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que existen columnas clave. Si falta alguna, la crea como NaN.
    Así la app no rompe si el dataset trae pequeñas variaciones.
    """
    expected = [
        "id", "date", "store_nbr", "family", "sales", "onpromotion",
        "holiday_type", "locale", "locale_name", "description", "transferred",
        "dcoilwtico", "city", "state", "store_type", "cluster", "transactions",
        "year", "month", "week", "quarter", "day_of_week",
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan
    return df


# Columns to load (reduce memory + avoid cold-start crashes)
USECOLS = set([
    "id", "date", "store_nbr", "family", "sales", "onpromotion",
    "holiday_type", "locale", "locale_name", "description", "transferred",
    "dcoilwtico", "city", "state", "store_type", "cluster", "transactions",
    "year", "month", "week", "quarter", "day_of_week",
])


def _read_gz_csv(path: Path) -> pd.DataFrame:
    """
    Lectura robusta de .csv.gz usando gzip.open (más fiable en Streamlit Cloud).
    Cargamos solo columnas necesarias para reducir RAM y acelerar el arranque.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        df_part = pd.read_csv(
            f,
            low_memory=False,
            usecols=lambda c: c in USECOLS or c == "Unnamed: 0",
        )
    return df_part


@st.cache_data(show_spinner=False)
def load_data(files: list[Path]) -> pd.DataFrame:
    """
    Carga y concatena los dos datasets .csv.gz.
    Incluye reintentos para evitar fallos en cold-start de Streamlit Cloud.
    """
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError("No encuentro estos ficheros en /data:\n- " + "\n- ".join(missing))

    last_err: Exception | None = None

    for attempt in range(3):  # retry
        try:
            dfs = []
            for f in files:
                if f.stat().st_size == 0:
                    raise ValueError(f"El fichero está vacío (0 bytes): {f}")

                df_part = _read_gz_csv(f)
                df_part["__source__"] = f.name
                dfs.append(df_part)

            df = pd.concat(dfs, ignore_index=True)
            df = ensure_cols(df)

            # Tipos
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
            df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce")
            df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").fillna(0)

            # Derivadas desde date si faltan
            if df["date"].notna().any():
                if df["year"].isna().all():
                    df["year"] = df["date"].dt.year
                if df["month"].isna().all():
                    df["month"] = df["date"].dt.month
                if df["quarter"].isna().all():
                    df["quarter"] = df["date"].dt.quarter
                if df["week"].isna().all():
                    df["week"] = df["date"].dt.isocalendar().week.astype("Int64")
                if df["day_of_week"].isna().all():
                    df["day_of_week"] = df["date"].dt.day_name()

            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=dow_order, ordered=True)

            df["is_promo"] = df["onpromotion"] > 0
            df["has_holiday"] = df["holiday_type"].notna() & (df["holiday_type"].astype(str).str.lower() != "none")

            return df

        except Exception as e:
            last_err = e
            time.sleep(1.5)

    raise RuntimeError(f"Error cargando datos tras varios intentos: {last_err}")


# -----------------------------
# Load
# -----------------------------
with st.spinner("Cargando datos (el primer arranque puede tardar un poco)..."):
    df = load_data(FILES)

st.title("📊 Dashboard de Ventas")
st.markdown(
    f"<div class='small-note'>Fuentes: <b>{FILES[0].name}</b> + <b>{FILES[1].name}</b> · "
    f"Filas totales: <b>{fmt_int(len(df))}</b></div>",
    unsafe_allow_html=True,
)

min_date = df["date"].min()
max_date = df["date"].max()


# -----------------------------
# Sidebar filters
# -----------------------------
with st.sidebar:
    st.header("⚙️ Filtros globales")

    if pd.isna(min_date) or pd.isna(max_date):
        st.warning("No hay fechas válidas en `date`. Se desactiva el filtro temporal.")
        date_range = None
    else:
        date_range = st.date_input(
            "Rango de fechas",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

    years = sorted(df["year"].dropna().unique().tolist())
    states = sorted(df["state"].dropna().astype(str).unique().tolist())
    families = sorted(df["family"].dropna().astype(str).unique().tolist())

    years_sel = st.multiselect("Años", years, default=years)
    states_sel = st.multiselect("Estados", states, default=states[: min(8, len(states))])
    fam_sel = st.multiselect("Familias", families, default=[])

    st.caption("Tip: deja Familias vacío para ver todo.")


# Apply filters
df_f = df.copy()

if date_range and len(date_range) == 2:
    d0 = pd.to_datetime(date_range[0])
    d1 = pd.to_datetime(date_range[1])
    df_f = df_f[(df_f["date"] >= d0) & (df_f["date"] <= d1)]

if years_sel:
    df_f = df_f[df_f["year"].isin(years_sel)]

if states_sel:
    df_f = df_f[df_f["state"].astype(str).isin(states_sel)]

if fam_sel:
    df_f = df_f[df_f["family"].astype(str).isin(fam_sel)]


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["1) Visión global", "2) Tiendas", "3) Estados", "4) Estacionalidad"]
)


# -----------------------------
# TAB 1
# -----------------------------
with tab1:
    st.header("📌 Visión global")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🏬 Nº tiendas", fmt_int(safe_nunique(df_f["store_nbr"])))
    k2.metric("🧺 Nº familias", fmt_int(safe_nunique(df_f["family"])))
    k3.metric("🗺️ Nº estados", fmt_int(safe_nunique(df_f["state"])))
    months_count = df_f[["year", "month"]].dropna().drop_duplicates().shape[0]
    k4.metric("🗓️ Meses con datos", fmt_int(months_count))

    st.markdown("---")

    c1, c2 = st.columns((1.15, 0.85), gap="large")

    with c1:
        st.subheader("🏆 Top 10 familias por ventas (total)")
        top_fam = (
            df_f.groupby("family", observed=False)["sales"]
            .sum(min_count=1)
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig = px.bar(top_fam, x="sales", y="family", orientation="h")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.subheader("🏪 Distribución de ventas por tienda")
        store_sales = (
            df_f.groupby("store_nbr", observed=False)["sales"]
            .sum(min_count=1)
            .reset_index()
            .rename(columns={"sales": "sales_total"})
        )
        fig2 = px.box(store_sales, y="sales_total", points="outliers")
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, width="stretch")

    st.markdown("---")

    st.subheader("🔥 Top 10 tiendas con más ventas en promoción (onpromotion > 0)")
    promo, = st.columns(1)
    promo_by_store = (
        df_f[df_f["is_promo"]]
        .groupby("store_nbr", observed=False)["sales"]
        .sum(min_count=1)
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"sales": "promo_sales"})
    )
    fig3 = px.bar(promo_by_store, x="store_nbr", y="promo_sales")
    fig3.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, width="stretch")


# -----------------------------
# TAB 2
# -----------------------------
with tab2:
    st.header("🏬 Análisis por tienda")

    stores = sorted(df_f["store_nbr"].dropna().unique().tolist())
    if not stores:
        st.info("No hay tiendas con los filtros actuales.")
    else:
        left, right = st.columns([0.35, 0.65], gap="large")

        with left:
            store_sel = st.selectbox("Selecciona una tienda", stores, index=0)
            df_s = df_f[df_f["store_nbr"] == store_sel].copy()

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.metric("📦 Ventas totales", fmt_float(df_s["sales"].sum(skipna=True)))
            st.metric("🧾 Transacciones totales", fmt_float(df_s["transactions"].sum(skipna=True)))
            st.metric("🏷️ Ventas en promo", fmt_float(df_s.loc[df_s["is_promo"], "sales"].sum(skipna=True)))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("")
            st.subheader("🧩 Mix de familias (Top 8)")
            mix = (
                df_s.groupby("family", observed=False)["sales"]
                .sum(min_count=1)
                .sort_values(ascending=False)
                .head(8)
                .reset_index()
            )
            fig = px.pie(mix, names="family", values="sales", hole=0.45)
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

        with right:
            st.subheader("📅 Ventas por año")
            by_year = (
                df_s.groupby("year", observed=False)["sales"]
                .sum(min_count=1)
                .reset_index()
                .sort_values("year")
            )
            fig = px.bar(by_year, x="year", y="sales")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

            st.subheader("📈 Serie mensual (ventas)")
            if df_s["date"].notna().any():
                by_month = (
                    df_s.dropna(subset=["date"])
                    .set_index("date")["sales"]
                    .resample("MS")
                    .sum(min_count=1)
                    .reset_index()
                    .rename(columns={"sales": "sales_month"})
                )
                fig2 = px.line(by_month, x="date", y="sales_month", markers=True)
                fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig2, width="stretch")
            else:
                st.info("No hay fechas válidas para construir la serie.")


# -----------------------------
# TAB 3
# -----------------------------
with tab3:
    st.header("🗺️ Análisis por estado")

    states = sorted(df_f["state"].dropna().astype(str).unique().tolist())
    if not states:
        st.info("No hay estados con los filtros actuales.")
    else:
        cL, cR = st.columns([0.35, 0.65], gap="large")

        with cL:
            state_sel = st.selectbox("Selecciona un estado", states, index=0)
            df_st = df_f[df_f["state"].astype(str) == state_sel].copy()

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.metric("🏬 Tiendas en el estado", fmt_int(safe_nunique(df_st["store_nbr"])))
            st.metric("🧺 Familias vendidas", fmt_int(safe_nunique(df_st["family"])))
            st.metric("📦 Ventas totales", fmt_float(df_st["sales"].sum(skipna=True)))
            st.metric("🧾 Transacciones", fmt_float(df_st["transactions"].sum(skipna=True)))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("")
            top_f = (
                df_st.groupby("family", observed=False)["sales"]
                .sum(min_count=1)
                .sort_values(ascending=False)
                .head(1)
            )
            st.subheader("🥇 Familia líder")
            if len(top_f) == 0:
                st.write("—")
            else:
                st.success(f"**{top_f.index[0]}** · ventas: **{fmt_float(top_f.iloc[0])}**")

        with cR:
            st.subheader("📆 Transacciones por año")
            tx_year = (
                df_st.groupby("year", observed=False)["transactions"]
                .sum(min_count=1)
                .reset_index()
                .sort_values("year")
            )
            fig = px.bar(tx_year, x="year", y="transactions")
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

            st.subheader("🏆 Top 10 tiendas por ventas (en el estado)")
            top_store = (
                df_st.groupby("store_nbr", observed=False)["sales"]
                .sum(min_count=1)
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            fig2 = px.bar(top_store, x="store_nbr", y="sales")
            fig2.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, width="stretch")

            st.subheader("🧺 Familia más vendida por tienda (muestra)")
            by_store_family = (
                df_st.groupby(["store_nbr", "family"], observed=False)["sales"]
                .sum(min_count=1)
                .reset_index()
            )
            if not by_store_family.empty:
                idx = by_store_family.groupby("store_nbr", observed=False)["sales"].idxmax()
                best = by_store_family.loc[idx].sort_values("sales", ascending=False).head(15)
                st.dataframe(
                    best.rename(columns={"family": "top_family", "sales": "top_family_sales"}),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No hay datos suficientes para esta tabla.")


# -----------------------------
# TAB 4
# -----------------------------
with tab4:
    st.header("📈 Estacionalidad")

    a, b, c = st.columns(3, gap="large")

    with a:
        st.subheader("📅 Ventas medias por día de la semana")
        dow = (
            df_f.groupby("day_of_week", observed=False)["sales"]
            .mean()
            .reset_index()
            .sort_values("day_of_week")
        )
        fig = px.bar(dow, x="day_of_week", y="sales")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    with b:
        st.subheader("🗓️ Ventas medias por semana (promedio entre años)")
        weekly = (
            df_f.dropna(subset=["year", "week"])
            .groupby(["year", "week"], observed=False)["sales"]
            .sum(min_count=1)
            .reset_index()
        )
        weekly_mean = (
            weekly.groupby("week", observed=False)["sales"]
            .mean()
            .reset_index()
            .sort_values("week")
        )
        fig2 = px.line(weekly_mean, x="week", y="sales", markers=True)
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, width="stretch")

    with c:
        st.subheader("📆 Ventas medias por mes (promedio entre años)")
        monthly = (
            df_f.dropna(subset=["year", "month"])
            .groupby(["year", "month"], observed=False)["sales"]
            .sum(min_count=1)
            .reset_index()
        )
        monthly_mean = (
            monthly.groupby("month", observed=False)["sales"]
            .mean()
            .reset_index()
            .sort_values("month")
        )
        fig3 = px.line(monthly_mean, x="month", y="sales", markers=True)
        fig3.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, width="stretch")

    st.markdown("---")
    st.subheader("🏷️ Promo vs No promo (ventas medias)")
    promo_cmp = (
        df_f.assign(promo=np.where(df_f["is_promo"], "Promo", "No promo"))
        .groupby("promo", observed=False)["sales"]
        .mean()
        .reset_index()
    )
    fig4 = px.bar(promo_cmp, x="promo", y="sales")
    fig4.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig4, width="stretch")


st.markdown("---")
st.caption("Lectura robusta en Streamlit Cloud: gzip.open + cache + reintentos.")
