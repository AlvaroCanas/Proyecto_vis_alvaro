from __future__ import annotations

from pathlib import Path
import gzip
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# Config + estilo
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

# Rutas robustas (NO dependen del working dir)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
F1 = DATA_DIR / "parte_1.csv.gz"
F2 = DATA_DIR / "parte_2.csv.gz"

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


USECOLS = {
    "id", "date", "store_nbr", "family", "sales", "onpromotion",
    "holiday_type", "locale", "locale_name", "description", "transferred",
    "dcoilwtico", "city", "state", "store_type", "cluster", "transactions",
    "year", "month", "week", "quarter", "day_of_week",
}


def _read_gz_csv(path: Path) -> pd.DataFrame:
    # Lectura robusta usando gzip.open (en Cloud va fino)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return pd.read_csv(
            f,
            low_memory=False,
            usecols=lambda c: (c in USECOLS) or (c == "Unnamed: 0"),
        )


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    # Reintentos suaves (a veces Cloud monta /data con delay)
    last_err = None
    for _ in range(3):
        try:
            if not DATA_DIR.exists():
                raise FileNotFoundError(f"No existe la carpeta: {DATA_DIR}")

            missing = [p for p in (F1, F2) if not p.exists()]
            if missing:
                raise FileNotFoundError("Faltan ficheros:\n- " + "\n- ".join(map(str, missing)))

            if F1.stat().st_size == 0 or F2.stat().st_size == 0:
                raise ValueError("Algún fichero está vacío (0 bytes).")

            df1 = _read_gz_csv(F1)
            df2 = _read_gz_csv(F2)
            df = pd.concat([df1, df2], ignore_index=True)

            df = ensure_cols(df)

            # Tipos
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
            df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce")
            df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").fillna(0)

            # Derivadas
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

            return df

        except Exception as e:
            last_err = e
            time.sleep(1.0)

    raise RuntimeError(f"Error cargando datos tras varios intentos: {last_err}")


# -----------------------------
# Main
# -----------------------------
st.title("📊 Dashboard de Ventas")

try:
    with st.spinner("Cargando datos..."):
        df = load_data()
except Exception as e:
    st.error(
        "No he podido cargar los datos.\n\n"
        "Verifica que existen estos ficheros en GitHub:\n"
        f"- {F1}\n- {F2}\n\n"
        "y que NO estén vacíos."
    )
    st.exception(e)
    st.stop()

# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Filas", fmt_int(len(df)))
k2.metric("Tiendas", fmt_int(safe_nunique(df["store_nbr"])))
k3.metric("Familias", fmt_int(safe_nunique(df["family"])))
k4.metric("Estados", fmt_int(safe_nunique(df["state"])))

st.markdown("---")

# -----------------------------
# Tabs (SIN selectbox)
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["1) Visión global", "2) Tiendas", "3) Estados", "4) Estacionalidad"]
)

# -----------------------------
# TAB 1
# -----------------------------
with tab1:
    st.header("📌 Visión global")

    c1, c2 = st.columns((1.15, 0.85), gap="large")

    with c1:
        st.subheader("🏆 Top 10 familias por ventas (total)")
        top_fam = (
            df.groupby("family", observed=False)["sales"]
            .sum(min_count=1)
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig = px.bar(top_fam, x="sales", y="family", orientation="h")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.subheader("🏷️ Promo vs No promo (ventas medias)")
        promo_cmp = (
            df.assign(promo=np.where(df["is_promo"], "Promo", "No promo"))
            .groupby("promo", observed=False)["sales"]
            .mean()
            .reset_index()
        )
        fig2 = px.bar(promo_cmp, x="promo", y="sales")
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, width="stretch")

    st.markdown("---")

    st.subheader("📈 Evolución mensual total (ventas)")
    if df["date"].notna().any():
        monthly = (
            df.dropna(subset=["date"])
            .set_index("date")["sales"]
            .resample("MS")
            .sum(min_count=1)
            .reset_index()
            .rename(columns={"sales": "sales_month"})
        )
        fig3 = px.line(monthly, x="date", y="sales_month", markers=True)
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, width="stretch")
    else:
        st.info("No hay fechas válidas para la serie temporal.")

# -----------------------------
# TAB 2
# -----------------------------
with tab2:
    st.header("🏬 Tiendas (rankings)")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader("🏆 Top 15 tiendas por ventas totales")
        top_store = (
            df.groupby("store_nbr", observed=False)["sales"]
            .sum(min_count=1)
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig = px.bar(top_store, x="store_nbr", y="sales")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.subheader("🔥 Top 15 tiendas con más ventas en promoción")
        promo_store = (
            df[df["is_promo"]]
            .groupby("store_nbr", observed=False)["sales"]
            .sum(min_count=1)
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
            .rename(columns={"sales": "promo_sales"})
        )
        fig2 = px.bar(promo_store, x="store_nbr", y="promo_sales")
        fig2.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, width="stretch")

    st.markdown("---")

    st.subheader("🧺 Familia líder por tienda (Top 20 tiendas por ventas)")
    by_store_family = (
        df.groupby(["store_nbr", "family"], observed=False)["sales"]
        .sum(min_count=1)
        .reset_index()
    )
    if not by_store_family.empty:
        idx = by_store_family.groupby("store_nbr", observed=False)["sales"].idxmax()
        best = by_store_family.loc[idx]
        top20stores = (
            df.groupby("store_nbr", observed=False)["sales"].sum(min_count=1)
            .sort_values(ascending=False)
            .head(20)
            .index
        )
        best = best[best["store_nbr"].isin(top20stores)].sort_values("sales", ascending=False)
        st.dataframe(
            best.rename(columns={"family": "top_family", "sales": "top_family_sales"}),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No hay datos suficientes para esta tabla.")

# -----------------------------
# TAB 3
# -----------------------------
with tab3:
    st.header("🗺️ Estados (rankings)")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader("🏆 Top 15 estados por ventas")
        top_state = (
            df.groupby("state", observed=False)["sales"]
            .sum(min_count=1)
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig = px.bar(top_state, x="sales", y="state", orientation="h")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.subheader("🏬 Nº de tiendas por estado (Top 15)")
        stores_state = (
            df.dropna(subset=["state", "store_nbr"])
            .groupby("state", observed=False)["store_nbr"]
            .nunique()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
            .rename(columns={"store_nbr": "n_stores"})
        )
        fig2 = px.bar(stores_state, x="n_stores", y="state", orientation="h")
        fig2.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, width="stretch")

    st.markdown("---")
    st.subheader("🏷️ Top 10 familias por ventas en el Top 5 estados")
    top5 = (
        df.groupby("state", observed=False)["sales"]
        .sum(min_count=1)
        .sort_values(ascending=False)
        .head(5)
        .index
    )
    df5 = df[df["state"].isin(top5)]
    fam5 = (
        df5.groupby(["state", "family"], observed=False)["sales"]
        .sum(min_count=1)
        .reset_index()
    )
    # nos quedamos con top 10 familias globalmente dentro de los top 5 estados
    top10_fams = (
        fam5.groupby("family", observed=False)["sales"].sum(min_count=1)
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    fam5 = fam5[fam5["family"].isin(top10_fams)]
    fig3 = px.bar(fam5, x="family", y="sales", color="state", barmode="group")
    fig3.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, width="stretch")

# -----------------------------
# TAB 4
# -----------------------------
with tab4:
    st.header("📈 Estacionalidad")

    a, b, c = st.columns(3, gap="large")

    with a:
        st.subheader("📅 Ventas medias por día de la semana")
        dow = (
            df.groupby("day_of_week", observed=False)["sales"]
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
            df.dropna(subset=["year", "week"])
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
            df.dropna(subset=["year", "month"])
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
