# -*- coding: utf-8 -*-
import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px

# =========================
# CONFIG API
# =========================

OFFERS_API = "https://api4.warera.io/trpc/itemOffer.getItemOffers?batch=1"

COMMON_HEADERS = {
    "accept": "*/*",
    "content-type": "application/json",
    "origin": "https://app.warera.io",
    "referer": "https://app.warera.io/",
    "user-agent": "Mozilla/5.0"
}

# ⚠️ REEMPLAZAR
COOKIES = {
    "cf_clearance": "F7SCO7lF0Dpcu9XsIE8TAFv.xf.XZy4rxC.JTMS3DhI-1772377972-1.2.1.1-.ffaii0U4FySymYo4Hz51UBHzkFJQKhdY._i5pli1YDteUOf0hQ59JA78aycp4I1RaVg3RYe9tM6HEm8nZFJo6T9FVUY_bYOnOxFkVInS3fgeJn2QhnXgeDkS1QM2y31eeBMikoHmgWPau.vf_n1q_0YQdAGXrIr4y6MioQzpZP.zPoRFP01c9dE6pZhOW.6HP4Rmh44ZOgUHYpkIx76Y8kE_3KfVl5I5z28JRqr5aY",
    "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhIjp7Il9pZCI6IjY4MTk2ZDM1ZGM2MTBlNzc0MDIzNDdmYSJ9LCJpYXQiOjE3NzYyNTg2OTEsImV4cCI6MTc3ODg1MDY5MX0._q8bi7XiN2AjznWI0OvPyZQdy0TZAyqj3WfVcRUc91c",
    "jwtdev": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhIjp7Il9pZCI6IjY5YzcyMDEyOTgwODczODc3YmU2ODZiNSJ9LCJpYXQiOjE3NzQ4NzQ4OTQsImV4cCI6MTc3NzQ2Njg5NH0.AcpkpWKZXCxBircSDzZ5wgr1V9GJGf6RKdP9C_JhPGY"
}

# =========================
# SCRAP
# =========================

SCRAP_PER_TIER = {
    1: 6,
    2: 18,
    3: 54,
    4: 162,
    5: 486,
    6: 1460
}

# =========================
# ITEMS
# =========================

EQUIPMENT_TYPES = {
    "armor": {
        "types": ["helmet", "pants", "gloves", "boots", "chest"],
        "suffixes": ["1", "2", "3", "4", "5", "6"]
    },
    "weapons": {
        "types": ["knife", "gun", "rifle", "sniper", "tank", "jet"],
        "suffixes": [""]
    }
}

def generate_codes():
    codes = []
    for t in EQUIPMENT_TYPES["armor"]["types"]:
        for s in EQUIPMENT_TYPES["armor"]["suffixes"]:
            codes.append(f"{t}{s}")
    for w in EQUIPMENT_TYPES["weapons"]["types"]:
        codes.append(w)
    return codes

CODES = generate_codes()

# =========================
# REQUEST
# =========================

def safe_post(payload):
    try:
        r = requests.post(
            OFFERS_API,
            headers=COMMON_HEADERS,
            cookies=COOKIES,
            json={"0": payload}
        )

        if r.status_code != 200:
            print("ERROR:", r.status_code, r.text)
            return None

        return r.json()

    except Exception as e:
        print("REQUEST ERROR:", e)
        return None

# =========================
# FETCH OFFERS ONLY
# =========================

def fetch_offers(code, pages=1, limit=50):
    offers = []
    cursor = None

    for _ in range(pages):
        payload = {
            "limit": limit,
            "transactionType": "itemMarket",
            "itemCode": code,
            "condition": "100%"
        }

        if cursor:
            payload["cursor"] = cursor

        data = safe_post(payload)
        if not data:
            break

        items = data[0]["result"]["data"]["items"]

        for o in items:
            itm = o["item"]
            if itm.get("state") == itm.get("maxState"):
                offers.append({
                    "price": o["price"],
                    "total_skill": sum(itm.get("skills", {}).values())
                })

        cursor = data[0]["result"]["data"].get("nextCursor")
        if not cursor:
            break

    return offers

# =========================
# MARKET LOGIC
# =========================

def build_market_structures(offers):
    if not offers:
        return None

    df = pd.DataFrame(offers)

    # price floor global
    floor = df["price"].min()

    # floor por stat
    floor_by_stat = df.groupby("total_skill")["price"].min().to_dict()

    return df, floor, floor_by_stat


def compute_edges(df):
    df = df.copy()

    edges = []

    for i, row in df.iterrows():
        stat = row["total_skill"]
        price = row["price"]

        # competencia cercana = mismo stat
        same_stat = df[df["total_skill"] == stat]

        if len(same_stat) <= 1:
            edges.append(0)
            continue

        # excluir el mismo item
        competitors = same_stat[same_stat["price"] != price]

        if competitors.empty:
            edges.append(0)
            continue

        min_competitor = competitors["price"].min()

        edge = min_competitor - price
        edges.append(edge)

    df["edge"] = edges
    return df

# =========================
# UI
# =========================

st.set_page_config(layout="wide")
st.title("Warera Analyzer (Live Market Only)")

# Sidebar
st.sidebar.header("Configuración")

pages = st.sidebar.slider("Pages", 1, 20, 1)
scrap_price = st.sidebar.number_input("Precio del scrap", value=0.214)

tab1, tab2 = st.tabs(["🔥 Ofertas", "📉 Price Floors"])

# =========================
# FETCH BUTTON
# =========================

if st.button("Actualizar Datos"):
    all_offers = []
    all_floors = {}

    progress = st.progress(0)

    for i, code in enumerate(CODES):
        offers = fetch_offers(code, pages)

        if not offers:
            continue

        df, floor, floor_by_stat = build_market_structures(offers)

        df["code"] = code
        df = compute_edges(df)

        all_offers.append(df)
        all_floors[code] = (floor, floor_by_stat)

        progress.progress((i + 1) / len(CODES))
        time.sleep(0.05)

    if all_offers:
        final_df = pd.concat(all_offers)
        st.session_state["df"] = final_df
        st.session_state["floors"] = all_floors

# =========================
# TAB 1
# =========================

with tab1:
    if "df" in st.session_state:
        df = st.session_state["df"].sort_values("edge", ascending=False)

        st.subheader("Mejores oportunidades (vs competencia directa)")
        st.dataframe(df.head(50))

        fig = px.histogram(df, x="edge", nbins=50)
        st.plotly_chart(fig)
    else:
        st.info("No hay datos")

# =========================
# TAB 2
# =========================

with tab2:
    st.header("📉 Price Floors (Ofertas actuales)")

    if "floors" not in st.session_state:
        st.info("Primero actualizá datos")
    else:
        floors = st.session_state["floors"]

        selected_item = st.selectbox("Seleccionar item", sorted(floors.keys()))

        floor, floor_by_stat = floors[selected_item]

        # detectar tier
        if selected_item[-1].isdigit():
            tier = int(selected_item[-1])
        else:
            weapon_tier_map = {
                "knife": 1,
                "gun": 2,
                "rifle": 3,
                "sniper": 4,
                "tank": 5,
                "jet": 6
            }
            tier = weapon_tier_map.get(selected_item, 1)

        scrap_value = SCRAP_PER_TIER.get(tier, 0) * scrap_price

        st.metric("Precio mínimo global", f"{floor:.2f}")
        st.metric("Valor scrap", f"{scrap_value:.2f}")

        rows = []

        for stat, price in floor_by_stat.items():
            rows.append({
                "stat": stat,
                "min_price": price,
                "scrap_value": scrap_value,
                "vs_scrap": price - scrap_value
            })

        df = pd.DataFrame(rows).sort_values("stat")

        st.dataframe(df)

        fig = px.line(df, x="stat", y="min_price", title="Price Floor por stat")
        st.plotly_chart(fig)
