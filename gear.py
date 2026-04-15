# -*- coding: utf-8 -*-
import streamlit as st
import requests
import json
import time
import pandas as pd
from statistics import mean
import plotly.express as px

# =========================
# CONFIG API
# =========================

TRANX_API = "https://api4.warera.io/trpc/transaction.getPaginatedTransactions?batch=1"
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
# CONFIG SCRAP
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
        "suffixes": ["1", "2", "3", "4", "5"]
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

def safe_post(url, payload):
    try:
        r = requests.post(
            url,
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
# FETCH
# =========================

def fetch_historical_transactions(code, pages=1, limit=50):
    transactions = []
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

        data = safe_post(TRANX_API, payload)
        if not data:
            break

        items = data[0]["result"]["data"]["items"]

        for tx in items:
            itm = tx["item"]
            if itm.get("state") == itm.get("maxState"):
                transactions.append(tx)

        cursor = data[0]["result"]["data"].get("nextCursor")
        if not cursor:
            break

    return transactions


def fetch_active_offers(code, pages=1, limit=30):
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

        data = safe_post(OFFERS_API, payload)
        if not data:
            break

        items = data[0]["result"]["data"]["items"]

        for o in items:
            itm = o["item"]
            if itm.get("state") == itm.get("maxState"):
                offers.append({
                    "price": o["price"],
                    "skills": itm.get("skills", {}),
                    "total_skill": sum(itm.get("skills", {}).values())
                })

        cursor = data[0]["result"]["data"].get("nextCursor")
        if not cursor:
            break

    return offers

# =========================
# PRICING
# =========================

def build_price_buckets(transactions):
    buckets = {}

    for tx in transactions:
        price = tx.get("money", 0)
        total = sum(tx["item"].get("skills", {}).values())

        if price <= 0 or total == 0:
            continue

        bucket = int(total / 10) * 10
        buckets.setdefault(bucket, []).append(price)

    return {k: mean(v) for k, v in buckets.items()}


def estimate_price(bucket_stats, total_skill):
    if not bucket_stats:
        return None

    bucket = int(total_skill / 10) * 10

    if bucket in bucket_stats:
        return bucket_stats[bucket]

    closest = min(bucket_stats.keys(), key=lambda x: abs(x - bucket))
    return bucket_stats[closest]


def build_min_price_table(transactions):
    stat_prices = {}

    for tx in transactions:
        price = tx.get("money", 0)
        total = sum(tx["item"].get("skills", {}).values())

        if price <= 0 or total == 0:
            continue

        stat_prices.setdefault(total, []).append(price)

    return {k: min(v) for k, v in stat_prices.items()}

# =========================
# UI
# =========================

st.set_page_config(layout="wide")
st.title("Warera Analyzer (Price Floors + Scrap)")

# Sidebar
st.sidebar.header("Configuración")

pages = st.sidebar.slider("Pages", 1, 5, 1)
scrap_price = st.sidebar.number_input("Precio del scrap", value=0.214)

# Tabs
tab1, tab2 = st.tabs(["🔥 Ofertas", "📉 Price Floors"])

# =========================
# BOTON
# =========================

if st.button("Actualizar Datos"):
    all_offers = []
    all_min_tables = {}

    progress = st.progress(0)

    for i, code in enumerate(CODES):
        txs = fetch_historical_transactions(code, pages)
        bucket_stats = build_price_buckets(txs)
        min_table = build_min_price_table(txs)

        all_min_tables[code] = min_table

        offers = fetch_active_offers(code, pages)

        for o in offers:
            est = estimate_price(bucket_stats, o["total_skill"])

            if est is None:
                continue

            edge = est - o["price"]

            all_offers.append({
                "code": code,
                "price": o["price"],
                "est_price": est,
                "edge": edge,
                "total_skill": o["total_skill"]
            })

        progress.progress((i + 1) / len(CODES))

    st.session_state["df"] = pd.DataFrame(all_offers)
    st.session_state["min_tables"] = all_min_tables

# =========================
# TAB 1 - OFERTAS
# =========================

with tab1:
    if "df" in st.session_state and not st.session_state["df"].empty:
        df = st.session_state["df"].sort_values("edge", ascending=False)

        st.subheader("Mejores oportunidades")
        st.dataframe(df.head(50))

        fig = px.histogram(df, x="edge", nbins=50)
        st.plotly_chart(fig)
    else:
        st.info("No hay datos")

# =========================
# TAB 2 - PRICE FLOORS
# =========================

with tab2:
    st.header("📉 Precio mínimo por stat")

    if "min_tables" not in st.session_state:
        st.info("Primero actualizá los datos")
    else:
        tables = st.session_state["min_tables"]

        selected_item = st.selectbox("Seleccionar item", sorted(tables.keys()))
        stat_table = tables[selected_item]

        if not stat_table:
            st.warning("Sin datos")
        else:
            rows = []

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

            st.metric("Valor scrap del item", f"{scrap_value:.2f}")

            for stat, price in stat_table.items():
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
