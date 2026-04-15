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

API_KEY = "wae_01ad18a7153dc706fcdc009f2c567b008eb5db21cf551884497ebc5bab2fd4f9"

COMMON_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "x-api-key": API_KEY,
    "origin": "https://app.warera.io",
    "referer": "https://app.warera.io/",
    "user-agent": "Mozilla/5.0"
}

TRANX_API = "https://api2.warera.io/trpc/transaction.getPaginatedTransactions"
OFFERS_API = "https://api2.warera.io/trpc/itemOffer.getItemOffers"

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
# REQUEST SAFE
# =========================

def safe_get(url, params):
    try:
        r = requests.get(url, params=params, headers=COMMON_HEADERS)
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

        params = {"batch": "1", "input": json.dumps({"0": payload})}
        data = safe_get(TRANX_API, params)

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

        params = {"batch": "1", "input": json.dumps({"0": payload})}
        data = safe_get(OFFERS_API, params)

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
# PRICING MODEL (BUCKET)
# =========================

def build_price_buckets(transactions):
    buckets = {}

    for tx in transactions:
        price = tx.get("money", 0)
        skills = tx["item"].get("skills", {})
        total = sum(skills.values())

        if price <= 0 or total == 0:
            continue

        bucket = int(total / 10) * 10

        buckets.setdefault(bucket, []).append(price)

    bucket_stats = {}

    for b, prices in buckets.items():
        if prices:
            bucket_stats[b] = {
                "mean": mean(prices),
                "min": min(prices),
                "max": max(prices),
                "count": len(prices)
            }

    return bucket_stats


def estimate_price(bucket_stats, total_skill):
    if not bucket_stats:
        return None

    bucket = int(total_skill / 10) * 10

    if bucket in bucket_stats:
        return bucket_stats[bucket]["mean"]

    closest = min(bucket_stats.keys(), key=lambda x: abs(x - bucket))
    return bucket_stats[closest]["mean"]

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(layout="wide")
st.title("Warera Analyzer (Fixed Version)")

pages = st.sidebar.slider("Pages", 1, 5, 1)

if st.button("Actualizar Datos"):
    all_offers = []

    progress = st.progress(0)

    for i, code in enumerate(CODES):
        txs = fetch_historical_transactions(code, pages)
        bucket_stats = build_price_buckets(txs)

        offers = fetch_active_offers(code, pages)

        print(f"{code} -> txs: {len(txs)}, offers: {len(offers)}")

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
        time.sleep(0.05)

    st.write("Cantidad de ofertas analizadas:", len(all_offers))

    if not all_offers:
        st.warning("No se encontraron ofertas o no se pudieron calcular precios.")
        st.stop()

    df = pd.DataFrame(all_offers)

    if "edge" not in df.columns:
        st.error("Error: columna 'edge' no encontrada")
        st.write(df.head())
        st.stop()

    df = df.sort_values("edge", ascending=False)

    st.session_state["df"] = df

# =========================
# RESULTADOS
# =========================

if "df" in st.session_state:
    df = st.session_state["df"]

    st.subheader("Mejores oportunidades")
    st.dataframe(df.head(50))

    fig = px.histogram(df, x="edge", nbins=50)
    st.plotly_chart(fig)
