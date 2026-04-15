# -*- coding: utf-8 -*-
import streamlit as st
import requests
import json
import time
import pandas as pd
from statistics import mean
import plotly.express as px

# =========================
# CONFIG API (REAL)
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

# ⚠️ REEMPLAZAR CON TUS COOKIES REALES
COOKIES = {
    "cf_clearance": "F7SCO7lF0Dpcu9XsIE8TAFv.xf.XZy4rxC.JTMS3DhI-1772377972-1.2.1.1-.ffaii0U4FySymYo4Hz51UBHzkFJQKhdY._i5pli1YDteUOf0hQ59JA78aycp4I1RaVg3RYe9tM6HEm8nZFJo6T9FVUY_bYOnOxFkVInS3fgeJn2QhnXgeDkS1QM2y31eeBMikoHmgWPau.vf_n1q_0YQdAGXrIr4y6MioQzpZP.zPoRFP01c9dE6pZhOW.6HP4Rmh44ZOgUHYpkIx76Y8kE_3KfVl5I5z28JRqr5aY",
    "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhIjp7Il9pZCI6IjY4MTk2ZDM1ZGM2MTBlNzc0MDIzNDdmYSJ9LCJpYXQiOjE3NzYyNTg2OTEsImV4cCI6MTc3ODg1MDY5MX0._q8bi7XiN2AjznWI0OvPyZQdy0TZAyqj3WfVcRUc91c",
    "jwtdev": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhIjp7Il9pZCI6IjY5YzcyMDEyOTgwODczODc3YmU2ODZiNSJ9LCJpYXQiOjE3NzQ4NzQ4OTQsImV4cCI6MTc3NzQ2Njg5NH0.AcpkpWKZXCxBircSDzZ5wgr1V9GJGf6RKdP9C_JhPGY"
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
# REQUEST (POST TRPC)
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
# FETCH DATA
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
st.title("Warera Analyzer (Full Fixed - POST + Cookies)")

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
        st.warning("No se encontraron ofertas o falló la autenticación.")
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
