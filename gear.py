# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 10:25:25 2025

@author: d908896
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import numpy as np
from statistics import mean
import plotly.express as px

# — API Endpoints —
TRANX_API = "https://api2.warera.io/trpc/transaction.getPaginatedTransactions"
OFFERS_API = "https://api2.warera.io/trpc/itemOffer.getItemOffers"

# — Configuración inicial con tier 1 y knife —
EQUIPMENT_TYPES = {
    "armor": {
        "types": ["helmet", "pants", "gloves", "boots", "chest"],
        "suffixes": ["1", "2", "3", "4", "5", "6"],  # Agregado tier 1
        "ranges": {
            "1": {"min": 1, "max": 5},   # Nuevo rango para tier 1
            "2": {"min": 4, "max": 10},
            "3": {"min": 8, "max": 15},
            "4": {"min": 14, "max": 25},
            "5": {"min": 19, "max": 35}
        }
    },
    "weapons": {
        "types": ["knife", "gun", "rifle", "sniper", "tank", "jet"],  # Agregado jet
        "suffixes": [""],
        "ranges": {
            "knife": {"min": 21, "max": 45},
            "gun": {"min": 51, "max": 70},
            "rifle": {"min": 75, "max": 105},
            "sniper": {"min": 110, "max": 140},
            "tank": {"min": 140, "max": 190},
            "jet": {"min": 200, "max": 250}  # Rango estimado para jet
        }
    }
}

# Probabilidades de loot cases para armaduras
LOOT_CASE_ARMOR_RATES = {
    "T1": 0.62,
    "T2": 0.30,
    "T3": 0.071,
    "T4": 0.0085,
    "T5": 0.0005
}

# Probabilidades de loot cases para armas (con jet)
LOOT_CASE_WEAPON_RATES = {
    "T1": 0.62,
    "T2": 0.30,
    "T3": 0.071,
    "T4": 0.0085,
    "T5": 0.0004,  # Tank reducido a 0.04%
    "T6": 0.0001   # Jet añadido con 0.01%
}

# Costos de merge por tier
MERGE_COSTS = {
    1: 2,
    2: 4,
    3: 8,
    4: 16,
    5: 32  # Agregado costo para merge de tier 5 (tank a jet)
}

# Inicializar estado de la sesión
if 'historical_stats' not in st.session_state:
    st.session_state.historical_stats = {}
    
if 'bargains' not in st.session_state:
    st.session_state.bargains = []
    
if 'merge_analysis' not in st.session_state:
    st.session_state.merge_analysis = []
    
if 'loot_case_value' not in st.session_state:
    st.session_state.loot_case_value = None

if 'active_lowest_prices' not in st.session_state:
    st.session_state.active_lowest_prices = {}

def generate_codes():
    """Genera todos los códigos de items a analizar"""
    codes = []
    for item_type in EQUIPMENT_TYPES["armor"]["types"]:
        for suffix in EQUIPMENT_TYPES["armor"]["suffixes"]:
            codes.append(f"{item_type}{suffix}")
    for weapon_type in EQUIPMENT_TYPES["weapons"]["types"]:
        codes.append(weapon_type)
    return codes

def fetch_historical_transactions(code: str, pages=1, limit=100):
    """Obtiene transacciones históricas para un item específico (solo state=100%)"""
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
        
        try:
            r = requests.get(TRANX_API, params=params, headers=HEADERS_TRX)
            r.raise_for_status()
            data = r.json()
            items = data[0]["result"]["data"]["items"] if data else []
            
            for tx in items:
                # Verificar condición 100% (doble verificación)
                state = tx["item"].get("state", 0)
                maxst = tx["item"].get("maxState", 100)
                if state != maxst or maxst == 0:
                    continue
                
                transactions.append(tx)

            cursor = data[0]["result"]["data"].get("nextCursor") if data else None
            if not cursor:
                break
                
        except Exception as e:
            st.error(f"Error histórico para {code}: {str(e)}")
            break
    
    return transactions

def calculate_historical_stats(transactions):
    """Calcula estadísticas históricas para un conjunto de transacciones"""
    if not transactions:
        return None
    
    # Precio por punto de habilidad (PPP) y precios reales
    ppp_values = []
    prices = []
    
    for tx in transactions:
        price = tx.get("money", 0)
        skills = tx["item"].get("skills", {})
        total_skill = sum(skills.values())
        
        if price > 0 and total_skill > 0:
            ppp = price / total_skill
            ppp_values.append(ppp)
            prices.append(price)
    
    if not ppp_values or not prices:
        return None
    
    # Estadísticas básicas
    sorted_ppp = sorted(ppp_values)
    n = len(sorted_ppp)
    
    stats = {
        "count": n,
        "min_ppp": min(ppp_values),
        "max_ppp": max(ppp_values),
        "mean_ppp": mean(ppp_values),
        "min_price": min(prices),
        "max_price": max(prices),
        "mean_price": mean(prices),
        "p10": sorted_ppp[int(n*0.1)] if n > 10 else sorted_ppp[0],
        "p25": sorted_ppp[int(n*0.25)] if n > 4 else sorted_ppp[0],
        "p50": sorted_ppp[int(n*0.5)],
        "p75": sorted_ppp[int(n*0.75)] if n > 4 else sorted_ppp[-1],
        "p90": sorted_ppp[int(n*0.9)] if n > 10 else sorted_ppp[-1],
    }
    
    return stats

def fetch_active_offers(code: str, pages=1, limit=50):
    """Trae todas las ofertas activas para un item específico (solo state=100%)"""
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
        
        try:
            r = requests.get(OFFERS_API, params=params, headers=HEADERS_OFF)
            r.raise_for_status()
            data = r.json()
            results = data[0]["result"]["data"]["items"] if data else []
            
            # Recoger todas las ofertas con condición 100%
            for o in results:
                itm = o["item"]
                state = itm.get("state", 0)
                maxst = itm.get("maxState", 100)
                price = o.get("price", 0)
                
                if state == maxst and maxst > 0 and price > 0:
                    offers.append({
                        "id": o.get("id"),
                        "code": code,
                        "price": price,
                        "item": itm,
                        "skills": itm.get("skills", {}),
                        "total_skill": sum(itm.get("skills", {}).values())
                    })
                    
        except Exception as e:
            st.error(f"Error ofertas para {code}: {str(e)}")
            break

        cursor = data[0]["result"]["data"].get("nextCursor") if data else None
        if not cursor:
            break

    return offers

def evaluate_offer(offer, stats):
    """Evalúa una oferta usando estadísticas históricas"""
    price = offer.get("price", 0)
    skills = offer.get("skills", {})
    total_skill = sum(skills.values())
    
    if price <= 0 or total_skill <= 0:
        return None
    
    # 1. Calcular Precio por Punto (PPP)
    current_ppp = price / total_skill
    
    # 2. Determinar percentil
    if current_ppp < stats["p10"]:
        percentile_rank = "Excelente (Top 10%)"
        percentile_score = 1.0
    elif current_ppp < stats["p25"]:
        percentile_rank = "Muy Buena (Top 25%)"
        percentile_score = 0.8
    elif current_ppp < stats["p50"]:
        percentile_rank = "Buena (Top 50%)"
        percentile_score = 0.6
    elif current_ppp < stats["p75"]:
        percentile_rank = "Normal (Top 75%)"
        percentile_score = 0.4
    elif current_ppp < stats["p90"]:
        percentile_rank = "Regular (Top 90%)"
        percentile_score = 0.2
    else:
        percentile_rank = "Cara"
        percentile_score = 0.0
    
    # 3. Calcular Z-Score
    mean_ppp = stats["mean_ppp"]
    std_ppp = (stats["p75"] - stats["p25"]) / 1.349  # Estimación robusta de desviación
    
    if std_ppp > 0:
        z_score = (current_ppp - mean_ppp) / std_ppp
    else:
        z_score = 0
    
    # 5. Calcular puntaje compuesto
    term1 = 0.5 * percentile_score
    
    ppp_ratio = current_ppp / stats["p50"] if stats["p50"] > 0 else 1
    clipped_ratio = min(1.0, max(0.0, ppp_ratio))
    term2 = 0.3 * (1.0 - clipped_ratio)
    
    term3 = 0.2 * 0.5  # Valor predeterminado
    
    composite_score = term1 + term2 + term3
    
    # 6. Calificar la oferta
    if composite_score > 0.8:
        rating = "⭐️⭐️⭐️⭐️⭐️ Excelente"
    elif composite_score > 0.6:
        rating = "⭐️⭐️⭐️⭐️ Muy Buena"
    elif composite_score > 0.4:
        rating = "⭐️⭐️⭐️ Buena"
    elif composite_score > 0.2:
        rating = "⭐️⭐️ Regular"
    else:
        rating = "⭐️ Cara"
    
    return {
        "price": price,
        "total_skill": total_skill,
        "current_ppp": current_ppp,
        "percentile_rank": percentile_rank,
        "z_score": z_score,
        "composite_score": composite_score,
        "rating": rating
    }

def analyze_merging_viability(historical_stats, active_lowest_prices):
    """Analiza la viabilidad de mergear items por tipo específico"""
    analysis = []
    
    # Para armaduras
    for armor_type in EQUIPMENT_TYPES["armor"]["types"]:
        for tier in range(1, 5):  # Tiers 1-4
            current_code = f"{armor_type}{tier}"
            next_code = f"{armor_type}{tier+1}"
            
            # Obtener precio más bajo actual
            current_lowest = active_lowest_prices.get(current_code)
            if current_lowest is None:
                continue
                
            # Obtener precio de referencia para el tier superior
            next_stats = historical_stats.get(next_code)
            next_lowest = active_lowest_prices.get(next_code)
            
            if not next_stats:
                continue
                
            # Usar el MÁXIMO entre el precio histórico promedio y la oferta más baja actual
            next_ref_price = max(
                next_stats["mean_price"], 
                next_lowest if next_lowest is not None else 0
            )
            
            # Calcular costo total de merge
            merge_cost = MERGE_COSTS.get(tier, 0)
            total_merge_cost = (current_lowest * 3) + merge_cost
            
            # Calcular viabilidad
            viable = total_merge_cost < next_ref_price
            savings = next_ref_price - total_merge_cost if viable else total_merge_cost - next_ref_price
            
            analysis.append({
                "item_type": armor_type,
                "current_tier": tier,
                "next_tier": tier+1,
                "current_lowest": current_lowest,
                "merge_cost": merge_cost,
                "total_merge_cost": total_merge_cost,
                "next_ref_price": next_ref_price,
                "viable": viable,
                "savings": savings,
                "savings_pct": (savings / next_ref_price * 100) if next_ref_price else 0
            })
    
    # Para armas: definimos todos los merges posibles
    weapon_merges = [
        # (current_weapon, current_tier, next_weapon, next_tier)
        ("knife", 1, "gun", 2),
        ("gun", 2, "rifle", 3),
        ("rifle", 3, "sniper", 4),
        ("sniper", 4, "tank", 5),
        ("tank", 5, "jet", 6)  # Nuevo merge: tank a jet
    ]
    
    for current_weapon, current_tier, next_weapon, next_tier in weapon_merges:
        current_lowest = active_lowest_prices.get(current_weapon)
        if current_lowest is None:
            continue
            
        # Obtener estadísticas del arma siguiente
        next_stats = historical_stats.get(next_weapon)
        next_lowest = active_lowest_prices.get(next_weapon)
        
        if not next_stats:
            continue
            
        # Precio de referencia para el siguiente tier (arma siguiente)
        next_ref_price = max(
            next_stats["mean_price"],
            next_lowest if next_lowest is not None else 0
        )
        
        # Calcular costo total de merge
        merge_cost = MERGE_COSTS.get(current_tier, 0)
        total_merge_cost = (current_lowest * 3) + merge_cost
        
        # Calcular viabilidad
        viable = total_merge_cost < next_ref_price
        savings = next_ref_price - total_merge_cost if viable else total_merge_cost - next_ref_price
        
        analysis.append({
            "item_type": current_weapon,
            "current_tier": current_tier,
            "next_tier": next_tier,
            "current_lowest": current_lowest,
            "merge_cost": merge_cost,
            "total_merge_cost": total_merge_cost,
            "next_ref_price": next_ref_price,
            "viable": viable,
            "savings": savings,
            "savings_pct": (savings / next_ref_price * 100) if next_ref_price else 0
        })
    
    return analysis

def calculate_loot_case_value(historical_stats, active_lowest_prices):
    """Calcula el valor esperado de abrir un loot case considerando 50% chance de arma y 50% de equipamiento"""
    # Separar items por tipo (armor vs weapons)
    armor_tier_prices = {}
    weapon_tier_prices = {}
    
    for code, stats in historical_stats.items():
        # Determinar si es armor o weapon y su tier
        if code[-1].isdigit():  # Es armor (termina con número)
            tier = int(code[-1])
            # Usar el MÁXIMO entre el precio histórico promedio y la oferta más baja actual
            ref_price = max(
                stats["mean_price"], 
                active_lowest_prices.get(code, 0)
            )
            if tier not in armor_tier_prices:
                armor_tier_prices[tier] = []
            armor_tier_prices[tier].append(ref_price)
        else:  # Es weapon
            # Mapear weapon a tier
            weapon_to_tier = {
                "knife": 1,
                "gun": 2,
                "rifle": 3,
                "sniper": 4,
                "tank": 5,
                "jet": 6
            }
            tier = weapon_to_tier.get(code)
            if tier:
                # Usar el MÁXIMO entre el precio histórico promedio y la oferta más baja actual
                ref_price = max(
                    stats["mean_price"], 
                    active_lowest_prices.get(code, 0)
                )
                if tier not in weapon_tier_prices:
                    weapon_tier_prices[tier] = []
                weapon_tier_prices[tier].append(ref_price)
    
    # Calcular valor promedio por tier para armor
    armor_tier_avg = {}
    for tier, prices in armor_tier_prices.items():
        armor_tier_avg[tier] = mean(prices) if prices else 0
    
    # Calcular valor promedio por tier para weapons
    weapon_tier_avg = {}
    for tier, prices in weapon_tier_prices.items():
        weapon_tier_avg[tier] = mean(prices) if prices else 0
    
    # Calcular valor esperado para armor
    armor_expected_value = 0
    for tier, rate in LOOT_CASE_ARMOR_RATES.items():
        tier_num = int(tier[1])  # Convertir "T1" a 1
        tier_value = armor_tier_avg.get(tier_num, 0)
        armor_expected_value += rate * tier_value
    
    # Calcular valor esperado para weapons
    weapon_expected_value = 0
    for tier, rate in LOOT_CASE_WEAPON_RATES.items():
        tier_num = int(tier[1])  # Convertir "T1" a 1
        tier_value = weapon_tier_avg.get(tier_num, 0)
        weapon_expected_value += rate * tier_value
    
    # Valor esperado final (50% chance de armor, 50% chance de weapon)
    expected_value = 0.5 * armor_expected_value + 0.5 * weapon_expected_value
    
    return expected_value, armor_expected_value, weapon_expected_value, armor_tier_avg, weapon_tier_avg

# Generar todos los códigos a analizar
CODES = generate_codes()

# Interfaz de Streamlit
st.set_page_config(page_title="Warera Item Analyzer", layout="wide")
st.title("🛡️ Warera Item Analyzer - Análisis Avanzado")

# Campo para ingresar el token
st.sidebar.header("🔐 Configuración de API")
api_token = st.sidebar.text_input("API Token", type="password", value="")

if not api_token:
    st.warning("🔑 Por favor ingresa tu API Token para comenzar")
    st.stop()

# Actualizar headers con el token
HEADERS_TRX = {"Authorization": api_token}
HEADERS_OFF = {
    "accept": "application/json, text/plain, */*",
    "authorization": api_token,
    "origin": "https://app.warera.io",
    "referer": "https://app.warera.io/"
}

# Parámetros configurables
st.sidebar.header("⚙️ Parámetros de Consulta")
col1, col2 = st.sidebar.columns(2)

with col1:
    st.subheader("Históricos")
    hist_pages = st.number_input("Páginas", min_value=1, value=1, key="hist_pages")
    hist_limit = st.number_input("Límite por página", min_value=1, max_value=100, value=30, key="hist_limit")

with col2:
    st.subheader("Ofertas Activas")
    offer_pages = st.number_input("Páginas", min_value=1, value=1, key="offer_pages")
    offer_limit = st.number_input("Límite por página", min_value=1, max_value=50, value=15, key="offer_limit")

# Botones de acción
st.header("🚀 Acciones")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Actualizar Todos los Datos", use_container_width=True):
        with st.spinner("Recolectando y analizando datos..."):
            historical_stats = {}
            active_lowest_prices = {}
            progress_bar = st.progress(0)
            total_codes = len(CODES)
            
            for i, code in enumerate(CODES):
                # Obtener datos históricos
                transactions = fetch_historical_transactions(
                    code, 
                    pages=hist_pages, 
                    limit=hist_limit
                )
                stats = calculate_historical_stats(transactions)
                if stats:
                    historical_stats[code] = stats
                
                # Obtener oferta más baja actual
                offers = fetch_active_offers(
                    code, 
                    pages=offer_pages, 
                    limit=offer_limit
                )
                if offers:
                    active_lowest_prices[code] = min(o['price'] for o in offers)
                else:
                    active_lowest_prices[code] = None
                
                progress_bar.progress((i+1) / total_codes)
                time.sleep(0.05)
            
            st.session_state.historical_stats = historical_stats
            st.session_state.active_lowest_prices = active_lowest_prices
            st.success(f"✅ Datos actualizados para {len(historical_stats)} items")
            
            # Calcular análisis de merge
            st.session_state.merge_analysis = analyze_merging_viability(
                historical_stats, 
                active_lowest_prices
            )
            
            # Calcular valor de loot case
            expected_value, armor_ev, weapon_ev, armor_tier_avg, weapon_tier_avg = calculate_loot_case_value(
                historical_stats, 
                active_lowest_prices
            )
            st.session_state.loot_case_value = (expected_value, armor_ev, weapon_ev, armor_tier_avg, weapon_tier_avg)
            
            # Mostrar resumen
            st.subheader("📊 Resumen Completo")
            
            # Resumen estadístico
            stats_data = []
            for code, stats in historical_stats.items():
                stats_data.append({
                    "Item": code,
                    "Transacciones": stats["count"],
                    "Precio Mínimo": stats["min_price"],
                    "Precio Promedio": stats["mean_price"],
                    "Precio Máximo": stats["max_price"]
                })
            
            if stats_data:
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, hide_index=True)

with col2:
    if st.button("🔥 Actualizar Ofertas y Bargains", use_container_width=True):
        if not st.session_state.historical_stats:
            st.warning("⚠️ Primero actualiza los datos históricos")
            st.stop()
            
        with st.spinner("Buscando ofertas activas y bargains..."):
            bargains = []
            progress_bar = st.progress(0)
            total_codes = len(CODES)
            
            for i, code in enumerate(CODES):
                # Obtener todas las ofertas activas
                offers = fetch_active_offers(
                    code, 
                    pages=offer_pages, 
                    limit=offer_limit
                )
                
                # Si no hay ofertas, continuar
                if not offers:
                    continue
                
                # Actualizar precio más bajo
                st.session_state.active_lowest_prices[code] = min(o['price'] for o in offers)
                
                # Evaluar cada oferta
                stats = st.session_state.historical_stats.get(code)
                if not stats:
                    continue
                    
                for offer in offers:
                    evaluation = evaluate_offer(offer, stats)
                    if evaluation and evaluation['composite_score'] > 0.4:
                        # Formatear nombre para mostrar
                        item_type = code
                        display_name = item_type.capitalize()
                        if item_type[-1].isdigit():
                            display_name = f"{item_type[:-1].capitalize()} T{item_type[-1]}"
                        
                        # Añadir a bargains
                        bargains.append({
                            **evaluation,
                            "display_name": display_name,
                            "code": code
                        })
                
                progress_bar.progress((i+1) / total_codes)
                time.sleep(0.05)
            
            bargains = sorted(bargains, key=lambda x: x["composite_score"], reverse=True)
            st.session_state.bargains = bargains
            st.success(f"✅ Encontradas {len(bargains)} ofertas interesantes")

# Mostrar resultados en pestañas
tab1, tab2, tab3 = st.tabs(["🔥 Ofertas", "🔄 Mergear", "🎁 Loot Cases"])

with tab1:
    if st.session_state.bargains:
        st.header("🔥 Ofertas Destacadas")
        
        # Preparar datos para tabla
        df = pd.DataFrame(st.session_state.bargains)
        
        # Formatear columnas
        df['price'] = df['price'].apply(lambda x: f"${x:,.2f}")
        df['current_ppp'] = df['current_ppp'].apply(lambda x: f"${x:.2f}/pt")
        
        # Mostrar tabla
        st.dataframe(df[[
            'display_name', 'price', 'total_skill', 'current_ppp', 
            'percentile_rank', 'rating'
        ]].rename(columns={
            'display_name': 'Ítem',
            'price': 'Precio',
            'total_skill': 'Skill Total',
            'current_ppp': 'PPP Actual',
            'percentile_rank': 'Percentil',
            'rating': 'Valoración'
        }), hide_index=True, height=500)
        
        # Gráfico de distribución de puntajes
        st.subheader("📈 Distribución de Valoraciones")
        if len(df) > 1:
            fig = px.histogram(df, x='composite_score', nbins=20, 
                              title='Distribución de Puntajes Compuestos')
            st.plotly_chart(fig)
        
        # Mostrar detalles expandibles
        st.header("🔍 Detalles de Ofertas")
        for b in st.session_state.bargains:
            with st.expander(f"{b['display_name']} - {b['rating']}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Precio", f"${b['price']:,.2f}")
                col2.metric("Skill Total", b['total_skill'])
                col3.metric("PPP Actual", f"${b['current_ppp']:.2f}/pt")
                
                col1, col2 = st.columns(2)
                col1.metric("Evaluación Percentil", b['percentile_rank'])
                col2.metric("Puntaje Compuesto", f"{b['composite_score']:.2f}/1.0")
                
                st.progress(min(1.0, max(0.0, b["composite_score"])), 
                           text=f"Puntaje: {b['composite_score']:.2f}")

    elif st.session_state.historical_stats:
        st.info("ℹ️ Actualiza la búsqueda de ofertas para ver resultados")
    else:
        st.info("🔄 Comienza actualizando los datos históricos")

with tab2:
    st.header("🔄 Análisis de Viabilidad para Mergear")
    
    if st.session_state.merge_analysis:
        # Preparar datos para tabla
        df_merge = pd.DataFrame(st.session_state.merge_analysis)
        
        # Formatear columnas
        df_merge['current_lowest'] = df_merge['current_lowest'].apply(lambda x: f"${x:,.2f}" if x else "N/A")
        df_merge['merge_cost'] = df_merge['merge_cost'].apply(lambda x: f"${x:,.2f}")
        df_merge['total_merge_cost'] = df_merge['total_merge_cost'].apply(lambda x: f"${x:,.2f}" if x else "N/A")
        df_merge['next_ref_price'] = df_merge['next_ref_price'].apply(lambda x: f"${x:,.2f}" if x else "N/A")
        df_merge['savings'] = df_merge['savings'].apply(lambda x: f"${x:,.2f}" if x else "N/A")
        df_merge['savings_pct'] = df_merge['savings_pct'].apply(lambda x: f"{x:.1f}%" if x else "N/A")
        
        # Determinar viabilidad visual
        df_merge['viable'] = df_merge['viable'].apply(lambda x: "✅ Sí" if x else "❌ No")
        
        # Mostrar tabla
        st.dataframe(df_merge[[
            'item_type', 'current_tier', 'next_tier', 'current_lowest', 'merge_cost', 'total_merge_cost',
            'next_ref_price', 'viable', 'savings', 'savings_pct'
        ]].rename(columns={
            'item_type': 'Tipo de Item',
            'current_tier': 'Tier Actual',
            'next_tier': 'Tier Siguiente',
            'current_lowest': 'Precio Más Bajo Actual',
            'merge_cost': 'Costo de Merge',
            'total_merge_cost': 'Costo Total',
            'next_ref_price': 'Precio Referencia Tier+1',
            'viable': '¿Viable?',
            'savings': 'Ahorro/Pérdida',
            'savings_pct': '% Diferencia'
        }), hide_index=True)
        
        # Explicación
        st.subheader("📝 Explicación")
        st.markdown("""
        **¿Cómo se calcula la viabilidad?**
        - Se toma el precio más bajo actual para el tipo y tier
        - Se multiplica por 3 (items necesarios para merge)
        - Se suma el costo de merge (T1: $2, T2: $4, T3: $8, T4: $16, T5: $32)
        - Se compara con el precio de referencia del tier superior
        - **Precio de referencia**: Máx(Promedio histórico, Oferta más baja actual)
        
        **Interpretación:**
        - ✅ Sí: Mergear es más económico que comprar directamente
        - ❌ No: Es más barato comprar el item del tier superior
        """)
        
        # Recomendaciones
        st.subheader("💡 Recomendaciones")
        viable_merges = [m for m in st.session_state.merge_analysis if m['viable']]
        if viable_merges:
            st.success("Mergear es viable para los siguientes tipos:")
            for vm in viable_merges:
                st.info(f"- {vm['item_type'].capitalize()} (T{vm['current_tier']}): Ahorro del {vm['savings_pct']:.1f}%")
        else:
            st.warning("Actualmente no es viable mergear ningún tipo de item")
            
    elif st.session_state.historical_stats:
        st.info("ℹ️ Actualiza los datos para ver el análisis de merge")
    else:
        st.info("🔄 Comienza actualizando los datos")

with tab3:
    st.header("🎁 Valor Esperado de un Loot Case")
    
    if st.session_state.loot_case_value is not None:
        # Desempaquetar los valores
        expected_value, armor_ev, weapon_ev, armor_tier_avg, weapon_tier_avg = st.session_state.loot_case_value
        
        # Mostrar valor esperado
        st.metric("Valor Esperado", f"${expected_value:,.2f}")
        
        # Mostrar valores por tipo
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Valor Esperado Armaduras", f"${armor_ev:,.2f}")
        with col2:
            st.metric("Valor Esperado Armas", f"${weapon_ev:,.2f}")
        
        # Explicación de cálculo
        st.subheader("📊 Cálculo Detallado")
        st.markdown("""
        **Probabilidades de Drop:**
        - **Armaduras:**
          - Tier 1: 62%
          - Tier 2: 30%
          - Tier 3: 7.1%
          - Tier 4: 0.85%
          - Tier 5: 0.05%
        
        - **Armas:**
          - Tier 1: 62%
          - Tier 2: 30%
          - Tier 3: 7.1%
          - Tier 4: 0.85%
          - Tier 5: 0.04% (Tank)
          - Tier 6: 0.01% (Jet)
        
        **Método de cálculo:**
        - 50% de probabilidad de obtener armadura
        - 50% de probabilidad de obtener arma
        - Valor Esperado = 0.5 × Valor_Armaduras + 0.5 × Valor_Armas
        - **Precio Referencia**: Máx(Promedio histórico, Oferta más baja actual)
        """)
        
        # Mostrar desglose por tier
        st.subheader("🧮 Desglose por Tier")
        
        # Crear tabla de desglose para armaduras
        st.markdown("**Armaduras**")
        armor_breakdown = []
        for tier_num in range(1, 6):
            rate = LOOT_CASE_ARMOR_RATES[f"T{tier_num}"]
            price = armor_tier_avg.get(tier_num, 0)
            contribution = rate * price
            
            armor_breakdown.append({
                "Tier": f"T{tier_num}",
                "Probabilidad": f"{rate*100:.2f}%",
                "Precio Referencia": f"${price:,.2f}",
                "Contribución": f"${contribution:,.2f}"
            })
        
        # Agregar total armaduras
        armor_breakdown.append({
            "Tier": "TOTAL ARMADURAS",
            "Probabilidad": "100%",
            "Precio Referencia": "-",
            "Contribución": f"${armor_ev:,.2f}"
        })
        
        st.dataframe(pd.DataFrame(armor_breakdown), hide_index=True)
        
        # Crear tabla de desglose para armas
        st.markdown("**Armas**")
        weapon_breakdown = []
        for tier_num in range(1, 7):
            rate_key = f"T{tier_num}"
            if tier_num <= 5:
                rate = LOOT_CASE_WEAPON_RATES.get(rate_key, 0)
            else:
                rate = LOOT_CASE_WEAPON_RATES.get("T6", 0)  # Jet es T6
                
            price = weapon_tier_avg.get(tier_num, 0)
            contribution = rate * price
            
            weapon_breakdown.append({
                "Tier": f"T{tier_num}",
                "Probabilidad": f"{rate*100:.2f}%",
                "Precio Referencia": f"${price:,.2f}",
                "Contribución": f"${contribution:,.2f}"
            })
        
        # Agregar total armas
        weapon_breakdown.append({
            "Tier": "TOTAL ARMAS",
            "Probabilidad": "100%",
            "Precio Referencia": "-",
            "Contribución": f"${weapon_ev:,.2f}"
        })
        
        st.dataframe(pd.DataFrame(weapon_breakdown), hide_index=True)
        
        # Recomendación
        st.subheader("💡 Recomendación")
        if expected_value > 1.0:
            st.success(f"¡Abrir loot cases es rentable! (Valor esperado: ${expected_value:,.2f})")
        else:
            st.warning(f"Actualmente no es rentable abrir loot cases (Valor esperado: ${expected_value:,.2f})")
            
    elif st.session_state.historical_stats:
        st.info("ℹ️ Actualiza los datos para ver el valor del loot case")
    else:
        st.info("🔄 Comienza actualizando los datos")

st.caption("⚠️ Solo se consideran items con condición 100% en todas las consultas")


