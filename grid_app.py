import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta, timezone

# Sayfa Yapılandırması
st.set_page_config(layout="wide")

# --- 1. CORE FONKSİYONLAR ---
@st.cache_data
def fetch_historical_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def get_current_live_price(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        live_df = ticker_obj.history(period="1d", interval="1m")
        if not live_df.empty:
            return float(live_df['Close'].iloc[-1])
    except Exception as e:
        return None
    return None

def calculate_dynamic_bounds(df, lookback_periods=30):
    analysis_window = df.head(lookback_periods)
    lowest_low = float(analysis_window['Low'].min())
    highest_high = float(analysis_window['High'].max())
    return lowest_low * 0.95, highest_high * 1.05

def calculate_screener_metrics(df):
    if len(df) < 20:
        return None, None
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    up_move = high.diff()
    down_move = low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-9)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    
    return float(rsi.iloc[-1]), float(adx.iloc[-1])

# --- BACKTEST MOTORU ---
def run_grid_bot_simulation_v2(df, p_min, p_max, grid_count, total_capital, stop_loss_pct, stop_loss_type):
    if df.empty:
        return None, None, {"bot_return": -999, "hodl_return": -999, "total_value": 0, "grid_profits": 0, "trade_count": 0, "is_stop": True}
    grids = np.linspace(p_min, p_max, grid_count)
    step_size = grids[1] - grids[0]
    start_price = float(df['Close'].iloc[0])
    
    cash = total_capital / 2
    asset_qty = (total_capital / 2) / start_price
    grid_profits = 0
    trade_log = []
    stop_loss_triggered = False
    initial_stop_level = p_min * (1 - stop_loss_pct)
    
    buy_orders = {g: True for g in grids if g < start_price}
    sell_orders = {g: True for g in grids if g > start_price}
    qty_per_grid = (total_capital / 2) / (grid_count / 2) / start_price
    channel_history = []

    for index, row in df.iterrows():
        current_price = float(row['Close'])
        low_p = float(row['Low'])
        high_p = float(row['High'])
        channel_history.append({"Date": index, "p_min": p_min, "p_max": p_max})
        
        current_stop_level = p_min * (1 - stop_loss_pct) if stop_loss_type == "İz Süren (Trailing)" else initial_stop_level
        if low_p <= current_stop_level:
            cash += asset_qty * current_stop_level
            trade_log.append({"Date": index, "Type": "STOP-LOSS", "Price": current_stop_level, "Qty": asset_qty, "Cash": cash, "Asset": 0.0})
            asset_qty = 0
            stop_loss_triggered = True
            break
            
        if high_p >= p_max:
            p_min += step_size
            p_max += step_size
            grids = np.linspace(p_min, p_max, grid_count)
            buy_orders = {g: True for g in grids if g < current_price}
            sell_orders = {g: True for g in grids if g > current_price}
            trade_log.append({"Date": index, "Type": "TRAILING_UP", "Price": current_price, "Qty": 0, "Cash": cash, "Asset": asset_qty})
            continue 
            
        for g in list(buy_orders.keys()):
            if buy_orders[g] and low_p <= g:
                if cash >= g * qty_per_grid:
                    cash -= g * qty_per_grid
                    asset_qty += qty_per_grid
                    buy_orders[g] = False 
                    target_sell_grid = g + step_size
                    if target_sell_grid <= p_max:
                        sell_orders[target_sell_grid] = True
                    trade_log.append({"Date": index, "Type": "BUY", "Price": g, "Qty": qty_per_grid, "Cash": cash, "Asset": asset_qty})
        
        for g in list(sell_orders.keys()):
            if sell_orders[g] and high_p >= g:
                if asset_qty >= qty_per_grid:
                    cash += g * qty_per_grid
                    asset_qty -= qty_per_grid
                    sell_orders[g] = False 
                    grid_profits += (step_size * qty_per_grid)
                    target_buy_grid = g - step_size
                    if target_buy_grid >= p_min:
                        buy_orders[target_buy_grid] = True
                    trade_log.append({"Date": index, "Type": "SELL", "Price": g, "Qty": qty_per_grid, "Cash": cash, "Asset": asset_qty})

    final_price = float(df['Close'].iloc[-1]) if not stop_loss_triggered else current_stop_level
    total_final_value = cash + (asset_qty * final_price)
    metrics = {
        "bot_return": ((total_final_value - total_capital) / total_capital) * 100,
        "hodl_return": (((total_capital / start_price) * float(df['Close'].iloc[-1]) - total_capital) / total_capital) * 100,
        "total_value": total_final_value, "grid_profits": grid_profits, "trade_count": len(trade_log), "is_stop": stop_loss_triggered
    }
    return pd.DataFrame(trade_log), pd.DataFrame(channel_history), metrics

# --- 2. STREAMLIT OTURUM HAFIZASI ---
if "bot_active" not in st.session_state:
    st.session_state.bot_active = False
if "live_logs" not in st.session_state:
    st.session_state.live_logs = []
if "live_portfolio" not in st.session_state:
    st.session_state.live_portfolio = {}

# --- 3. ARAYÜZ TASARIMI ---
st.title("📊 Gelişmiş Algoritmik Grid Bot İstasyonu")

tab1, tab2, tab3 = st.tabs(["🔍 Tarihsel Backtest Analizi", "⚡ Canlı Sanal Trading (Paper Trading)", "🎯 BIST Grid Tarayıcı"])

st.sidebar.header("🔧 Genel Ayarlar")
hisse_kodu = st.sidebar.text_input("Hisse Kodu (yfinance)", value="GARAN.IS")
sermaye = st.sidebar.number_input("Sermaye (TL)", value=50000, step=5000)
grid_sayisi = st.sidebar.slider("Izgara Sayısı", 4, 30, 12)
stop_orani = st.sidebar.slider("Stop-Loss (%)", 1.0, 10.0, 2.0, 0.5) / 100.0
stop_tipi = st.sidebar.selectbox("Stop Tipi", ["İlk Güne Sabitlenmiş", "İz Süren (Trailing)"])

# --- TAB 1: BACKTEST & OTOMATİK OPTİMİZASYON ---
with tab1:
    st.header("Geçmiş Dönem Simülasyonu ve Optimizasyon")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        start_date = st.date_input("Başlangıç Tarihi", pd.to_datetime("2026-04-01"), key="b_start")
    with col_t2:
        end_date = st.date_input("Bitiş Tarihi", pd.to_datetime("2026-07-01"), key="b_end")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        manual_run = st.button("Manual Ayarlarla Simülasyonu Başlat", use_container_width=True)
    with col_btn2:
        auto_run = st.button("🤖 EN İYİ AYARLARI OTOMATİK BUL (AUTO-OPTIMIZE)", use_container_width=True)
        
    veri = fetch_historical_data(hisse_kodu, start_date, end_date)
    
    # SENARYO 1: OTOMATİK OPTİMİZASYON BUTONUNA BASILDIYSA
    if auto_run and not veri.empty:
        a_min, a_max = calculate_dynamic_bounds(veri, lookback_periods=30)
        
        best_return = -999
        best_grid = 12
        best_stop = 0.02
        
        grid_testleri = [8, 10, 12, 14, 16, 18, 20]
        stop_testleri = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
        
        with st.spinner("🤖 Algoritma tüm olasılıkları deniyor, lütfen bekleyin..."):
            for g_t in grid_testleri:
                for s_t in stop_testleri:
                    _, _, m_t = run_grid_bot_simulation_v2(veri, a_min, a_max, g_t, sermaye, s_t, stop_tipi)
                    if m_t["bot_return"] > best_return:
                        best_return = m_t["bot_return"]
                        best_grid = g_t
                        best_stop = s_t
        
        st.success(f"🎯 HİSSE İÇİN EN İDEAL AYARLAR BULUNDU!\n\n"
                   f"*   **En İdeal Izgara Sayısı (Grid):** {best_grid}\n"
                   f"*   **En İdeal Stop-Loss Oranı:** %{best_stop*100:.2f}\n"
                   f"*   **Bu Ayarlarla Alınan Maksimum Net Kâr:** %{best_return:.2f}")
        
        st.info("ℹ️ Sol menüdeki kaydırıcıları bu ideal rakamlara getirerek grafiği detaylıca inceleyebilirsin ortak!")

    # SENARYO 2: MANUEL ÇALIŞTIRMA VEYA GRAFİK GÖSTERİMİ
    if manual_run and not veri.empty:
        a_min, a_max = calculate_dynamic_bounds(veri, lookback_periods=30)
        trade_log_df, channel_df, metrics = run_grid_bot_simulation_v2(veri, a_min, a_max, grid_sayisi, sermaye, stop_orani, stop_tipi)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bot Net Getiri", f"% {metrics['bot_return']:.2f}")
        c2.metric("Al-Bekle (HODL)", f"% {metrics['hodl_return']:.2f}")
        c3.metric("Toplam İşlem", f"{metrics['trade_count']} Adet")
        c4.metric("Saf Izgara Kârı", f"{metrics['grid_profits']:,.2f} TL")
        
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(veri.index, veri['Close'], color='black', alpha=0.5, label='Fiyat')
        if not channel_df.empty:
            ax.plot(channel_df['Date'], channel_df['p_max'], 'r--', alpha=0.5, label='Üst Sınır')
            ax.plot(channel_df['Date'], channel_df['p_min'], 'b--', alpha=0.5, label='Alt Sınır')
        st.pyplot(fig)

# --- TAB 2: CANLI SANAL TRADING ---
with tab2:
    st.header("⚡ Gerçek Zamanlı Sanal Piyasa Takibi")
    st.write("Bu mod, Yahoo Finance üzerinden anlık fiyatı sorgular ve açık ızgara emirlerinizi canlı simüle eder.")
    
    col_control1, col_control2 = st.columns(2)
    with col_control1:
        if st.button("🔴 CANLI BOTU BAŞLAT", use_container_width=True):
            current_p = get_current_live_price(hisse_kodu)
            if current_p:
                st.session_state.bot_active = True
                st.session_state.live_portfolio = {
                    "cash": sermaye / 2,
                    "asset_qty": (sermaye / 2) / current_p,
                    "p_min": current_p * 0.90,
                    "p_max": current_p * 1.10,
                    "start_price": current_p,
                    "qty_per_grid": (sermaye / 2) / (grid_sayisi / 2) / current_p
                }
                grids = np.linspace(st.session_state.live_portfolio["p_min"], st.session_state.live_portfolio["p_max"], grid_sayisi)
                st.session_state.live_portfolio["buy_orders"] = {g: True for g in grids if g < current_p}
                st.session_state.live_portfolio["sell_orders"] = {g: True for g in grids if g > current_p}
                st.session_state.live_portfolio["step_size"] = grids[1] - grids[0]
                st.session_state.live_logs = [{"Zaman": datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M:%S'), "Olay": "Bot Başlatıldı", "Fiyat": current_p}]
            else:
                st.error("Canlı fiyat çekilemedi. Piyasa kapalı veya kod hatalı olabilir.")
                
    with col_control2:
        if st.button("⏹️ BOTU DURDUR / SIFIRLA", use_container_width=True):
            st.session_state.bot_active = False
            st.session_state.live_portfolio = {}
            st.session_state.live_logs = []
            st.success("Canlı bot durduruldu ve portföy sıfırlandı.")

    if st.session_state.bot_active:
        p = st.session_state.live_portfolio
        current_p = get_current_live_price(hisse_kodu)
        if current_p:
            step = p["step_size"]
            qty = p["qty_per_grid"]
            
            for g in list(p["buy_orders"].keys()):
                if p["buy_orders"][g] and current_p <= g:
                    if p["cash"] >= g * qty:
                        p["cash"] -= g * qty
                        p["asset_qty"] += qty
                        p["buy_orders"][g] = False
                        if g + step <= p["p_max"]:
                            p["sell_orders"][g + step] = True
                        st.session_state.live_logs.append({"Zaman": datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M:%S'), "Olay": f"🛒 CANLI ALIM YAPILDI ({qty:.2f} Adet)", "Fiyat": g})
            
            for g in list(p["sell_orders"].keys()):
                if p["sell_orders"][g] and current_p >= g:
                    if p["asset_qty"] >= qty:
                        p["cash"] += g * qty
                        p["asset_qty"] -= qty
                        p["sell_orders"][g] = False
                        if g - step >= p["p_min"]:
                            p["buy_orders"][g - step] = True
                        st.session_state.live_logs.append({"Zaman": datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M:%S'), "Olay": f"💰 CANLI SATIM YAPILDI ({qty:.2f} Adet)", "Fiyat": g})
            
            current_total_value = p["cash"] + (p["asset_qty"] * current_p)
            net_profit_loss = current_total_value - sermaye
            
            c_live1, c_live2, c_live3, c_live4 = st.columns(4)
            c_live1.metric("Anlık Canlı Fiyat", f"{current_p:.2f} TL")
            c_live2.metric("Toplam Portföy Değeri", f"{current_total_value:,.2f} TL")
            c_live3.metric("Kasa Nakit Durumu", f"{p['cash']:,.2f} TL")
            c_live4.metric("Net Kar / Zarar", f"{net_profit_loss:,.2f} TL", delta=f"{((current_total_value-sermaye)/sermaye)*100:.2f}%")
            
            st.subheader("📌 Aktif Bekleyen Limit Emir Çizgileri")
            col_b, col_s = st.columns(2)
            with col_b:
                st.write("🟢 **Bekleyen Alım Seviyeleri (Buy Limit)**")
                st.write([round(k, 2) for k, v in p["buy_orders"].items() if v])
            with col_s:
                st.write("🔴 **Bekleyen Satım Seviyeleri (Sell Limit)**")
                st.write([round(k, 2) for k, v in p["sell_orders"].items() if v])

            st.subheader("📰 Canlı İşlem ve Akış Günlüğü")
            if st.session_state.live_logs:
                live_df = pd.DataFrame(st.session_state.live_logs).iloc[::-1]
                st.dataframe(live_df, use_container_width=True)
            else:
                st.info("Henüz bir akış veya işlem gerçekleşmedi. Bot veri bekliyor...")
            
            time.sleep(5)
            st.rerun()

# --- TAB 3: BIST GRID TARAYICI ---
with tab3:
    st.header("🎯 Grid Bot İçin En İdeal Hisseleri Bul")
    bist_sepeti = ["THYAO.IS", "AKBNK.IS", "EREGL.IS", "TUPRS.IS", "ASELS.IS", "BIMAS.IS", "GARAN.IS", "KCHOL.IS", "ISCTR.IS", "YKBNK.IS", "SAHOL.IS", "SISE.IS", "PETKM.IS"]
    
    if st.button("🔍 BIST Pazar Taramasını Başlat", use_container_width=True):
        rapor_verileri = []
        progress_bar = st.progress(0)
        durum_yazisi = st.empty()
        
        bitis_tarihi = datetime.now(timezone(timedelta(hours=3)))
        baslangic_tarihi = bitis_tarihi - timedelta(days=45)
        
        for idx, s in enumerate(bist_sepeti):
            durum_yazisi.write(f"⏳ {s} analiz ediliyor...")
            progress_bar.progress((idx + 1) / len(bist_sepeti))
            try:
                hisse_veri = yf.download(s, start=baslangic_tarihi.strftime('%Y-%m-%d'), end=bitis_tarihi.strftime('%Y-%m-%d'), progress=False)
                if isinstance(hisse_veri.columns, pd.MultiIndex):
                    hisse_veri.columns = hisse_veri.columns.get_level_values(0)
                if not hisse_veri.empty and len(hisse_veri) >= 20:
                    rsi_val, adx_val = calculate_screener_metrics(hisse_veri)
                    son_fiyat = float(hisse_veri['Close'].iloc[-1])
                    if rsi_val is not None and adx_val is not None:
                        if adx_val < 20 and (42 <= rsi_val <= 58): skor = "🔥 ÇOK YÜKSEK (Mükemmel Yatay)"
                        elif adx_val < 24 and (40 <= rsi_val <= 60): skor = "🟢 YÜKSEK (Güvenli Kanal)"
                        elif adx_val < 28: skor = "🟡 ORTA (Hafif Trendli)"
                        else: skor = "🔴 ZAYIF (Sert Trend Var!)"
                            
                        rapor_verileri.append({"Hisse Kodu": s, "Son Fiyat (TL)": round(son_fiyat, 2), "ADX (Trend Gücü)": round(adx_val, 2), "RSI (14)": round(rsi_val, 2), "Grid Uygunluk Skoru": skor})
            except Exception as e: continue
        progress_bar.empty(); durum_yazisi.empty()
        if rapor_verileri:
            rapor_df = pd.DataFrame(rapor_verileri).sort_values(by=["ADX (Trend Gücü)"], ascending=True)
            st.subheader("📋 Piyasa Tarama Sonuç Raporu")
            st.dataframe(rapor_df, use_container_width=True, hide_index=True)
