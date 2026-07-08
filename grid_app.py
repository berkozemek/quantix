import pandas as pd
import numpy as np
import yfinance as yf
import time

# =============================================================================
# --- 1. SMC ANALİZ MOTORU ---
# =============================================================================

def detect_swing_points(df, window=2):
    df['Swing_High'] = np.nan
    df['Swing_Low'] = np.nan
    for i in range(window, len(df) - window):
        if df['high'].iloc[i] == df['high'].iloc[i - window : i + window + 1].max():
            df.loc[df.index[i], 'Swing_High'] = df['high'].iloc[i]
        if df['low'].iloc[i] == df['low'].iloc[i - window : i + window + 1].min():
            df.loc[df.index[i], 'Swing_Low'] = df['low'].iloc[i]
    return df

def find_bos_choch(df):
    df['BOS'] = 0    
    df['CHoCH'] = 0  
    last_high = None
    last_low = None
    current_trend = 1 

    for i in range(len(df)):
        if not pd.isna(df['Swing_High'].iloc[i]):
            last_high = df['Swing_High'].iloc[i]
        if not pd.isna(df['Swing_Low'].iloc[i]):
            last_low = df['Swing_Low'].iloc[i]
            
        close_price = df['close'].iloc[i]
        
        if current_trend == 1:
            if last_high and close_price > last_high:
                df.loc[df.index[i], 'BOS'] = 1
                last_high = None
            elif last_low and close_price < last_low:
                df.loc[df.index[i], 'CHoCH'] = -1
                current_trend = -1
                last_low = None
        elif current_trend == -1:
            if last_low and close_price < last_low:
                df.loc[df.index[i], 'BOS'] = -1
                last_low = None
            elif last_high and close_price > last_high:
                df.loc[df.index[i], 'CHoCH'] = 1
                current_trend = 1
                last_high = None
    return df

def detect_fvg_zones(df):
    df['Active_Demand_FVGs'] = [[] for _ in range(len(df))]
    open_demand_fvgs = [] 
    
    for i in range(2, len(df)):
        current_low = df['low'].iloc[i]
        prev2_high = df['high'].iloc[i - 2]
        
        if prev2_high < current_low:
            open_demand_fvgs.append({'top': current_low, 'bottom': prev2_high, 'index': i})
            
        open_demand_fvgs = [fvg for fvg in open_demand_fvgs if current_low > fvg['bottom']]
        df.at[df.index[i], 'Active_Demand_FVGs'] = list(open_demand_fvgs)
    return df

# =============================================================================
# --- 2. ANA ÇALIŞTIRMA SCRIPT'İ (YAKINDAN UZAĞA AKILLI SIRALAMA) ---
# =============================================================================

if __name__ == "__main__":
    BIST_100_HAVUZU = [
        "AEFES.IS", "AGROT.IS", "AKBNK.IS", "AKCNS.IS", "AKFGY.IS", "AKFYE.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS", "ALBRK.IS",
        "ALFAS.IS", "ANACM.IS", "ANSGR.IS", "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "BERA.IS", "BIMAS.IS", "BRSAN.IS", "BRYAT.IS",
        "BUCIM.IS", "CCOLA.IS", "CATES.IS", "CEMTS.IS", "CIMSA.IS", "CWENE.IS", "DOAS.IS", "DOHOL.IS", "ECILC.IS", "ECZYT.IS",
        "EGEEN.IS", "EKGYO.IS", "ENJSA.IS", "ENKAI.IS", "EREGL.IS", "EUPWR.IS", "FROTO.IS", "GARAN.IS", "GESAN.IS", "GUBRF.IS",
        "GWIND.IS", "HALKB.IS", "HEKTS.IS", "IMASM.IS", "ISCTR.IS", "ISGYO.IS", "ISMEN.IS", "IZMDC.IS", "KCAER.IS", "KCHOL.IS",
        "KLSER.IS", "KONTR.IS", "KONYA.IS", "KORDS.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "LMKDC.IS", "MAVI.IS", "MGROS.IS",
        "MIATK.IS", "NETAS.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PEKGY.IS", "PETKM.IS", "PGSUS.IS", "QUAGR.IS", "REEDR.IS",
        "SAHOL.IS", "SASA.IS", "SAYAS.IS", "SDTTR.IS", "SISE.IS", "SKBNK.IS", "SMRTG.IS", "SOKM.IS", "TABGD.IS", "TARKM.IS",
        "TATEN.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS", "TTRAK.IS", "TUPRS.IS", "TURSG.IS",
        "ULKER.IS", "VAKBN.IS", "VESBE.IS", "VESTL.IS", "YEOTK.IS", "YKBNK.IS", "ZOREN.IS"
    ]
    
    print("==================================================")
    print("🎯 BIST 100 AKILLI SIRALAMALI SMC RADARI 🎯")
    print("==================================================")
    print(f"📋 BIST 100'deki {len(BIST_100_HAVUZU)} hisse paket halinde çekiliyor...")
    
    # Toplu indirme
    toplu_data = yf.download(tickers=BIST_100_HAVUZU, period="1mo", interval="1h", group_by='ticker', progress=True)
    
    print("\n🔎 Veriler RAM belleğe alındı. Yakından uzağa pusu analizi yapılıyor...\n")
    
    BULUNAN_SINYALLER = []
    total_balance = 100000.0  # Hesaplama için baz bakiye
    
    for hisse in BIST_100_HAVUZU:
        try:
            if hisse in toplu_data.columns.levels[0]:
                df_raw = toplu_data[hisse].dropna()
                if df_raw.empty:
                    continue
                    
                df = df_raw.copy()
                df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                }, inplace=True)
                
                df = detect_swing_points(df, window=2)
                df = find_bos_choch(df)
                df = detect_fvg_zones(df)
                
                recent_candles = df.tail(40)
                latest_candle = df.iloc[-1]
                
                choch_detected = (recent_candles['CHoCH'] == 1).any()
                
                if choch_detected:
                    active_demands = latest_candle['Active_Demand_FVGs']
                    
                    if len(active_demands) > 0:
                        target_fvg = active_demands[-1]
                        entry_price = target_fvg['top']
                        current_price = latest_candle['close']
                        
                        # Anlık fiyat, pusu fiyatının altına indiyse o sinyali işleme almıyoruz (Kutu içi pusu için)
                        if current_price >= entry_price:
                            # Giriş seviyesine olan yüzde uzaklığı hesapla
                            mesafe_yuzde = ((current_price - entry_price) / entry_price) * 100
                            
                            stop_loss = target_fvg['bottom'] * 0.995
                            take_profit = entry_price + (abs(entry_price - stop_loss) * 3)
                            
                            max_tl_risk = total_balance * 0.01
                            price_risk = entry_price - stop_loss
                            lot_size = max_tl_risk / price_risk if price_risk > 0 else 0
                            
                            # Bilgileri listeye at
                            BULUNAN_SINYALLER.append({
                                'symbol': hisse.replace(".IS", ""),
                                'entry': entry_price,
                                'current': current_price,
                                'stop': stop_loss,
                                'tp': take_profit,
                                'lot': int(lot_size),
                                'risk': max_tl_risk,
                                'mesafe': mesafe_yuzde
                            })
        except:
            continue
            
    # =============================================================================
    # 🔥 EN KRİTİK NOKTA: MESAFEYE GÖRE YAKINDAN UZAĞA SIRALAMA 🔥
    # =============================================================================
    BULUNAN_SINYALLER = sorted(BULUNAN_SINYALLER, key=lambda x: x['mesafe'])
    
    # Sıralı listeyi ekrana kurumsal panellerle basıyoruz
    for idx, sny in enumerate(BULUNAN_SINYALLER, 1):
        print("\n" + "🇹🇷" * 20)
        print(f"🚨 [SIRA #{idx} | EN YAKIN PUSU] -> {sny['symbol']}")
        print("🇹🇷" * 20)
        print(f"📏 Pusu Girişine Uzaklık     :  %{sny['mesafe']:.2f} (Çok Yakın!)")
        print(f"📉 Anlık Cari Fiyat          :  {sny['current']:.2f} TL")
        print("-" * 45)
        print(f"🟢 LİMİT GİRİŞ (Alış Fiyatı) :  {sny['entry']:.2f} TL")
        print(f"🔴 ZARAR DURDUR (Stop-Loss) :  {sny['stop']:.2f} TL")
        print(f"🎯 KÂR AL (Take-Profit)      :  {sny['tp']:.2f} TL [1:3 R/R]")
        print("-" * 45)
        print(f"💰 100K Kasada %1 Risk İçin  :  {sny['lot']} Adet Hisse")
        print(f"💼 Risk Edilen Net Tutar     :  {sny['risk']:.2f} TL")
        print("🇹🇷" * 20 + "\n")
        
    print("\n==================================================")
    print(f"✅ RADAR TARAMASI TAMAMLANDI!")
    print(f"🎯 Toplam {len(BULUNAN_SINYALLER)} hisse, 'Anlık Fiyata En Yakından En Uzağa' kusursuzca dizildi.")
    print("==================================================")
