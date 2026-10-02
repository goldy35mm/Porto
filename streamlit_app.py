
import os
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime

# ============================================================
# ADVANCED PORTFOLIO DASHBOARD | ZAPI IDX
# ============================================================
st.set_page_config(
    page_title="Leo | Advanced Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- STYLE --------------------
st.markdown("""
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(120,120,120,0.07);
        border: 1px solid rgba(120,120,120,0.15);
        padding: 14px 16px;
        border-radius: 12px;
    }
    div[data-testid="stMetricValue"] {font-size: 1.45rem;}
    .small-note {font-size: 0.82rem; opacity: 0.75;}
</style>
""", unsafe_allow_html=True)

# -------------------- PORTFOLIO --------------------
# Jumlah lot dan average sesuai kode portofolio yang kamu kirim.
HOLDINGS = {
    "ADRO": {"lots": 261, "avg": 2066.36},
    "ADMR": {"lots": 355, "avg": 1589.32},
    "ESSA": {"lots": 1000, "avg": 612.68},
    "MEDC": {"lots": 82, "avg": 1471.10},
}
CASH = 20027683
TARGET_EQUITY = 200_000_000
SHARES_PER_LOT = 100

# -------------------- API --------------------
try:
    API_KEY = st.secrets["ZAPI_API_KEY"]
except Exception:
    API_KEY = os.getenv("ZAPI_API_KEY", "")

URL = "https://api.zapi.ink/v1/finance:idx/stock-history"
HEADERS = {
    "x-api-key": API_KEY,
    "accept": "application/json"
}

# -------------------- HELPERS --------------------
def rupiah(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"Rp {value:,.0f}"

def pct(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:+.2f}%"

def extract_rows(obj):
    if isinstance(obj, list):
        if not obj:
            return []
        if all(isinstance(x, dict) for x in obj):
            return obj
        for item in obj:
            found = extract_rows(item)
            if found:
                return found
    elif isinstance(obj, dict):
        for key in (
            "data", "results", "result", "items",
            "rows", "stock_history", "history"
        ):
            if key in obj:
                found = extract_rows(obj[key])
                if found:
                    return found
    return []

def normalize_history(rows):
    df = pd.DataFrame(rows)
    df.columns = [str(c).lower().strip() for c in df.columns]

    aliases = {
        "date": ["date", "datetime", "timestamp",
                 "trading_date", "trade_date", "time"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "price", "last"],
        "volume": ["volume", "vol", "v"],
        "value": ["value", "turnover"]
    }

    rename = {}
    for target, candidates in aliases.items():
        for col in df.columns:
            if col in candidates:
                rename[col] = target
                break
    df = df.rename(columns=rename)

    if "close" not in df.columns:
        raise ValueError(
            "Kolom harga penutupan tidak ditemukan. Kolom API: "
            + ", ".join(df.columns)
        )

    if "date" in df.columns:
        raw_date = df["date"]
        # Coba parsing tanggal biasa terlebih dahulu.
        parsed = pd.to_datetime(raw_date, errors="coerce")
        # Jika mayoritas gagal dan data berupa Unix timestamp.
        if parsed.notna().sum() < max(1, len(raw_date) // 2):
            numeric_date = pd.to_numeric(raw_date, errors="coerce")
            if numeric_date.notna().any():
                unit = "ms" if numeric_date.dropna().median() > 10**11 else "s"
                parsed = pd.to_datetime(
                    numeric_date, unit=unit, errors="coerce", utc=True
                ).dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
        elif getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
        df["date"] = parsed
        df = df.dropna(subset=["date"])

    for col in ["open", "high", "low", "close", "volume", "value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])
    if "date" in df.columns:
        df = df.sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")
    return df.reset_index(drop=True)

def add_indicators(df):
    df = df.copy()
    close = df["close"]
    df["return"] = close.pct_change() * 100
    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI14"] = 100
    df.loc[(avg_loss == 0) & (avg_gain == 0), "RSI14"] = 50

    if all(c in df.columns for c in ["high", "low"]):
        prev_close = close.shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        df["ATR14"] = tr.ewm(
            alpha=1/14, min_periods=14, adjust=False
        ).mean()
    else:
        df["ATR14"] = close.diff().abs().rolling(14).mean()

    df["Volatility20"] = df["return"].rolling(20).std() * np.sqrt(20)
    df["Drawdown"] = (
        close / close.cummax() - 1
    ) * 100

    if "volume" in df.columns:
        df["VolumeMA20"] = df["volume"].rolling(20).mean()
        df["VolumeRatio"] = df["volume"] / df["VolumeMA20"].replace(0, np.nan)

    return df

# -------------------- DATA FETCH --------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock(code, start_date, end_date):
    params = {
        "code": code,
        "from": start_date,
        "to": end_date,
        "length": 2000
    }
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(
                URL, headers=HEADERS, params=params, timeout=(20, 60)
            )
            response.raise_for_status()
            rows = extract_rows(response.json())
            if not rows:
                raise ValueError("Respons API tidak berisi data historis")
            return add_indicators(normalize_history(rows))
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(str(last_error))

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("⚙️ Pengaturan")
    st.caption("ZAPI IDX • Cached 5 menit")
    start = st.date_input("Mulai data", value=date(2024, 1, 1))
    end = st.date_input("Akhir data", value=date.today())
    target = st.number_input(
        "Target ekuitas (Rp)",
        min_value=1_000_000,
        value=TARGET_EQUITY,
        step=5_000_000
    )
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption("Data harga mengikuti tanggal terakhir yang dikembalikan API.")
    st.caption("Forecast adalah skenario statistik historis, bukan kepastian.")

st.title("📈 Advanced Portfolio Dashboard")
st.caption("Leo | ADRO • ADMR • ESSA • MEDC | ZAPI IDX")

if not API_KEY:
    st.error(
        "API key belum ditemukan. Masukkan ZAPI_API_KEY pada "
        "Streamlit Cloud → App → Settings → Secrets."
    )
    st.stop()

if start > end:
    st.error("Tanggal mulai harus sebelum tanggal akhir.")
    st.stop()

# -------------------- FETCH --------------------
history, errors, prices, last_dates = {}, {}, {}, {}

with st.spinner("Mengambil data historis dari ZAPI..."):
    for code in HOLDINGS:
        try:
            df = fetch_stock(code, start.isoformat(), end.isoformat())
            if df.empty:
                raise ValueError("Data kosong")
            history[code] = df
            prices[code] = float(df.iloc[-1]["close"])
            last_dates[code] = (
                df.iloc[-1]["date"] if "date" in df.columns else None
            )
        except Exception as exc:
            errors[code] = str(exc)

for code, err in errors.items():
    st.warning(f"{code}: {err}")

if not history:
    st.error("Tidak ada data yang berhasil diambil dari ZAPI.")
    st.stop()

# -------------------- PORTFOLIO CALC --------------------
records = []
for code, info in HOLDINGS.items():
    qty = info["lots"] * SHARES_PER_LOT
    avg = info["avg"]
    price = prices.get(code)
    cost = qty * avg
    market = qty * price if price is not None else np.nan
    pnl = market - cost if pd.notna(market) else np.nan
    pnl_pct = pnl / cost * 100 if pd.notna(pnl) and cost else np.nan
    df = history.get(code)
    day_change = (
        float(df["return"].iloc[-1])
        if df is not None and len(df) > 1 and pd.notna(df["return"].iloc[-1])
        else np.nan
    )
    records.append({
        "Kode": code, "Lot": info["lots"], "Saham": qty,
        "Avg": avg, "Harga": price, "Modal": cost,
        "Nilai Pasar": market, "P&L": pnl, "P&L %": pnl_pct,
        "Perubahan Harian %": day_change
    })

table = pd.DataFrame(records)
valid = table.dropna(subset=["Nilai Pasar"])
market_value = valid["Nilai Pasar"].sum()
invested_known = valid["Modal"].sum()
pnl_total = valid["P&L"].sum()
equity_partial = CASH + market_value
pnl_percent = pnl_total / invested_known * 100 if invested_known else 0
complete = len(valid) == len(HOLDINGS)
total_cost = table["Modal"].sum()
allocation = valid[["Kode", "Nilai Pasar"]].copy()
if CASH > 0:
    allocation.loc[len(allocation)] = ["Cash", CASH]
allocation["Bobot %"] = (
    allocation["Nilai Pasar"] / allocation["Nilai Pasar"].sum() * 100
)

# -------------------- TOP METRICS --------------------
st.subheader("Ringkasan Ekuitas")
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Total Ekuitas" if complete else "Ekuitas terhitung",
    rupiah(equity_partial)
)
m2.metric("Nilai Pasar Saham", rupiah(market_value))
m3.metric("Cash", rupiah(CASH))
m4.metric(
    "Floating P&L",
    rupiah(pnl_total),
    pct(pnl_percent)
)

if not complete:
    st.info(
        "Ekuitas dan P&L di atas hanya menghitung saham yang berhasil "
        "diambil dari ZAPI. Periksa saham yang gagal dimuat sebelum "
        "menggunakan total ini."
    )

# Progress to goal
progress = min(max(equity_partial / float(target), 0), 1)
st.markdown("#### Progres menuju target ekuitas")
st.progress(progress)
p1, p2, p3 = st.columns(3)
p1.metric("Target", rupiah(target))
p2.metric("Progres", f"{progress*100:.2f}%")
p2.caption("Berdasarkan ekuitas yang berhasil dihitung")
p3.metric("Selisih ke target", rupiah(max(float(target) - equity_partial, 0)))

st.divider()

# -------------------- TABS --------------------
tab_overview, tab_monitor, tab_technical, tab_forecast, tab_risk, tab_data = st.tabs([
    "🏠 Overview",
    "📡 Live Monitor",
    "📊 Teknikal",
    "🧠 Forecast",
    "🛡️ Risk",
    "📁 Data"
])

# -------------------- OVERVIEW --------------------
with tab_overview:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Komposisi Portofolio")
        if not allocation.empty:
            alloc_chart = allocation.set_index("Kode")[["Nilai Pasar"]]
            st.bar_chart(alloc_chart, height=300)
            alloc_view = allocation.copy()
            alloc_view["Nilai Pasar"] = alloc_view["Nilai Pasar"].map(rupiah)
            alloc_view["Bobot %"] = alloc_view["Bobot %"].map(lambda x: f"{x:.2f}%")
            st.dataframe(alloc_view, hide_index=True, width="stretch")

    with right:
        st.subheader("Kinerja per Saham")
        perf = table[[
            "Kode", "Harga", "Avg", "P&L", "P&L %", "Perubahan Harian %"
        ]].copy()
        for c in ["Harga", "Avg", "P&L"]:
            perf[c] = perf[c].map(rupiah)
        for c in ["P&L %", "Perubahan Harian %"]:
            perf[c] = perf[c].map(pct)
        st.dataframe(perf, hide_index=True, width="stretch")

    st.subheader("Nilai Portofolio per Saham")
    market_bars = valid.set_index("Kode")[["Nilai Pasar"]]
    st.bar_chart(market_bars, height=260)

    st.caption(
        "P&L dihitung dari harga rata-rata dan harga terakhir yang tersedia. "
        "Belum memperhitungkan biaya transaksi, pajak, dividen, atau aksi korporasi."
    )

# -------------------- LIVE MONITOR --------------------
with tab_monitor:
    st.subheader("Harga dan Indikator Terakhir")
    monitor_rows = []
    for code, df in history.items():
        last = df.iloc[-1]
        price = float(last["close"])
        prev = float(df.iloc[-2]["close"]) if len(df) > 1 else np.nan
        chg = (price / prev - 1) * 100 if pd.notna(prev) and prev else np.nan
        avg = HOLDINGS[code]["avg"]
        monitor_rows.append({
            "Kode": code,
            "Harga": price,
            "Perubahan %": chg,
            "RSI 14": last.get("RSI14", np.nan),
            "EMA 9": last.get("EMA9", np.nan),
            "EMA 21": last.get("EMA21", np.nan),
            "EMA 50": last.get("EMA50", np.nan),
            "ATR 14": last.get("ATR14", np.nan),
            "Volume": last.get("volume", np.nan),
            "Tanggal Data": (
                last["date"].strftime("%d-%m-%Y")
                if "date" in df.columns and pd.notna(last.get("date"))
                else "Tidak tersedia"
            )
        })
    monitor = pd.DataFrame(monitor_rows)
    show_monitor = monitor.copy()
    for c in ["Harga", "EMA 9", "EMA 21", "EMA 50", "ATR 14"]:
        show_monitor[c] = show_monitor[c].map(
            lambda x: rupiah(x) if pd.notna(x) else "N/A"
        )
    show_monitor["Perubahan %"] = show_monitor["Perubahan %"].map(pct)
    show_monitor["RSI 14"] = show_monitor["RSI 14"].map(
        lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
    )
    show_monitor["Volume"] = show_monitor["Volume"].map(
        lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
    )
    st.dataframe(show_monitor, hide_index=True, width="stretch")

    st.subheader("Snapshot Saham")
    cols = st.columns(min(4, len(history)))
    for i, (code, df) in enumerate(history.items()):
        last = df.iloc[-1]
        change = (
            float(df["return"].iloc[-1])
            if len(df) > 1 and pd.notna(df["return"].iloc[-1])
            else np.nan
        )
        with cols[i % len(cols)]:
            st.markdown(f"**{code}**")
            st.metric(
                "Harga terakhir",
                rupiah(last["close"]),
                pct(change)
            )
            rsi = last.get("RSI14", np.nan)
            st.caption(
                f"RSI: {rsi:.1f}" if pd.notna(rsi) else "RSI: belum cukup data"
            )
            date_text = (
                last["date"].strftime("%d %b %Y")
                if "date" in df.columns and pd.notna(last.get("date"))
                else "Tanggal tidak tersedia"
            )
            st.caption(f"Data: {date_text}")

# -------------------- TECHNICAL --------------------
with tab_technical:
    selected_tech = st.selectbox(
        "Pilih saham untuk analisis teknikal",
        list(history.keys()),
        key="technical_stock"
    )
    df = history[selected_tech].copy()
    last = df.iloc[-1]
    price = float(last["close"])

    t1, t2, t3, t4 = st.columns(4)
    rsi = last.get("RSI14", np.nan)
    atr = last.get("ATR14", np.nan)
    vol = last.get("Volatility20", np.nan)
    t1.metric("RSI 14", f"{rsi:.1f}" if pd.notna(rsi) else "N/A")
    t2.metric("ATR 14", rupiah(atr))
    t3.metric("Volatilitas 20 sesi", f"{vol:.2f}%" if pd.notna(vol) else "N/A")
    t4.metric("Drawdown berjalan", pct(last.get("Drawdown", np.nan)))

    if pd.notna(rsi):
        if rsi >= 70:
            st.warning("RSI berada di zona tinggi (≥70). Ini bukan sinyal jual otomatis.")
        elif rsi <= 30:
            st.info("RSI berada di zona rendah (≤30). Ini bukan sinyal beli otomatis.")
        else:
            st.caption("RSI berada di antara zona rendah dan tinggi.")

    st.markdown("#### Harga dan Moving Average")
    if "date" in df.columns:
        chart_cols = ["close"]
        for c in ["EMA9", "EMA21", "EMA50", "MA20", "MA50", "MA200"]:
            if c in df.columns and df[c].notna().any():
                chart_cols.append(c)
        price_chart = df.set_index("date")[chart_cols].rename(
            columns={"close": "Harga"}
        )
        st.line_chart(price_chart, height=380)
    else:
        st.line_chart(df[["close", "EMA9", "EMA21", "EMA50"]], height=380)

    st.markdown("#### Volume")
    if "volume" in df.columns:
        volume_cols = ["volume"]
        if "VolumeMA20" in df.columns and df["VolumeMA20"].notna().any():
            volume_cols.append("VolumeMA20")
        volume_data = df.set_index("date")[volume_cols] if "date" in df.columns else df[volume_cols]
        st.bar_chart(volume_data, height=220)
    else:
        st.info("Data volume tidak tersedia pada respons API.")

    st.markdown("#### Level Referensi ATR")
    if pd.notna(atr):
        lv1, lv2, lv3, lv4 = st.columns(4)
        lv1.metric("Harga terakhir", rupiah(price))
        lv2.metric("−1 ATR", rupiah(price - atr))
        lv3.metric("−2 ATR", rupiah(price - 2 * atr))
        lv4.metric("+1 ATR", rupiah(price + atr))
        st.caption(
            "Level ATR adalah jarak volatilitas historis untuk referensi risiko, "
            "bukan level support/resistance yang terkonfirmasi."
        )
    else:
        st.info("ATR belum dapat dihitung dari data yang tersedia.")

# -------------------- FORECAST --------------------
with tab_forecast:
    st.subheader("Forecast Statistik Berbasis Return Historis")
    st.caption(
        "Skenario menggunakan distribusi return historis bergulir. "
        "Angka ini bukan prediksi terjamin dan bukan output model LSTM."
    )
    forecast_code = st.selectbox(
        "Pilih saham",
        list(history.keys()),
        key="forecast_stock"
    )
    fdf = history[forecast_code].copy()
    close = fdf["close"].dropna()
    current_price = float(close.iloc[-1])
    daily_returns = close.pct_change().dropna()
    horizons = {"D5": 5, "D20": 20, "D60": 60, "Year-end proxy": 60}

    forecast_rows = []
    for label, h in horizons.items():
        hist_forward = (close.shift(-h) / close - 1).dropna() * 100
        if len(hist_forward) >= 10:
            p10, p50, p90 = np.percentile(hist_forward, [10, 50, 90])
            probability_positive = (hist_forward > 0).mean() * 100
            forecast_rows.append({
                "Horizon": label,
                "Sesi": h,
                "P10": current_price * (1 + p10 / 100),
                "P50": current_price * (1 + p50 / 100),
                "P90": current_price * (1 + p90 / 100),
                "Return median %": p50,
                "Historis positif %": probability_positive,
                "Jumlah sampel": len(hist_forward)
            })
        else:
            forecast_rows.append({
                "Horizon": label, "Sesi": h,
                "P10": np.nan, "P50": np.nan, "P90": np.nan,
                "Return median %": np.nan,
                "Historis positif %": np.nan,
                "Jumlah sampel": len(hist_forward)
            })

    ftable = pd.DataFrame(forecast_rows)
    fview = ftable.copy()
    for c in ["P10", "P50", "P90"]:
        fview[c] = fview[c].map(rupiah)
    fview["Return median %"] = fview["Return median %"].map(pct)
    fview["Historis positif %"] = fview["Historis positif %"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    )
    st.metric("Harga acuan terakhir", rupiah(current_price))
    st.dataframe(fview, hide_index=True, width="stretch")

    valid_forecast = ftable.dropna(subset=["P10", "P50", "P90"])
    if not valid_forecast.empty:
        st.markdown("#### Rentang skenario harga")
        forecast_chart = valid_forecast.set_index("Horizon")[["P10", "P50", "P90"]]
        st.line_chart(forecast_chart, height=280)
    st.caption(
        "P10/P50/P90 adalah persentil return historis untuk horizon serupa "
        "yang diterapkan pada harga terakhir. Label year-end proxy memakai "
        "60 sesi, bukan kalender bursa yang presisi."
    )

    st.markdown("#### Simulasi Bootstrap")
    sim_horizon = st.select_slider(
        "Horizon simulasi (sesi)",
        options=[5, 10, 20, 40, 60],
        value=20
    )
    if len(daily_returns) >= 30:
        rng = np.random.default_rng(42)
        ret = daily_returns.to_numpy()
        n_sims = 2000
        sampled = rng.choice(
            ret, size=(n_sims, sim_horizon), replace=True
        )
        paths = current_price * np.cumprod(1 + sampled, axis=1)
        final_prices = paths[:, -1]
        q10, q50, q90 = np.percentile(final_prices, [10, 50, 90])
        b1, b2, b3 = st.columns(3)
        b1.metric("P10 simulasi", rupiah(q10))
        b2.metric("Median simulasi", rupiah(q50))
        b3.metric("P90 simulasi", rupiah(q90))
        st.caption(
            "Simulasi bootstrap mengulang sampel return harian historis secara acak "
            "dan mengasumsikan return tidak bergantung satu sama lain."
        )
    else:
        st.info("Data return belum cukup untuk simulasi.")

# -------------------- RISK --------------------
with tab_risk:
    st.subheader("Risk Management")
    risk_rows = []
    for code, df in history.items():
        last = df.iloc[-1]
        price = float(last["close"])
        atr = last.get("ATR14", np.nan)
        dd = last.get("Drawdown", np.nan)
        vol = last.get("Volatility20", np.nan)
        qty = HOLDINGS[code]["lots"] * SHARES_PER_LOT
        risk_rows.append({
            "Kode": code,
            "Harga": price,
            "ATR14": atr,
            "ATR %": atr / price * 100 if pd.notna(atr) and price else np.nan,
            "Harga - 2 ATR": price - 2 * atr if pd.notna(atr) else np.nan,
            "Drawdown %": dd,
            "Volatilitas 20 sesi %": vol,
            "Nilai Pasar": price * qty
        })
    risk_df = pd.DataFrame(risk_rows)
    risk_view = risk_df.copy()
    for c in ["Harga", "ATR14", "Harga - 2 ATR", "Nilai Pasar"]:
        risk_view[c] = risk_view[c].map(
            lambda x: rupiah(x) if pd.notna(x) else "N/A"
        )
    for c in ["ATR %", "Drawdown %", "Volatilitas 20 sesi %"]:
        risk_view[c] = risk_view[c].map(pct)
    st.dataframe(risk_view, hide_index=True, width="stretch")

    st.markdown("#### Estimasi penurunan portofolio")
    risk_fraction = st.slider(
        "Asumsi penurunan harga saham (%)",
        min_value=1, max_value=30, value=10, step=1
    )
    estimated_loss = market_value * risk_fraction / 100
    r1, r2, r3 = st.columns(3)
    r1.metric("Nilai saham saat ini", rupiah(market_value))
    r2.metric("Asumsi penurunan", f"{risk_fraction}%")
    r3.metric("Dampak estimasi", f"-{rupiah(estimated_loss)}")
    st.caption(
        "Ini simulasi sensitivitas sederhana dengan asumsi semua saham turun "
        "dalam persentase yang sama, bukan VaR atau prediksi kerugian."
    )

    st.markdown("#### Konsentrasi kepemilikan")
    if not allocation.empty:
        max_weight = allocation["Bobot %"].max()
        largest = allocation.loc[allocation["Bobot %"].idxmax(), "Kode"]
        st.write(
            f"Posisi dengan bobot terbesar: **{largest}** "
            f"({max_weight:.2f}% dari nilai saham + cash yang terhitung)."
        )

# -------------------- DATA --------------------
with tab_data:
    st.subheader("Riwayat Data ZAPI")
    data_code = st.selectbox(
        "Pilih kode saham",
        list(history.keys()),
        key="data_stock"
    )
    data_df = history[data_code].copy()
    if "date" in data_df.columns:
        data_df = data_df.sort_values("date", ascending=False)
    st.write(f"Jumlah baris: {len(data_df):,}")
    if "date" in data_df.columns:
        st.write(
            "Rentang data: "
            f"{data_df['date'].min().strftime('%d-%m-%Y')} sampai "
            f"{data_df['date'].max().strftime('%d-%m-%Y')}"
        )
    st.dataframe(data_df, width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Download CSV",
        data=data_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{data_code}_ZAPI_history.csv",
        mime="text/csv",
        use_container_width=True
    )

# -------------------- FOOTER --------------------
st.divider()
st.caption(
    f"Terakhir dirender: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')} WIB. "
    "Harga mengikuti data terakhir yang dikembalikan ZAPI, bukan selalu real-time. "
    "Perhitungan tidak termasuk biaya transaksi, pajak, dividen, dan aksi korporasi."
)
