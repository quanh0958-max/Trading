import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from functools import partial
import os

warnings.filterwarnings("ignore")

# Cố gắng import backtesting & hyperopt, nếu môi trường chưa cài đặt đầy đủ thì có cơ chế fallback
try:
    from backtesting import Backtest, Strategy
    HAS_BACKTESTING = True
except Exception as e:
    HAS_BACKTESTING = False

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HAS_HYPEROPT = True
except Exception as e:
    HAS_HYPEROPT = False

# ==============================================================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Kiểm định Chiến lược SMA + OBV | Phân Tích Định Lượng",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho phong cách hiện đại, trực quan
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.15);
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-primary { background-color: #dbeafe; color: #1e40af; }
    .badge-success { background-color: #dcfce7; color: #166534; }
    .badge-warning { background-color: #fef9c3; color: #854d0e; }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #2563eb;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. XỬ LÝ DỮ LIỆU (DATA PREPARATION)
# ==============================================================================
@st.cache_data
def prepare_stock_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa dữ liệu tương tự hàm trong notebook."""
    df = df_raw.copy()
    df.columns = df.columns.str.strip().str.lower()

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu tải lên bị thiếu các cột bắt buộc: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }
    df = df.rename(columns=rename_map)
    df = df[["date", "Open", "High", "Low", "Close", "Volume"]].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates(subset="date", keep="last")
    df = df.set_index("date")
    return df

# ==============================================================================
# 3. TẠO TÍN HIỆU GIAO DỊCH (SIGNAL GENERATION)
# ==============================================================================
def find_position_sma(df: pd.DataFrame, paras: dict) -> pd.Series:
    """Tín hiệu SMA: Golden Cross (1) và Death Cross (-1)."""
    position = pd.Series(data=0.0, index=df.index, name="position", dtype=float)
    ma_short = int(paras.get("ma_short", 150))
    ma_long = int(paras.get("ma_long", 210))

    if ma_short >= ma_long:
        return position

    ma_s = ta.trend.SMAIndicator(close=df["Close"], window=ma_short).sma_indicator()
    ma_l = ta.trend.SMAIndicator(close=df["Close"], window=ma_long).sma_indicator()

    buy_signal = (ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))
    sell_signal = (ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))

    position.loc[buy_signal] = 1.0
    position.loc[sell_signal] = -1.0
    return position

def find_position_obv(df: pd.DataFrame, paras: dict) -> pd.Series:
    """Tín hiệu OBV: Cắt lên MA(OBV) (1) và cắt xuống MA(OBV) (-1)."""
    position = pd.Series(data=0.0, index=df.index, name="position", dtype=float)
    obv_window = int(paras.get("obv_window", 5))

    obv = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    obv_ma = obv.rolling(window=obv_window).mean()

    buy_signal = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell_signal = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))

    position.loc[buy_signal] = 1.0
    position.loc[sell_signal] = -1.0
    return position

def find_position_combined(df: pd.DataFrame, sma_paras: dict, obv_paras: dict, mode: str = "strict_and") -> pd.Series:
    """
    Kết hợp tín hiệu SMA và OBV:
    - strict_and: Cả 2 cùng phát tín hiệu giao cắt trong cùng một phiên (logic gốc của notebook).
    - regime_confirm: SMA xác định xu hướng lớn (MA ngắn > MA dài), OBV phát tín hiệu kích hoạt điểm vào lệnh (Momentum Trigger).
    """
    position = pd.Series(data=0.0, index=df.index, name="position", dtype=float)

    if mode == "strict_and":
        sma_position = find_position_sma(df, sma_paras)
        obv_position = find_position_obv(df, obv_paras)
        buy_signal = (sma_position == 1.0) & (obv_position == 1.0)
        sell_signal = (sma_position == -1.0) & (obv_position == -1.0)
    else:
        # Chế độ Regime + Trigger (Thực tế hơn trong giao dịch định lượng)
        ma_short = int(sma_paras.get("ma_short", 150))
        ma_long = int(sma_paras.get("ma_long", 210))
        if ma_short >= ma_long:
            return position
        
        ma_s = ta.trend.SMAIndicator(close=df["Close"], window=ma_short).sma_indicator()
        ma_l = ta.trend.SMAIndicator(close=df["Close"], window=ma_long).sma_indicator()
        sma_uptrend = ma_s > ma_l
        sma_downtrend = ma_s < ma_l

        obv_pos = find_position_obv(df, obv_paras)
        buy_signal = sma_uptrend & (obv_pos == 1.0)
        sell_signal = sma_downtrend & (obv_pos == -1.0)

    position.loc[buy_signal] = 1.0
    position.loc[sell_signal] = -1.0
    return position

# ==============================================================================
# 4. BACKTEST SIMULATOR & PERFORMANCE METRICS
# ==============================================================================
def compute_sharpe(returns: pd.Series, window: int = 252) -> float:
    """Tính chỉ số Sharpe hàng năm."""
    valid_returns = returns.dropna()
    if valid_returns.empty or valid_returns.std() == 0:
        return 0.0
    return float(np.sqrt(window) * valid_returns.mean() / valid_returns.std())

if HAS_BACKTESTING:
    class GeneralStrategy(Strategy):
        def init(self):
            pass

        def next(self):
            signal = self.data.position[-1]
            if signal == 1 and not self.position:
                self.buy()
            elif signal == -1 and self.position:
                self.position.close()

def run_simulation(df_input: pd.DataFrame, cash: float = 1_000_000, commission: float = 0.0):
    """
    Chạy Backtest và tính toán các chỉ số hiệu suất.
    Hỗ trợ cả thư viện backtesting và công cụ mô phỏng thuần Python chính xác 100%.
    """
    df = df_input.copy()
    if "position" not in df.columns:
        raise ValueError("DataFrame chưa có cột 'position'.")

    # Sử dụng backtesting.py nếu có sẵn
    if HAS_BACKTESTING:
        try:
            bt = Backtest(
                df,
                GeneralStrategy,
                cash=cash,
                commission=commission,
                trade_on_close=True,
                exclusive_orders=True
            )
            stats_series = bt.run()
            stats = stats_series.to_frame(name="Value")
            stats.loc["Duration", "Value"] = len(df)

            equity_curve = stats_series["_equity_curve"]["Equity"]
            returns = equity_curve.pct_change()
            stats.loc["Sharpe Ratio", "Value"] = compute_sharpe(returns, window=252)

            # Lấy danh sách giao dịch
            trades_df = stats_series["_trades"] if "_trades" in stats_series else pd.DataFrame()

            # Format số
            stats["Value"] = stats["Value"].apply(
                lambda x: round(x, 2) if isinstance(x, (int, float, np.number)) else x
            )
            return stats, equity_curve, trades_df
        except Exception:
            pass  # Fallback sang engine thuần nếu gặp vấn đề môi trường

    # Engine thuần Python tối ưu hóa cao (tái hiện chính xác quy tắc GeneralStrategy)
    close_prices = df["Close"].values
    signals = df["position"].values
    dates = df.index
    n = len(df)

    current_cash = float(cash)
    shares = 0.0
    in_pos = False
    entry_price = 0.0
    entry_idx = 0

    equity_list = np.zeros(n, dtype=float)
    trades_records = []

    for i in range(n):
        sig = signals[i]
        price = close_prices[i]

        if sig == 1.0 and not in_pos:
            # Mua tại Close bar hiện tại
            effective_cash = current_cash * (1.0 - commission)
            shares = effective_cash / price
            current_cash = 0.0
            in_pos = True
            entry_price = price
            entry_idx = i
        elif sig == -1.0 and in_pos:
            # Bán tại Close bar hiện tại
            proceeds = shares * price * (1.0 - commission)
            pnl_pct = (price / entry_price - 1.0) * 100.0
            trades_records.append({
                "Size": shares,
                "EntryBar": entry_idx,
                "ExitBar": i,
                "EntryPrice": entry_price,
                "ExitPrice": price,
                "EntryTime": dates[entry_idx],
                "ExitTime": dates[i],
                "PnL": proceeds - (shares * entry_price),
                "ReturnPct": pnl_pct,
                "Duration": i - entry_idx
            })
            current_cash = proceeds
            shares = 0.0
            in_pos = False

        if in_pos:
            equity_list[i] = shares * price
        else:
            equity_list[i] = current_cash

    equity_series = pd.Series(equity_list, index=dates, name="Equity")
    daily_returns = equity_series.pct_change()

    total_return = ((equity_series.iloc[-1] / cash) - 1.0) * 100.0 if n > 0 else 0.0
    bh_return = ((close_prices[-1] / close_prices[0]) - 1.0) * 100.0 if n > 0 else 0.0
    sharpe = compute_sharpe(daily_returns, window=252)

    # Max Drawdown
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_drawdown = float(drawdown.min() * 100.0) if not drawdown.empty else 0.0

    trades_df = pd.DataFrame(trades_records)
    num_trades = len(trades_df)
    win_rate = (len(trades_df[trades_df["ReturnPct"] > 0]) / num_trades * 100.0) if num_trades > 0 else 0.0

    stats_dict = {
        "Start": dates[0] if n > 0 else None,
        "End": dates[-1] if n > 0 else None,
        "Duration": n,
        "Equity Final [$]": round(equity_series.iloc[-1], 2),
        "Return [%]": round(total_return, 2),
        "Buy & Hold Return [%]": round(bh_return, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max. Drawdown [%]": round(max_drawdown, 2),
        "# Trades": num_trades,
        "Win Rate [%]": round(win_rate, 2),
        "Best Trade [%]": round(trades_df["ReturnPct"].max(), 2) if num_trades > 0 else 0.0,
        "Worst Trade [%]": round(trades_df["ReturnPct"].min(), 2) if num_trades > 0 else 0.0,
    }
    stats = pd.DataFrame.from_dict(stats_dict, orient="index", columns=["Value"])
    return stats, equity_series, trades_df

# ==============================================================================
# 5. HÀM TỐI ƯU HÓA HYPEROPT (CHỈ ÁP DỤNG TRÊN TẬP TRAIN)
# ==============================================================================
def score_objective(paras: dict, df: pd.DataFrame, strat_type: str, cash: float, commission: float, fitness: str) -> float:
    """Tính điểm hàm mục tiêu cho Hyperopt."""
    df_temp = df.copy()
    if strat_type == "sma":
        ma_short = int(paras["ma_short"])
        ma_long = int(paras["ma_long"])
        if ma_short >= ma_long:
            return 999999.0
        df_temp["position"] = find_position_sma(df_temp, paras)
    elif strat_type == "obv":
        df_temp["position"] = find_position_obv(df_temp, paras)

    try:
        stats, _, _ = run_simulation(df_temp, cash=cash, commission=commission)
        ret = float(stats.loc["Return [%]", "Value"])
        sharpe = float(stats.loc["Sharpe Ratio", "Value"])
        mdd = abs(float(stats.loc["Max. Drawdown [%]", "Value"]))
        worst_t = abs(float(stats.loc["Worst Trade [%]", "Value"]))

        if fitness == "return":
            return -ret
        elif fitness == "Sharpe":
            return -sharpe
        elif fitness == "max_drawdown":
            return mdd
        elif fitness == "worst_trade":
            return worst_t
    except Exception:
        return 999999.0
    return 999999.0

def run_hyperopt_optimization(df_train: pd.DataFrame, cash: float, commission: float, fitness: str, max_evals: int = 50):
    """Thực hiện tối ưu hóa tham số SMA và OBV độc lập trên tập Train."""
    if not HAS_HYPEROPT:
        st.error("Thư viện 'hyperopt' chưa được cài đặt. Vui lòng kiểm tra requirements.txt.")
        return None, None

    # 1. Tối ưu SMA
    fspace_sma = {
        "ma_short": hp.quniform("ma_short", 10, 150, 5),
        "ma_long": hp.quniform("ma_long", 160, 400, 5)
    }
    fmin_sma = partial(score_objective, df=df_train, strat_type="sma", cash=cash, commission=commission, fitness=fitness)
    trials_sma = Trials()
    best_sma_raw = fmin(fn=fmin_sma, space=fspace_sma, algo=tpe.suggest, max_evals=max_evals, trials=trials_sma)
    best_sma = {
        "ma_short": int(best_sma_raw["ma_short"]),
        "ma_long": int(best_sma_raw["ma_long"])
    }

    # 2. Tối ưu OBV
    fspace_obv = {
        "obv_window": hp.quniform("obv_window", 3, 100, 1)
    }
    fmin_obv = partial(score_objective, df=df_train, strat_type="obv", cash=cash, commission=commission, fitness=fitness)
    trials_obv = Trials()
    best_obv_raw = fmin(fn=fmin_obv, space=fspace_obv, algo=tpe.suggest, max_evals=max_evals, trials=trials_obv)
    best_obv = {
        "obv_window": int(best_obv_raw["obv_window"])
    }

    return best_sma, best_obv

# ==============================================================================
# 6. SIDEBAR CONTROLS & THIẾT LẬP
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
    st.title("⚙️ Thiết Lập Hệ Thống")

    # Phân vùng 1: Dữ liệu
    st.subheader("1. Nguồn Dữ Liệu")
    data_source = st.radio(
        "Chọn nguồn nạp dữ liệu:",
        options=["File mẫu (ACB.csv)", "Tải lên file CSV mới"],
        index=0
    )

    df_raw = None
    if data_source == "File mẫu (ACB.csv)":
        sample_path = "ACB.csv"
        if os.path.exists(sample_path):
            df_raw = pd.read_csv(sample_path, encoding="utf-8-sig", low_memory=False)
            st.success(f"✅ Đã tải file mẫu `{sample_path}` ({len(df_raw)} dòng)")
        else:
            st.warning(f"⚠️ Không tìm thấy `{sample_path}` trong thư mục làm việc. Vui lòng tải file lên.")
    else:
        uploaded_file = st.file_uploader("Tải lên file dữ liệu chứng khoán (CSV)", type=["csv"])
        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file, encoding="utf-8-sig", low_memory=False)
            st.success("✅ Đã nạp file dữ liệu thành công!")

    st.markdown("---")

    # Phân vùng 2: Phân chia Train/Test & Giao dịch
    st.subheader("2. Cấu Hình Kiểm Định")
    train_ratio = st.slider("Tỷ lệ tập huấn luyện (Train Ratio)", min_value=0.5, max_value=0.9, value=0.8, step=0.05,
                            help="Chia chuỗi thời gian: phần đầu dùng để Train / Tối ưu, phần sau làm Out-of-sample Test.")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        initial_cash = st.number_input("Vốn ban đầu (VND)", min_value=10_000, value=1_000_000, step=100_000)
    with col_c2:
        commission_rate = st.number_input("Phí GD (%)", min_value=0.0, max_value=1.0, value=0.0, step=0.05) / 100.0

    st.markdown("---")

    # Phân vùng 3: Tham số Chiến lược
    st.subheader("3. Tham Số Chiến Lược")
    
    # Khởi tạo session state lưu tham số nếu chưa có
    if "ma_short" not in st.session_state:
        st.session_state.ma_short = 150
    if "ma_long" not in st.session_state:
        st.session_state.ma_long = 210
    if "obv_window" not in st.session_state:
        st.session_state.obv_window = 5

    strategy_mode = st.selectbox(
        "Cơ chế kết hợp tín hiệu:",
        options=["strict_and", "regime_confirm"],
        format_func=lambda x: "🎯 Crossover Đồng Thời (Chuẩn Notebook AND)" if x == "strict_and" else "⚡ Bộ Lọc Xu Hướng + Điểm Kích Hoạt (Regime + Trigger)",
        help="Chuẩn notebook: Cả SMA và OBV phải cùng cắt trong đúng 1 phiên.\nBộ lọc xu hướng: SMA định xu hướng (MA ngắn > MA dài), OBV phát tín hiệu vào lệnh."
    )

    ma_short = st.slider("SMA Ngắn (ma_short)", min_value=5, max_value=200, value=st.session_state.ma_short, step=5)
    ma_long = st.slider("SMA Dài (ma_long)", min_value=20, max_value=400, value=st.session_state.ma_long, step=5)
    obv_window = st.slider("Cửa sổ OBV MA (obv_window)", min_value=2, max_value=100, value=st.session_state.obv_window, step=1)

    # Cập nhật session_state từ slider
    st.session_state.ma_short = ma_short
    st.session_state.ma_long = ma_long
    st.session_state.obv_window = obv_window

    st.markdown("---")

    # Phân vùng 4: Tối ưu hóa Hyperopt
    st.subheader("4. Tối Ưu Hóa Tham Số (Hyperopt)")
    st.caption("Chỉ chạy trên tập Train để tránh rò rỉ dữ liệu (Data Leakage).")
    opt_fitness = st.selectbox("Mục tiêu tối ưu (Fitness):", options=["return", "Sharpe", "max_drawdown", "worst_trade"], index=0)
    opt_evals = st.select_slider("Số lần thử nghiệm (Evals):", options=[20, 50, 100, 200, 500], value=50)

    btn_optimize = st.button("🚀 Chạy Tối Ưu Hóa Trên Train", use_container_width=True)

# ==============================================================================
# 7. MAIN DASHBOARD LOGIC
# ==============================================================================
st.markdown('<div class="main-title">📈 Hệ Thống Kiểm Định Chiến Lược Giao Dịch Định Lượng</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Mô hình kết hợp tín hiệu <b>SMA (Simple Moving Average)</b> và <b>OBV (On-Balance Volume)</b> — Nhóm 5</div>', unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 1.2rem;">
    <span class="badge badge-primary">Quy tắc AND</span>
    <span class="badge badge-success">Hyperopt TPE Optimization</span>
    <span class="badge badge-warning">Train/Test Out-of-Sample</span>
</div>
""", unsafe_allow_html=True)

if df_raw is None:
    st.info("👋 Chào bạn! Hãy nạp file dữ liệu chứng khoán (như `ACB.csv`) từ thanh bên trái (Sidebar) để bắt đầu phân tích.")
    st.stop()

# Chuẩn bị dữ liệu
try:
    df_clean = prepare_stock_data(df_raw)
except Exception as e:
    st.error(f"Lỗi khi xử lý dữ liệu: {e}")
    st.stop()

# Phân chia Train / Test
split_idx = int(len(df_clean) * train_ratio)
df_train = df_clean.iloc[:split_idx].copy()
df_test = df_clean.iloc[split_idx:].copy()

# Xử lý sự kiện bấm nút Tối ưu hóa
if btn_optimize:
    with st.spinner(f"Đang chạy Hyperopt ({opt_evals} lần thử nghiệm) trên tập Train... Vui lòng chờ vài giây..."):
        best_sma, best_obv = run_hyperopt_optimization(
            df_train=df_train,
            cash=initial_cash,
            commission=commission_rate,
            fitness=opt_fitness,
            max_evals=opt_evals
        )
        if best_sma and best_obv:
            st.session_state.ma_short = best_sma["ma_short"]
            st.session_state.ma_long = best_sma["ma_long"]
            st.session_state.obv_window = best_obv["obv_window"]
            st.success(f"🎉 Tối ưu hoàn tất! Tham số tốt nhất tìm được: SMA Short = {best_sma['ma_short']}, SMA Long = {best_sma['ma_long']}, OBV Window = {best_obv['obv_window']}")
            st.rerun()

# Tham số đang được áp dụng
current_sma_paras = {"ma_short": st.session_state.ma_short, "ma_long": st.session_state.ma_long}
current_obv_paras = {"obv_window": st.session_state.obv_window}

# Tính toán vị thế trên Train & Test
df_train_sma = df_train.copy()
df_train_sma["position"] = find_position_sma(df_train_sma, current_sma_paras)
df_train_obv = df_train.copy()
df_train_obv["position"] = find_position_obv(df_train_obv, current_obv_paras)
df_train_comb = df_train.copy()
df_train_comb["position"] = find_position_combined(df_train_comb, current_sma_paras, current_obv_paras, mode=strategy_mode)

df_test_sma = df_test.copy()
df_test_sma["position"] = find_position_sma(df_test_sma, current_sma_paras)
df_test_obv = df_test.copy()
df_test_obv["position"] = find_position_obv(df_test_obv, current_obv_paras)
df_test_comb = df_test.copy()
df_test_comb["position"] = find_position_combined(df_test_comb, current_sma_paras, current_obv_paras, mode=strategy_mode)

# Chạy mô phỏng Backtest
stats_train_sma, eq_train_sma, tr_train_sma = run_simulation(df_train_sma, initial_cash, commission_rate)
stats_train_obv, eq_train_obv, tr_train_obv = run_simulation(df_train_obv, initial_cash, commission_rate)
stats_train_comb, eq_train_comb, tr_train_comb = run_simulation(df_train_comb, initial_cash, commission_rate)

stats_test_sma, eq_test_sma, tr_test_sma = run_simulation(df_test_sma, initial_cash, commission_rate)
stats_test_obv, eq_test_obv, tr_test_obv = run_simulation(df_test_obv, initial_cash, commission_rate)
stats_test_comb, eq_test_comb, tr_test_comb = run_simulation(df_test_comb, initial_cash, commission_rate)

# ==============================================================================
# 8. CÁC TABS HIỂN THỊ NỘI DUNG CHÍNH
# ==============================================================================
tab_overview, tab_backtest, tab_optimization, tab_insights = st.tabs([
    "📊 Biểu Đồ & Tín Hiệu",
    "📈 Kết Quả Kiểm Định (Train vs. Test)",
    "🔬 Tối Ưu Hóa Tham Số",
    "📑 Phân Tích & Khuyến Nghị Định Lượng"
])

# ------------------------------------------------------------------------------
# TAB 1: BIỂU ĐỒ & TÍN HIỆU
# ------------------------------------------------------------------------------
with tab_overview:
    st.markdown('<div class="section-title">Tổng quan chuỗi dữ liệu & Tín hiệu kỹ thuật</div>', unsafe_allow_html=True)

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    with col_meta1:
        st.metric("Tổng số phiên", f"{len(df_clean):,} phiên")
    with col_meta2:
        st.metric("Tập Train", f"{len(df_train):,} phiên ({train_ratio*100:.0f}%)", f"{df_train.index.min().strftime('%d/%m/%Y')} → {df_train.index.max().strftime('%d/%m/%Y')}")
    with col_meta3:
        st.metric("Tập Test (Out-of-sample)", f"{len(df_test):,} phiên ({(1-train_ratio)*100:.0f}%)", f"{df_test.index.min().strftime('%d/%m/%Y')} → {df_test.index.max().strftime('%d/%m/%Y')}")
    with col_meta4:
        st.metric("Bộ tham số đang dùng", f"SMA({st.session_state.ma_short}, {st.session_state.ma_long})", f"OBV MA({st.session_state.obv_window})")

    # Chọn tập hiển thị biểu đồ
    sub_choice = st.radio("Chọn tập dữ liệu hiển thị trên đồ thị:", ["Toàn bộ dữ liệu (Full)", "Tập Huấn luyện (Train)", "Tập Kiểm định (Test)"], horizontal=True)
    if sub_choice == "Tập Huấn luyện (Train)":
        df_plot = df_train_comb.copy()
    elif sub_choice == "Tập Kiểm định (Test)":
        df_plot = df_test_comb.copy()
    else:
        df_plot = df_clean.copy()
        df_plot["position"] = find_position_combined(df_plot, current_sma_paras, current_obv_paras, mode=strategy_mode)

    # Tính toán đường SMA và OBV để vẽ
    sma_s_series = ta.trend.SMAIndicator(close=df_plot["Close"], window=int(current_sma_paras["ma_short"])).sma_indicator()
    sma_l_series = ta.trend.SMAIndicator(close=df_plot["Close"], window=int(current_sma_paras["ma_long"])).sma_indicator()
    obv_series = ta.volume.OnBalanceVolumeIndicator(close=df_plot["Close"], volume=df_plot["Volume"]).on_balance_volume()
    obv_ma_series = obv_series.rolling(window=int(current_obv_paras["obv_window"])).mean()

    # Biểu đồ kỹ thuật đa tầng (Price, Volume, OBV)
    fig_tech = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.20, 0.25],
        subplot_titles=("Giá & Đường Trung Bình Động (SMA)", "Khối lượng Giao dịch (Volume)", "Chỉ số Khối lượng Cân bằng (OBV & OBV MA)")
    )

    # Nến giá & SMA
    fig_tech.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["Open"],
        high=df_plot["High"],
        low=df_plot["Low"],
        close=df_plot["Close"],
        name="Giá Cổ Phiếu",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444"
    ), row=1, col=1)

    fig_tech.add_trace(go.Scatter(
        x=df_plot.index, y=sma_s_series,
        line=dict(color="#3b82f6", width=1.5),
        name=f"SMA Ngắn ({current_sma_paras['ma_short']})"
    ), row=1, col=1)

    fig_tech.add_trace(go.Scatter(
        x=df_plot.index, y=sma_l_series,
        line=dict(color="#f97316", width=1.5),
        name=f"SMA Dài ({current_sma_paras['ma_long']})"
    ), row=1, col=1)

    # Đánh dấu điểm Mua & Bán của chiến lược kết hợp
    buy_signals = df_plot[df_plot["position"] == 1.0]
    sell_signals = df_plot[df_plot["position"] == -1.0]

    if not buy_signals.empty:
        fig_tech.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals["Low"] * 0.98,
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#16a34a"),
            name="Điểm Mua (BUY Signal)"
        ), row=1, col=1)

    if not sell_signals.empty:
        fig_tech.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals["High"] * 1.02,
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#dc2626"),
            name="Điểm Bán (SELL Signal)"
        ), row=1, col=1)

    # Đường phân cách Train/Test nếu hiển thị toàn bộ dữ liệu
    if sub_choice == "Toàn bộ dữ liệu (Full)":
        split_date = df_test.index[0]
        fig_tech.add_vline(x=split_date, line_width=1.5, line_dash="dash", line_color="#64748b",
                           annotation_text="Phân định Train / Test", annotation_position="top left", row=1, col=1)

    # Volume
    colors_vol = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(df_plot["Close"], df_plot["Open"])]
    fig_tech.add_trace(go.Bar(
        x=df_plot.index, y=df_plot["Volume"],
        marker_color=colors_vol, name="Volume", showlegend=False
    ), row=2, col=1)

    # OBV & OBV MA
    fig_tech.add_trace(go.Scatter(
        x=df_plot.index, y=obv_series,
        line=dict(color="#8b5cf6", width=1.5), name="OBV"
    ), row=3, col=1)

    fig_tech.add_trace(go.Scatter(
        x=df_plot.index, y=obv_ma_series,
        line=dict(color="#ec4899", width=1.2, dash="dot"), name=f"OBV MA ({current_obv_paras['obv_window']})"
    ), row=3, col=1)

    fig_tech.update_layout(
        height=800,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig_tech, use_container_width=True)

    # Danh sách các tín hiệu phát sinh
    st.subheader("📋 Bảng Thống Kê Các Tín Hiệu Đã Phát Sinh")
    signals_summary = df_plot[df_plot["position"] != 0].copy()
    if signals_summary.empty:
        st.warning("⚠️ Không có tín hiệu giao dịch nào được phát sinh với bộ tham số và cơ chế hiện tại trên tập này.")
    else:
        signals_display = pd.DataFrame({
            "Ngày": signals_summary.index.strftime('%d/%m/%Y'),
            "Giá Đóng Cửa": signals_summary["Close"],
            "Khối Lượng": signals_summary["Volume"],
            "Loại Tín Hiệu": signals_summary["position"].apply(lambda x: "🟢 MUA (BUY)" if x == 1.0 else "🔴 BÁN (SELL)")
        })
        st.dataframe(signals_display, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: KẾT QUẢ KIỂM ĐỊNH (TRAIN VS. TEST)
# ------------------------------------------------------------------------------
with tab_backtest:
    st.markdown('<div class="section-title">So sánh Hiệu Suất Chiến Lược (Train vs. Test)</div>', unsafe_allow_html=True)

    # Bảng kết quả tổng hợp định dạng chuẩn như notebook
    def extract_row(name, stats):
        return {
            "Chiến lược": name,
            "Lợi nhuận Return [%]": stats.loc["Return [%]", "Value"],
            "Sharpe Ratio": stats.loc["Sharpe Ratio", "Value"],
            "Max Drawdown [%]": stats.loc["Max. Drawdown [%]", "Value"],
            "Số Lệnh (# Trades)": stats.loc["# Trades", "Value"],
            "Win Rate [%]": stats.loc.get("Win Rate [%]", pd.Series({"Value": 0})).get("Value", 0)
        }

    comp_train = pd.DataFrame([
        extract_row("Chỉ dùng SMA", stats_train_sma),
        extract_row("Chỉ dùng OBV", stats_train_obv),
        extract_row("Kết hợp SMA + OBV", stats_train_comb)
    ])

    comp_test = pd.DataFrame([
        extract_row("Chỉ dùng SMA", stats_test_sma),
        extract_row("Chỉ dùng OBV", stats_test_obv),
        extract_row("Kết hợp SMA + OBV", stats_test_comb)
    ])

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### 🏋️ Tập Huấn Luyện (TRAIN SET)")
        st.dataframe(comp_train.style.format({
            "Lợi nhuận Return [%]": "{:.2f}%",
            "Sharpe Ratio": "{:.2f}",
            "Max Drawdown [%]": "{:.2f}%",
            "Win Rate [%]": "{:.2f}%"
        }), use_container_width=True)

    with col_t2:
        st.markdown("#### 🎯 Tập Kiểm Định Mù (TEST SET - Out of Sample)")
        st.dataframe(comp_test.style.format({
            "Lợi nhuận Return [%]": "{:.2f}%",
            "Sharpe Ratio": "{:.2f}",
            "Max Drawdown [%]": "{:.2f}%",
            "Win Rate [%]": "{:.2f}%"
        }), use_container_width=True)

    st.markdown("---")

    # Các thẻ chỉ số hiệu suất của Chiến lược Kết hợp trên tập Test
    st.markdown("#### 🏆 Chỉ số Chiến Lược Kết Hợp SMA + OBV trên Tập Test:")
    kp1, kp2, kp3, kp4, kp5 = st.columns(5)
    with kp1:
        st.metric("Lợi nhuận Test", f"{stats_test_comb.loc['Return [%]', 'Value']}%")
    with kp2:
        st.metric("Sharpe Ratio", f"{stats_test_comb.loc['Sharpe Ratio', 'Value']}")
    with kp3:
        st.metric("Max Drawdown", f"{stats_test_comb.loc['Max. Drawdown [%]', 'Value']}%")
    with kp4:
        st.metric("Số lệnh thực thi", f"{stats_test_comb.loc['# Trades', 'Value']}")
    with kp5:
        st.metric("Win Rate", f"{stats_test_comb.loc.get('Win Rate [%]', pd.Series({'Value': 0})).get('Value', 0)}%")

    # Biểu đồ Đường cong vốn (Equity Curve)
    st.markdown("#### 📈 So sánh Đường Cong Vốn (Equity Curve)")
    dataset_view = st.radio("Xem đường cong vốn của tập:", ["Tập Huấn Luyện (Train)", "Tập Kiểm Định (Test)"], horizontal=True)

    if dataset_view == "Tập Huấn Luyện (Train)":
        eq_sma = eq_train_sma
        eq_obv = eq_train_obv
        eq_comb = eq_train_comb
        df_target = df_train
    else:
        eq_sma = eq_test_sma
        eq_obv = eq_test_obv
        eq_comb = eq_test_comb
        df_target = df_test

    # Chuỗi vốn chuẩn hóa theo %
    eq_sma_pct = (eq_sma / initial_cash - 1) * 100
    eq_obv_pct = (eq_obv / initial_cash - 1) * 100
    eq_comb_pct = (eq_comb / initial_cash - 1) * 100
    eq_bh_pct = (df_target["Close"] / df_target["Close"].iloc[0] - 1) * 100

    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=df_target.index, y=eq_comb_pct, mode="lines", name="SMA + OBV (Combined)", line=dict(color="#2563eb", width=2.5)))
    fig_equity.add_trace(go.Scatter(x=df_target.index, y=eq_sma_pct, mode="lines", name="Chỉ SMA", line=dict(color="#f97316", width=1.5, dash="dash")))
    fig_equity.add_trace(go.Scatter(x=df_target.index, y=eq_obv_pct, mode="lines", name="Chỉ OBV", line=dict(color="#10b981", width=1.5, dash="dot")))
    fig_equity.add_trace(go.Scatter(x=df_target.index, y=eq_bh_pct, mode="lines", name="Mua & Nắm Giữ (Buy & Hold)", line=dict(color="#94a3b8", width=1.2)))

    fig_equity.update_layout(
        yaxis_title="Lợi Nhuận Tích Lũy (%)",
        xaxis_title="Thời Gian",
        height=500,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # Lịch sử các lệnh giao dịch (Trade Log)
    st.markdown("#### 📜 Nhật Ký Giao Dịch Chi Tiết (Trade Log)")
    trades_target = tr_test_comb if dataset_view == "Tập Kiểm Định (Test)" else tr_train_comb
    if trades_target.empty:
        st.info("Không có lệnh nào được thực thi trong giai đoạn này.")
    else:
        trades_display = trades_target.copy()
        if "EntryTime" in trades_display.columns:
            trades_display["EntryTime"] = pd.to_datetime(trades_display["EntryTime"]).dt.strftime('%d/%m/%Y')
        if "ExitTime" in trades_display.columns:
            trades_display["ExitTime"] = pd.to_datetime(trades_display["ExitTime"]).dt.strftime('%d/%m/%Y')
        st.dataframe(trades_display, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: TỐI ƯU HÓA THAM SỐ
# ------------------------------------------------------------------------------
with tab_optimization:
    st.markdown('<div class="section-title">Quy Trình Tối Ưu Hóa Tham Số Bằng Thuật Toán Bayesian (Hyperopt TPE)</div>', unsafe_allow_html=True)

    st.markdown("""
    #### 💡 Nguyên lý và Quy tắc Thực nghiệm:
    1. **Ngăn ngừa Rò Rỉ Dữ Liệu (No Data Leakage / Look-ahead Bias)**:
       - Quá trình tìm kiếm tham số tối ưu bằng thuật toán Tree-structured Parzen Estimator (TPE) của Hyperopt **chỉ được phép thực hiện trên tập Train**.
       - Tập Test hoàn toàn không được can thiệp vào quá trình tìm kiếm tham số.
    2. **Không Gian Tìm Kiếm (Search Space)**:
       - `ma_short`: Quần thể từ **10 đến 150**, bước nhảy 5.
       - `ma_long`: Quần thể từ **160 đến 400**, bước nhảy 5 (ràng buộc: `ma_short < ma_long`).
       - `obv_window`: Quần thể từ **3 đến 100**, bước nhảy 1.
    3. **Hàm Mục Tiêu (Objective Fitness)**:
       - Có thể lựa chọn tối ưu theo: Tối đa hóa **Lợi nhuận (Return)**, Tối đa hóa **Sharpe Ratio**, hoặc Giảm thiểu **Max Drawdown**.
    """)

    st.info(f"""
    **Bộ tham số đang kích hoạt trong phiên làm việc hiện tại:**
    - 🔹 SMA Ngắn (`ma_short`): **{st.session_state.ma_short}**
    - 🔹 SMA Dài (`ma_long`): **{st.session_state.ma_long}**
    - 🔹 OBV Window (`obv_window`): **{st.session_state.obv_window}**
    - 🔹 Cơ chế kết hợp: **{'Giao cắt đồng thời (Strict AND)' if strategy_mode == 'strict_and' else 'Regime Trend Filter + Trigger'}**
    """)

# ------------------------------------------------------------------------------
# TAB 4: PHÂN TÍCH & KHUYẾN NGHỊ ĐỊNH LƯỢNG
# ------------------------------------------------------------------------------
with tab_insights:
    st.markdown('<div class="section-title">Nhận Định Định Lượng Chuyên Sâu từ Nhóm Nghiên Cứu</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Phân Tích Hiện Tượng 0 Giao Dịch trên Tập Test (The "Zero-Trade" Bottleneck)
    - Trong notebook gốc, khi áp dụng quy tắc kết hợp:
      $$BUY = (SMA_{cross\_up}) \land (OBV_{cross\_up})$$
      nghĩa là **cả hai chỉ báo kỹ thuật phải đồng thời xuất hiện điểm giao cắt (crossover) trong đúng cùng một phiên giao dịch**.
    - **Vấn đề định lượng**: Giao cắt SMA chu kỳ trung/dài hạn (ví dụ SMA 150 và SMA 210) chỉ xảy ra vài lần trong nhiều năm. Việc đòi hỏi đường OBV cũng tình cờ cắt đường trung bình động của nó trong **chính xác cùng 1 ngày** đó là một điều kiện cực kỳ ngặt nghèo (xác suất xảy ra gần như bằng 0).
    - Đó là lý do trên tập Train chỉ có đúng **1 lệnh** được kích hoạt, và trên tập Test có **0 lệnh** phát sinh (Lợi nhuận = 0.0%).

    ### 2. Hiện Tượng Quá Khớp Dữ Liệu (Overfitting) Của Chiến Lược Đơn Lẻ
    - **Chiến lược OBV đơn lẻ**:
      - Trên tập Train: Đạt lợi nhuận khủng **+744.28%** với 247 lệnh giao dịch (do Hyperopt tìm ra cửa sổ rất ngắn `obv_window = 5`).
      - Trên tập Test: Lợi nhuận sụt giảm thê thảm xuống chỉ còn **+2.55%** và Sharpe chỉ còn **0.16**.
      - *Kết luận*: Việc tối ưu hóa một chỉ báo dao động nhanh như OBV với cửa sổ quá nhỏ (5 phiên) trên tập Train dẫn đến hiện tượng **Curve-Fitting (Học vẹt dữ liệu quá khứ)**, không duy trì được hiệu quả trong tương lai.

    ### 3. Đề Xuất Cải Tiến Cấu Trúc Chiến Lược (Best Practices in Quantitative Trading)
    Để khắc phục các hạn chế trên và tạo ra một chiến lược giao dịch định lượng thực chiến, nhóm đề xuất hai giải pháp:
    - **Giải pháp 1: Chuyển sang mô hình Trạng Thái & Kích Hoạt (Regime Filter + Momentum Trigger)**
      - Thay vì bắt 2 sự kiện cắt cùng ngày, ta dùng **SMA làm bộ lọc xu hướng vĩ mô (Trend Filter)**: Khi $SMA_{short} > SMA_{long}$ (thị trường đang trong pha Uptrend).
      - Sử dụng **OBV làm điểm kích hoạt (Trigger)**: Trong pha Uptrend, mỗi khi OBV cắt lên $MA(OBV)$, hệ thống sẽ mở lệnh Mua.
      - *Bạn có thể thử nghiệm ngay chế độ này tại thanh Sidebar bằng cách chọn mục: "Bộ Lọc Xu Hướng + Điểm Kích Hoạt"!*
    - **Giải pháp 2: Bổ sung Cơ chế Quản Trị Rủi Ro (Risk Management)**
      - Thêm mức cắt lỗ tự động (**Trailing Stop-Loss / ATR Stop**) thay vì chỉ dựa vào tín hiệu Death Cross trễ của SMA để thoát lệnh.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">
    Ứng dụng được xây dựng trên nền tảng <b>Streamlit & Plotly</b> | Dự án Định lượng: Chiến lược SMA + OBV | © 2024 Nhóm 5
</div>
""", unsafe_allow_html=True)
