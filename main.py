import yfinance as yf
import pandas as pd
import pandas_ta as ta
import ollama
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import customtkinter as ctk
import threading
from typing import Optional, Dict, Any


# ============================================================
# FORMATTING HELPERS
# ============================================================
def _sf(x, default=None):
    """Safe float."""
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def fmt_num(x, nd=2):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "N/A"


def fmt_pct(x, nd=2):
    if x is None:
        return "N/A"
    try:
        return f"{float(x) * 100:.{nd}f}%"
    except Exception:
        return "N/A"


def fmt_money(x, nd=2):
    if x is None:
        return "N/A"
    try:
        return f"${float(x):,.{nd}f}"
    except Exception:
        return "N/A"


def fmt_big_money(x):
    """Format big $ values in B/M/K if possible."""
    if x is None:
        return "N/A"
    try:
        v = float(x)
        abs_v = abs(v)
        if abs_v >= 1e12:
            return f"${v/1e12:,.2f}T"
        if abs_v >= 1e9:
            return f"${v/1e9:,.2f}B"
        if abs_v >= 1e6:
            return f"${v/1e6:,.2f}M"
        if abs_v >= 1e3:
            return f"${v/1e3:,.2f}K"
        return f"${v:,.0f}"
    except Exception:
        return "N/A"


def first_col_like(df: pd.DataFrame, prefix: str) -> Optional[str]:
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    return cols[0] if cols else None


def latest_statement_value(stmt: pd.DataFrame, row_name: str) -> Optional[float]:
    """
    yfinance statements are DataFrames with rows as line items and columns as dates.
    Returns latest column value for a given row.
    """
    try:
        if stmt is None or stmt.empty:
            return None
        if row_name not in stmt.index:
            return None
        latest_col = stmt.columns[0]  # yfinance usually returns latest first
        return _sf(stmt.loc[row_name, latest_col], None)
    except Exception:
        return None


# ============================================================
# --- DEEP DATA ENGINE ---
# ============================================================
def get_pro_analysis(ticker_symbol: str):
    """
    Returns:
      - df: price dataframe with deep technical indicators + levels
      - fundamentals: dict with raw key fundamental fields + computed metrics + formatted dashboard text
    """
    try:
        ticker = yf.Ticker(ticker_symbol)

        # Use 2y daily to support 200D + 52W structure; still reasonably fast.
        df = ticker.history(period="2y", interval="1d", auto_adjust=False)
        if df.empty or len(df) < 120:
            return None, None

        df = df.copy()
        df.dropna(inplace=True)

        # ----------------------------
        # TECHNICALS (Deeper Stack)
        # ----------------------------
        # Trend + momentum
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.sma(length=200, append=True)          # long-term regime
        df.ta.rsi(length=14, append=True)
        df.ta.stochrsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.adx(length=14, append=True)

        # Volatility + bands
        df.ta.atr(length=14, append=True)
        df.ta.bbands(length=20, append=True)

        # Volume / flow
        df.ta.obv(append=True)
        df.ta.mfi(length=14, append=True)

        # Trend-following filter
        # (supertrend column names vary slightly; still very useful when present)
        try:
            df.ta.supertrend(length=10, multiplier=3.0, append=True)
        except Exception:
            pass

        # Relative volume (confirmation / exhaustion)
        df["VOL_MA20"] = df["Volume"].rolling(window=20).mean()
        df["RVOL"] = df["Volume"] / df["VOL_MA20"]

        # ----------------------------
        # MARKET GEOMETRY (Levels)
        # ----------------------------
        df["RES_50"] = df["High"].rolling(window=50).max()
        df["SUP_50"] = df["Low"].rolling(window=50).min()
        df["RES_200"] = df["High"].rolling(window=200).max()
        df["SUP_200"] = df["Low"].rolling(window=200).min()

        # 52-week structure (252 trading days)
        df["HI_52W"] = df["High"].rolling(window=252).max()
        df["LO_52W"] = df["Low"].rolling(window=252).min()

        # ----------------------------
        # FUNDAMENTALS (Deeper Stack)
        # ----------------------------
        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            info = {}

        # Statements (annual)
        try:
            fin = ticker.financials  # income statement
        except Exception:
            fin = pd.DataFrame()

        try:
            bs = ticker.balance_sheet
        except Exception:
            bs = pd.DataFrame()

        try:
            cf = ticker.cashflow
        except Exception:
            cf = pd.DataFrame()

        # Core info fields (availability varies by ticker/market)
        market_cap = _sf(info.get("marketCap"), None)
        enterprise_value = _sf(info.get("enterpriseValue"), None)

        trailing_pe = _sf(info.get("trailingPE"), None)
        forward_pe = _sf(info.get("forwardPE"), None)
        peg = _sf(info.get("pegRatio"), None)
        ps = _sf(info.get("priceToSalesTrailing12Months"), None)
        pb = _sf(info.get("priceToBook"), None)

        beta = _sf(info.get("beta"), None)
        div_yield = _sf(info.get("dividendYield"), None)
        payout = _sf(info.get("payoutRatio"), None)

        roe = _sf(info.get("returnOnEquity"), None)
        roa = _sf(info.get("returnOnAssets"), None)
        gross_m = _sf(info.get("grossMargins"), None)
        op_m = _sf(info.get("operatingMargins"), None)
        net_m = _sf(info.get("profitMargins"), None)

        rev_growth = _sf(info.get("revenueGrowth"), None)
        earn_growth = _sf(info.get("earningsGrowth"), None)

        current_ratio = _sf(info.get("currentRatio"), None)
        quick_ratio = _sf(info.get("quickRatio"), None)
        debt_to_equity = _sf(info.get("debtToEquity"), None)

        total_cash = _sf(info.get("totalCash"), None)
        total_debt = _sf(info.get("totalDebt"), None)

        # Try to compute/confirm from statements where possible
        ebitda = _sf(info.get("ebitda"), None)
        if ebitda is None:
            ebitda = latest_statement_value(fin, "EBITDA")

        revenue = _sf(info.get("totalRevenue"), None)
        if revenue is None:
            revenue = latest_statement_value(fin, "Total Revenue")

        net_income = latest_statement_value(fin, "Net Income")

        # Cash flow
        fcf = _sf(info.get("freeCashflow"), None)
        if fcf is None:
            cfo = latest_statement_value(cf, "Total Cash From Operating Activities")
            capex = latest_statement_value(cf, "Capital Expenditures")  # often negative
            if cfo is not None and capex is not None:
                fcf = cfo + capex  # capex negative -> subtract in effect

        op_cf = latest_statement_value(cf, "Total Cash From Operating Activities")

        # Debt metrics
        net_debt = None
        if total_debt is not None and total_cash is not None:
            net_debt = total_debt - total_cash

        ev_to_ebitda = None
        if enterprise_value is not None and ebitda not in (None, 0):
            ev_to_ebitda = enterprise_value / ebitda

        fcf_yield = None
        if market_cap not in (None, 0) and fcf is not None:
            fcf_yield = fcf / market_cap

        net_debt_to_ebitda = None
        if net_debt is not None and ebitda not in (None, 0):
            net_debt_to_ebitda = net_debt / ebitda

        # Short/ownership (optional)
        short_pct_float = _sf(info.get("shortPercentOfFloat"), None)
        short_ratio = _sf(info.get("shortRatio"), None)
        inst_own = _sf(info.get("heldPercentInstitutions"), None)
        insider_own = _sf(info.get("heldPercentInsiders"), None)

        target = _sf(info.get("targetMeanPrice"), None)

        name = info.get("shortName") or info.get("longName") or ticker_symbol

        # Build the NEW “Key Fundamental Metrics” dashboard text
        fund_dashboard = (
            f"KEY FUNDAMENTAL METRICS DASHBOARD\n"
            f"------------------------------------------------------------\n"
            f"Size / Risk\n"
            f"• Market Cap:        {fmt_big_money(market_cap)}\n"
            f"• Enterprise Value:  {fmt_big_money(enterprise_value)}\n"
            f"• Beta:              {fmt_num(beta, 2)}\n\n"
            f"Valuation\n"
            f"• P/E (TTM):         {fmt_num(trailing_pe, 2)}\n"
            f"• P/E (Forward):     {fmt_num(forward_pe, 2)}\n"
            f"• PEG:               {fmt_num(peg, 2)}\n"
            f"• P/S (TTM):         {fmt_num(ps, 2)}\n"
            f"• P/B:               {fmt_num(pb, 2)}\n"
            f"• EV/EBITDA:         {fmt_num(ev_to_ebitda, 2)}\n"
            f"• FCF Yield:         {fmt_pct(fcf_yield, 2)}\n\n"
            f"Growth\n"
            f"• Revenue Growth:    {fmt_pct(rev_growth, 2)}\n"
            f"• Earnings Growth:   {fmt_pct(earn_growth, 2)}\n\n"
            f"Profitability\n"
            f"• Gross Margin:      {fmt_pct(gross_m, 2)}\n"
            f"• Operating Margin:  {fmt_pct(op_m, 2)}\n"
            f"• Net Margin:        {fmt_pct(net_m, 2)}\n"
            f"• ROE:               {fmt_pct(roe, 2)}\n"
            f"• ROA:               {fmt_pct(roa, 2)}\n\n"
            f"Balance Sheet / Liquidity\n"
            f"• Current Ratio:     {fmt_num(current_ratio, 2)}\n"
            f"• Quick Ratio:       {fmt_num(quick_ratio, 2)}\n"
            f"• Debt/Equity:       {fmt_num(debt_to_equity, 2)}\n"
            f"• Total Debt:        {fmt_big_money(total_debt)}\n"
            f"• Total Cash:        {fmt_big_money(total_cash)}\n"
            f"• Net Debt:          {fmt_big_money(net_debt)}\n"
            f"• Net Debt/EBITDA:   {fmt_num(net_debt_to_ebitda, 2)}\n\n"
            f"Cash Flow / Shareholder Return\n"
            f"• Operating Cashflow:{fmt_big_money(op_cf)}\n"
            f"• Free Cash Flow:    {fmt_big_money(fcf)}\n"
            f"• Dividend Yield:    {fmt_pct(div_yield, 2)}\n"
            f"• Payout Ratio:      {fmt_pct(payout, 2)}\n\n"
            f"Positioning (if available)\n"
            f"• Institutional Own: {fmt_pct(inst_own, 2)}\n"
            f"• Insider Own:       {fmt_pct(insider_own, 2)}\n"
            f"• Short % of Float:  {fmt_pct(short_pct_float, 2)}\n"
            f"• Short Ratio (DTC): {fmt_num(short_ratio, 2)}\n"
        )

        fundamentals = {
            "name": name,
            "target": target,
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "peg": peg,
            "ps": ps,
            "pb": pb,
            "ev_to_ebitda": ev_to_ebitda,
            "fcf_yield": fcf_yield,
            "rev_growth": rev_growth,
            "earn_growth": earn_growth,
            "gross_m": gross_m,
            "op_m": op_m,
            "net_m": net_m,
            "roe": roe,
            "roa": roa,
            "current_ratio": current_ratio,
            "quick_ratio": quick_ratio,
            "debt_to_equity": debt_to_equity,
            "total_debt": total_debt,
            "total_cash": total_cash,
            "net_debt": net_debt,
            "net_debt_to_ebitda": net_debt_to_ebitda,
            "op_cf": op_cf,
            "fcf": fcf,
            "div_yield": div_yield,
            "payout": payout,
            "beta": beta,
            "short_pct_float": short_pct_float,
            "short_ratio": short_ratio,
            "inst_own": inst_own,
            "insider_own": insider_own,
            "dashboard_text": fund_dashboard,
        }

        return df, fundamentals

    except Exception as e:
        print(f"Deep Analysis Error: {e}")
        return None, None


# ============================================================
# --- GUI APPLICATION ---
# ============================================================
class TradingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.df = None
        self.fund = None

        self.title("AI Quant Research Terminal v7.2")
        self.geometry("1200x1200")
        ctk.set_appearance_mode("dark")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. SIDEBAR
        self.sidebar = ctk.CTkFrame(self, width=280)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self.sidebar, text="DEEP SCANNER", font=("Arial", 22, "bold")).pack(pady=20)
        self.ticker_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Ticker (e.g. AAPL)")
        self.ticker_entry.pack(pady=10, padx=20)
        self.btn_run = ctk.CTkButton(self.sidebar, text="GENERATE REPORT", command=self.start_thread, fg_color="#1e3799")
        self.btn_run.pack(pady=10, padx=20)
        self.status = ctk.CTkLabel(self.sidebar, text="System Standby", text_color="gray")
        self.status.pack(side="bottom", pady=20)

        # 2. MAIN AREA
        self.main_frame = ctk.CTkScrollableFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # HEADER: PRICE & TICKER
        self.ticker_label = ctk.CTkLabel(self.main_frame, text="--", font=("Arial", 28, "bold"))
        self.ticker_label.pack(pady=(10, 0))
        self.price_label = ctk.CTkLabel(self.main_frame, text="$0.00", font=("Arial", 56, "bold"), text_color="#3498db")
        self.price_label.pack(pady=(0, 5))

        # STATUS TAGS
        self.tag_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.tag_frame.pack(fill="x", pady=10)
        self.val_tag = ctk.CTkLabel(self.tag_frame, text="VALUATION: --", font=("Arial", 16, "bold"), height=50,
                                    corner_radius=8, fg_color="#333")
        self.val_tag.pack(side="left", padx=5, expand=True, fill="x")
        self.tech_tag = ctk.CTkLabel(self.tag_frame, text="TECHNICAL: --", font=("Arial", 16, "bold"), height=50,
                                     corner_radius=8, fg_color="#333")
        self.tech_tag.pack(side="left", padx=5, expand=True, fill="x")

        # SECTION: MARKET GEOMETRY
        ctk.CTkLabel(self.main_frame, text="MARKET GEOMETRY & LIQUIDITY LEVELS", font=("Arial", 14, "bold")).pack(pady=(15, 5))
        self.levels_frame = ctk.CTkFrame(self.main_frame, fg_color="#1a1a1a")
        self.levels_frame.pack(fill="x", padx=10, pady=5)
        self.res_lbl = ctk.CTkLabel(self.levels_frame, text="RESISTANCE: --", font=("Consolas", 18, "bold"), text_color="#ff7675")
        self.res_lbl.grid(row=0, column=0, padx=40, pady=15)
        self.sup_lbl = ctk.CTkLabel(self.levels_frame, text="SUPPORT: --", font=("Consolas", 18, "bold"), text_color="#55efc4")
        self.sup_lbl.grid(row=0, column=1, padx=40, pady=15)
        self.hi52_lbl = ctk.CTkLabel(self.levels_frame, text="52W HIGH: --", font=("Consolas", 18, "bold"), text_color="#feca57")
        self.hi52_lbl.grid(row=1, column=0, padx=40, pady=(0, 15))
        self.lo52_lbl = ctk.CTkLabel(self.levels_frame, text="52W LOW: --", font=("Consolas", 18, "bold"), text_color="#54a0ff")
        self.lo52_lbl.grid(row=1, column=1, padx=40, pady=(0, 15))

        # SECTION: TRADE EXECUTION PLAN
        ctk.CTkLabel(self.main_frame, text="EXECUTION PLAN (ATR ADAPTIVE)", font=("Arial", 14, "bold")).pack(pady=(15, 5))
        self.exec_frame = ctk.CTkFrame(self.main_frame, fg_color="#0c2461", corner_radius=10)
        self.exec_frame.pack(fill="x", padx=10, pady=5)
        self.buy_lbl = ctk.CTkLabel(self.exec_frame, text="BUY: --", font=("Consolas", 18, "bold"))
        self.buy_lbl.grid(row=0, column=0, padx=30, pady=20)
        self.sl_lbl = ctk.CTkLabel(self.exec_frame, text="SL: --", font=("Consolas", 18, "bold"), text_color="#ff7675")
        self.sl_lbl.grid(row=0, column=1, padx=30, pady=20)
        self.tp_lbl = ctk.CTkLabel(self.exec_frame, text="TP: --", font=("Consolas", 18, "bold"), text_color="#55efc4")
        self.tp_lbl.grid(row=0, column=2, padx=30, pady=20)

        # SECTION: QUANT RAW (Tech + quick blend)
        ctk.CTkLabel(self.main_frame, text="QUANTITATIVE RAW DATA (TECHNICAL SNAPSHOT)", font=("Arial", 13, "bold")).pack(pady=(15, 0))
        self.metrics_box = ctk.CTkTextbox(self.main_frame, height=170, font=("Consolas", 14), fg_color="#1a1a1a")
        self.metrics_box.pack(fill="x", padx=10, pady=5)

        # NEW SECTION: KEY FUNDAMENTAL METRICS (requested)
        ctk.CTkLabel(self.main_frame, text="KEY FUNDAMENTAL METRICS (VALUATION • GROWTH • CASH FLOW • BALANCE SHEET)",
                     font=("Arial", 13, "bold"), text_color="#82ccdd").pack(pady=(15, 0))
        self.fund_metrics_box = ctk.CTkTextbox(self.main_frame, height=300, font=("Consolas", 13), fg_color="#1a1a1a")
        self.fund_metrics_box.pack(fill="x", padx=10, pady=5)

        # SECTION: AI DEEP DIVE
        ctk.CTkLabel(self.main_frame, text="DEEP FUNDAMENTAL ANALYSIS (MOAT • QUALITY • VALUATION • RISKS)",
                     font=("Arial", 13, "bold"), text_color="#38ada9").pack(pady=(15, 0))
        self.fund_ai = ctk.CTkTextbox(self.main_frame, height=220, wrap="word")
        self.fund_ai.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(self.main_frame, text="DEEP TECHNICAL ANALYSIS (REGIME • CONFLUENCE • LEVELS • PLAN)",
                     font=("Arial", 13, "bold"), text_color="#e58e26").pack(pady=(15, 0))
        self.tech_ai = ctk.CTkTextbox(self.main_frame, height=220, wrap="word")
        self.tech_ai.pack(fill="x", padx=10, pady=5)

        self.btn_chart = ctk.CTkButton(self.main_frame, text="📊 Open Advanced Multi-Pane Chart", command=self.show_chart, state="disabled")
        self.btn_chart.pack(pady=20)

    def start_thread(self):
        ticker = self.ticker_entry.get().upper().strip()
        if not ticker:
            return
        self.status.configure(text=f"Deep Analysis: {ticker}...", text_color="orange")
        threading.Thread(target=self.process, args=(ticker,), daemon=True).start()

    def process(self, ticker: str):
        df, f = get_pro_analysis(ticker)
        if df is None or f is None:
            self.after(0, lambda: self.status.configure(text="Data Error", text_color="red"))
            return

        self.df = df
        self.fund = f
        curr = float(df["Close"].iloc[-1])

        # Pull key columns safely
        rsi_col = first_col_like(df, "RSI")
        adx_col = first_col_like(df, "ADX")
        atr_col = first_col_like(df, "ATR")

        ema50_col = first_col_like(df, "EMA_50")
        sma200_col = first_col_like(df, "SMA_200")

        macd_line_col = first_col_like(df, "MACD_")
        macd_hist_col = first_col_like(df, "MACDh_")
        macd_sig_col = first_col_like(df, "MACDs_")

        rsi = float(df[rsi_col].iloc[-1]) if rsi_col else None
        adx = float(df[adx_col].iloc[-1]) if adx_col else None
        atr = float(df[atr_col].iloc[-1]) if atr_col else None

        ema50 = float(df[ema50_col].iloc[-1]) if ema50_col else None
        sma200 = float(df[sma200_col].iloc[-1]) if sma200_col else None

        macd_line = float(df[macd_line_col].iloc[-1]) if macd_line_col else None
        macd_hist = float(df[macd_hist_col].iloc[-1]) if macd_hist_col else None
        macd_sig = float(df[macd_sig_col].iloc[-1]) if macd_sig_col else None

        rvol = float(df["RVOL"].iloc[-1]) if "RVOL" in df.columns else None

        # ----------------------------
        # VALUATION TAG (improved heuristic)
        # ----------------------------
        # If target is present, retain your target-vs-price logic, but add an internal quality/price overlay.
        target = f.get("target")
        target_based = None
        if target not in (None, 0):
            if target > curr * 1.10:
                target_based = "UNDERVALUED"
            elif target < curr * 0.90:
                target_based = "OVERVALUED"
            else:
                target_based = "FAIR VALUE"

        # Quality/price heuristics (non-sector-adjusted, but more informative than a single threshold)
        score = 0
        if f.get("fcf_yield") is not None:
            score += 1 if f["fcf_yield"] >= 0.05 else (-1 if f["fcf_yield"] <= 0.02 else 0)
        if f.get("ev_to_ebitda") is not None:
            score += 1 if f["ev_to_ebitda"] <= 12 else (-1 if f["ev_to_ebitda"] >= 20 else 0)
        if f.get("forward_pe") is not None:
            score += 1 if f["forward_pe"] <= 18 else (-1 if f["forward_pe"] >= 30 else 0)
        if f.get("roe") is not None:
            score += 1 if f["roe"] >= 0.15 else (-1 if f["roe"] <= 0.07 else 0)
        if f.get("net_debt_to_ebitda") is not None:
            score += 1 if f["net_debt_to_ebitda"] <= 2 else (-1 if f["net_debt_to_ebitda"] >= 4 else 0)

        if score >= 2:
            overlay = "ATTRACTIVE"
        elif score <= -2:
            overlay = "EXPENSIVE / RISK"
        else:
            overlay = "NEUTRAL"

        v_s = target_based if target_based else overlay
        if "UNDER" in v_s or "ATTRACTIVE" in v_s:
            v_c = "#2ecc71"
        elif "OVER" in v_s or "EXPENSIVE" in v_s:
            v_c = "#e74c3c"
        else:
            v_c = "#3498db"

        # ----------------------------
        # TECHNICAL TAG (trend regime + momentum)
        # ----------------------------
        regime = "RANGE / TRANSITION"
        if sma200 is not None and ema50 is not None and adx is not None:
            if curr > sma200 and ema50 > sma200 and adx >= 20:
                regime = "BULL TREND"
            elif curr < sma200 and ema50 < sma200 and adx >= 20:
                regime = "BEAR TREND"
            else:
                regime = "RANGE / TRANSITION"

        momentum = ""
        if rsi is not None:
            if rsi < 35:
                momentum = " (OVERSOLD)"
            elif rsi > 65:
                momentum = " (OVERBOUGHT)"
            else:
                momentum = " (NEUTRAL)"

        t_s = f"{regime}{momentum}"
        if "OVERSOLD" in t_s or "BULL" in t_s:
            t_c = "#2ecc71"
        elif "OVERBOUGHT" in t_s or "BEAR" in t_s:
            t_c = "#e74c3c"
        else:
            t_c = "#3498db"

        # ----------------------------
        # RAW TECH METRICS DISPLAY (expanded)
        # ----------------------------
        metric_txt = (
            f"--- TREND / MOMENTUM ---                 --- VOL / FLOW ---\n"
            f"RSI (14):      {fmt_num(rsi, 2):<15}      ATR (14):      {fmt_money(atr, 2)}\n"
            f"ADX (14):      {fmt_num(adx, 2):<15}      RVOL (vs 20D): {fmt_num(rvol, 2)}\n"
            f"EMA 50:        {fmt_money(ema50, 2):<15}  SMA 200:       {fmt_money(sma200, 2)}\n"
            f"MACD:          {fmt_num(macd_line, 3):<15}  Signal:        {fmt_num(macd_sig, 3)}\n"
            f"MACD Hist:     {fmt_num(macd_hist, 3):<15}\n"
        )

        # ----------------------------
        # TRADE PLAN (ATR + structure-aware)
        # ----------------------------
        res_50 = float(df["RES_50"].iloc[-1])
        sup_50 = float(df["SUP_50"].iloc[-1])
        hi_52w = float(df["HI_52W"].iloc[-1]) if not pd.isna(df["HI_52W"].iloc[-1]) else None
        lo_52w = float(df["LO_52W"].iloc[-1]) if not pd.isna(df["LO_52W"].iloc[-1]) else None

        # stop: below support and volatility cushion
        if atr is None:
            atr = max(0.01 * curr, 1.0)

        sl = min(curr - (1.5 * atr), sup_50 - (0.5 * atr))
        tp = curr + (3.0 * atr)

        # ----------------------------
        # AI PROMPTS (much deeper)
        # ----------------------------
        # Fundamental prompt: multi-dimension output
        f_p = f"""
You are a buy-side equity analyst. Write a deep but concise fundamental note for {ticker}.
Use bullet points and clear section headers. Keep it to ~12-16 bullets total.

Include:
1) Business/segment snapshot (1-2 bullets)
2) Quality & profitability (ROE, margins)
3) Growth profile (revenue/earnings growth if available)
4) Valuation (TTM/forward P/E, EV/EBITDA, P/S, FCF yield)
5) Balance sheet & solvency (current/quick, net debt, net debt/EBITDA)
6) Shareholder returns (dividend yield/payout if available)
7) 3 key bull-case catalysts and 3 key bear-case risks
8) What to monitor next quarter

Metrics (may contain N/A; reason around it):
- Market Cap {fmt_big_money(f.get("market_cap"))}, EV {fmt_big_money(f.get("enterprise_value"))}, Beta {fmt_num(f.get("beta"),2)}
- P/E TTM {fmt_num(f.get("trailing_pe"),2)}, P/E Fwd {fmt_num(f.get("forward_pe"),2)}, PEG {fmt_num(f.get("peg"),2)}
- EV/EBITDA {fmt_num(f.get("ev_to_ebitda"),2)}, P/S {fmt_num(f.get("ps"),2)}, P/B {fmt_num(f.get("pb"),2)}, FCF Yield {fmt_pct(f.get("fcf_yield"),2)}
- Rev Growth {fmt_pct(f.get("rev_growth"),2)}, Earn Growth {fmt_pct(f.get("earn_growth"),2)}
- Gross {fmt_pct(f.get("gross_m"),2)}, Oper {fmt_pct(f.get("op_m"),2)}, Net {fmt_pct(f.get("net_m"),2)}
- ROE {fmt_pct(f.get("roe"),2)}, ROA {fmt_pct(f.get("roa"),2)}
- Current {fmt_num(f.get("current_ratio"),2)}, Quick {fmt_num(f.get("quick_ratio"),2)}, D/E {fmt_num(f.get("debt_to_equity"),2)}
- Net Debt {fmt_big_money(f.get("net_debt"))}, Net Debt/EBITDA {fmt_num(f.get("net_debt_to_ebitda"),2)}
- Div Yield {fmt_pct(f.get("div_yield"),2)}, Payout {fmt_pct(f.get("payout"),2)}
- Street Target Mean {fmt_money(f.get("target"),2)}

Do not add disclaimers. Do not mention you are an AI.
""".strip()

        # Technical prompt: confluence-based regime read + levels + plan
        t_p = f"""
You are a technical strategist. Provide a deep technical read for {ticker} using the latest daily data.
Write bullet points, ~10-14 bullets total.

Include:
1) Trend regime (price vs SMA200 and EMA50, ADX strength)
2) Momentum (RSI + MACD line/signal + histogram direction)
3) Volatility (ATR and Bollinger bands implication)
4) Volume confirmation (RVOL + OBV/MFI if relevant)
5) Key levels: Support/Resistance (50D) and 52-week high/low
6) Tactical swing plan: entry logic, stop (invalidation), 2 targets, and risk/reward comment

Data:
- Price {fmt_money(curr,2)}
- RSI {fmt_num(rsi,1)}, ADX {fmt_num(adx,1)}
- EMA50 {fmt_money(ema50,2)}, SMA200 {fmt_money(sma200,2)}
- MACD {fmt_num(macd_line,3)}, Signal {fmt_num(macd_sig,3)}, Hist {fmt_num(macd_hist,3)}
- ATR {fmt_money(atr,2)}, RVOL {fmt_num(rvol,2)}
- Resistance(50D) {fmt_money(res_50,2)}, Support(50D) {fmt_money(sup_50,2)}
- 52W High {fmt_money(hi_52w,2)}, 52W Low {fmt_money(lo_52w,2)}

Do not add disclaimers. Do not mention you are an AI.
""".strip()

        try:
            f_res = ollama.generate(model="llama3", prompt=f_p)["response"]
            t_res = ollama.generate(model="llama3", prompt=t_p)["response"]
        except Exception:
            f_res = "AI Summary offline."
            t_res = "Technical AI offline."

        self.after(
            0,
            lambda: self.update_ui(
                f["name"],
                curr,
                v_s,
                v_c,
                t_s,
                t_c,
                sl,
                tp,
                metric_txt,
                f.get("dashboard_text", ""),
                f_res,
                t_res,
            ),
        )

    def update_ui(self, name, cp, v_s, v_c, t_s, t_c, sl, tp, m_txt, fund_metrics_txt, f_ai, t_ai):
        self.ticker_label.configure(text=str(name).upper())
        self.price_label.configure(text=f"${cp:.2f}")
        self.val_tag.configure(text=v_s, fg_color=v_c)
        self.tech_tag.configure(text=t_s, fg_color=t_c)

        # Market Geometry
        res, sup = float(self.df["RES_50"].iloc[-1]), float(self.df["SUP_50"].iloc[-1])
        hi52 = float(self.df["HI_52W"].iloc[-1]) if not pd.isna(self.df["HI_52W"].iloc[-1]) else None
        lo52 = float(self.df["LO_52W"].iloc[-1]) if not pd.isna(self.df["LO_52W"].iloc[-1]) else None

        self.res_lbl.configure(text=f"RESISTANCE: ${res:.2f}")
        self.sup_lbl.configure(text=f"SUPPORT: ${sup:.2f}")
        self.hi52_lbl.configure(text=f"52W HIGH: {fmt_money(hi52,2)}")
        self.lo52_lbl.configure(text=f"52W LOW:  {fmt_money(lo52,2)}")

        # Trade Execution
        self.buy_lbl.configure(text=f"BUY: ${cp:.2f}")
        self.sl_lbl.configure(text=f"SL: ${sl:.2f}")
        self.tp_lbl.configure(text=f"TP: ${tp:.2f}")

        self.metrics_box.delete("1.0", "end")
        self.metrics_box.insert("1.0", m_txt)

        self.fund_metrics_box.delete("1.0", "end")
        self.fund_metrics_box.insert("1.0", fund_metrics_txt)

        self.fund_ai.delete("1.0", "end")
        self.fund_ai.insert("1.0", f_ai)

        self.tech_ai.delete("1.0", "end")
        self.tech_ai.insert("1.0", t_ai)

        self.btn_chart.configure(state="normal")
        self.status.configure(text="Analysis Finalized", text_color="green")

    def show_chart(self):
        # 3-pane: Price+levels, RSI, MACD
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)

        fig.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df["Open"],
                high=self.df["High"],
                low=self.df["Low"],
                close=self.df["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        fig.add_hline(y=float(self.df["RES_50"].iloc[-1]), line_color="#ff7675", line_dash="dash",
                      annotation_text="RES(50D)", row=1, col=1)
        fig.add_hline(y=float(self.df["SUP_50"].iloc[-1]), line_color="#55efc4", line_dash="dash",
                      annotation_text="SUP(50D)", row=1, col=1)

        # RSI
        rsi_col = first_col_like(self.df, "RSI")
        if rsi_col:
            fig.add_trace(go.Scatter(x=self.df.index, y=self.df[rsi_col], name="RSI(14)"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", annotation_text="70", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", annotation_text="30", row=2, col=1)

        # MACD histogram
        macd_hist_col = first_col_like(self.df, "MACDh_")
        macd_line_col = first_col_like(self.df, "MACD_")
        macd_sig_col = first_col_like(self.df, "MACDs_")

        if macd_hist_col:
            fig.add_trace(go.Bar(x=self.df.index, y=self.df[macd_hist_col], name="MACD Hist"), row=3, col=1)
        if macd_line_col and macd_sig_col:
            fig.add_trace(go.Scatter(x=self.df.index, y=self.df[macd_line_col], name="MACD"), row=3, col=1)
            fig.add_trace(go.Scatter(x=self.df.index, y=self.df[macd_sig_col], name="Signal"), row=3, col=1)

        fig.update_layout(template="plotly_dark", height=950, xaxis_rangeslider_visible=False)
        fig.show()


if __name__ == "__main__":
    app = TradingApp()
    app.mainloop()
