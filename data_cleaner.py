import pandas as pd
import numpy as np
import yfinance as yf

nasdaq_tickers = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "PLTR", 
    "ASML", "NFLX", "COST", "AMD", "MU", "CSCO", "AZN", "APP", "LRCX", "TMUS", 
    "SHOP", "AMAT", "ISRG", "LIN", "PEP", "INTU", "QCOM", "AMGN", "BKNG", "INTC", 
    "KLAC", "PDD", "TXN", "GILD", "ADBE", "ADI", "PANW", "HON", "CRWD", "ARM", 
    "VRTX", "CEG", "CMCSA", "ADP", "MELI", "DASH", "SBUX", "SNPS", "CDNS", "MAR", 
    "ABNB", "REGN", "ORLY", "CTAS", "MNST", "MRVL", "WBD", "MDLZ", "CSX", "ADSK", 
    "AEP", "FTNT", "TRI", "ROST", "PCAR", "WDAY", "NXPI", "PYPL", "IDXX", "EA", 
    "ROP", "DDOG", "FAST", "AXON", "TTWO", "MSTR", "BKR", "EXC", "XEL", "TEAM", 
    "FANG", "CTSH", "PAYX", "CCEP", "KDP", "GEHC", "CPRT", "ZS", "MCHP", "ODFL", 
    "VRSK", "KHC", "CSGP", "CHTR", "DXCM", "BIIB", "LULU", "ON", "GFS", "TTD", "CDW"
]

start_date = "2024-12-24"
end_date_inclusive = "2025-12-24"
end_date = "2025-12-25"  # pour inclure le 24/12/2025

# --- 1) Download (Close ou Adj Close) ---
# Si tu veux STRICT "prix de clôture": utilise "Close"
# Si tu veux mieux pour stats (splits/dividendes): utilise "Adj Close"
field = "Close"  # change en "Adj Close" si tu préfères

raw = yf.download(
    tickers=nasdaq_tickers,
    start=start_date,
    end=end_date,
    auto_adjust=False,
    group_by="column",
    progress=False,
    threads=True
)

# yfinance renvoie souvent un MultiIndex de colonnes (ex: ('Close','AAPL'))
prices = raw[field].copy()

# --- 2) Nettoyage de base ---
# Tri des dates, suppression doublons éventuels
prices = prices.sort_index()
prices = prices[~prices.index.duplicated(keep="first")]

# Option: supprimer les tickers totalement vides
all_nan = prices.columns[prices.isna().all()].tolist()
if all_nan:
    print("Tickers sans données (supprimés):", all_nan)
    prices = prices.drop(columns=all_nan)

# Option PCA: garder uniquement les titres avec assez de données
min_coverage = 0.95  # 95% des jours doivent être présents
coverage = 1 - prices.isna().mean()
keep = coverage[coverage >= min_coverage].index
dropped = coverage[coverage < min_coverage].index.tolist()

if dropped:
    print(f"Tickers retirés (couverture < {min_coverage:.0%}):", dropped)

prices = prices[keep]

# Option: pour la PCA/corr, on veut un tableau complet -> on peut forward-fill puis dropna
prices_ffill = prices.ffill()
prices_clean = prices_ffill.dropna(axis=0, how="any")  # garde seulement les dates où tout est dispo

print("Shape prices_clean:", prices_clean.shape)
print("Période effective:", prices_clean.index.min().date(), "→", prices_clean.index.max().date())

# --- 3) Log returns quotidiens ---
log_returns = np.log(prices_clean / prices_clean.shift(1)).dropna(how="any")

print("Shape log_returns:", log_returns.shape)
print(log_returns.head())

# À ce stade:
# - prices_clean : prix alignés (DataFrame dates x tickers)
# - log_returns  : log-returns quotidiens (base pour corr + PCA)
