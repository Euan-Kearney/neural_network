import yfinance as yf

nvidia = yf.Ticker("NVDA")
print(nvidia.history(period="1y"))

