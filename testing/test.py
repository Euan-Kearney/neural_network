import yfinance as yf
import pandas as pd
import numpy as np

nvidia = yf.Ticker("NVDA")
print(nvidia.history(period="1y"))

