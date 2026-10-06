import sys
import os
print('python', sys.executable)
# ensure project root is on sys.path so `src` package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
modules = ['pandas','yfinance','src.data_loader','src.eda','src.forecasting']
for m in modules:
    try:
        __import__(m)
        print(m, 'OK')
    except Exception as e:
        print(m, 'ERROR', type(e).__name__, e)
        raise
