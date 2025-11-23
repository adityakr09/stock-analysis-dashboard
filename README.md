# 📊 Stock Market Analysis & Forecast Dashboard

An interactive stock analysis dashboard built using **Python, Pandas, Plotly, and Streamlit**.  
This project analyzes real NSE stocks and provides:

### 🔍 Core Features
| Feature | Description |
|--------|-------------|
| 📈 Technical Indicators | SMA20, SMA50, Daily Returns, Volatility |
| 💡 Trend Forecast | Explainable linear regression-based trend model |
| 📉 ARIMA Forecast | Advanced statistical model for future prices |
| 🛒 Buy/Sell Signals | Based on moving average crossover strategy |
| 🖥 Interactive UI | Choose stocks, indicators, forecast days |

---

### 📌 Why This Project?
Most stock projects online use ML they can’t explain.  
This one balances **explainability + real forecasting**, ideal for:

- 📌 **Data Analyst**
- 📌 **Python Developer**
- 📌 **Finance/Data roles**

It demonstrates:
- Clean code
- Real‐world data workflows
- Understanding of financial logic

---

### 🛠 Technologies Used
- **Python**
- **Pandas, NumPy**
- **Plotly**
- **Streamlit**
- **pmdarima (ARIMA)**

---

### 🚀 How to Run Locally

```bash
# 1) Clone this repository
git clone https://github.com/your-username/stock-analysis.git
cd stock-analysis

# 2) Create virtual environment (optional)
python -m venv venv
# Activate it (Windows)
venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the dashboard
streamlit run app.py
