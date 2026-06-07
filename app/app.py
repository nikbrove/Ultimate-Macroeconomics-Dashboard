import streamlit as st
import pandas as pd
import plotly.express as px
from fredapi import Fred

# Configurazione Pagina
st.set_page_config(page_title="Ultimate Macro Dashboard", layout="wide")

# Recupero Chiave API dai Secrets (Metodo Standard Streamlit)
try:
    # Cerchiamo la chiave sia in minuscolo che in maiuscolo per sicurezza
    if "fred_api_key" in st.secrets:
        key = st.secrets["fred_api_key"]
    elif "FRED_API_KEY" in st.secrets:
        key = st.secrets["FRED_API_KEY"]
    else:
        st.error("Chiave API 'fred_api_key' non trovata nei Secrets!")
        st.stop()
    
    fred = Fred(api_key=key)
except Exception as e:
    st.error(f"Errore inizializzazione FRED: {e}")
    st.stop()

st.title("🏛️ Ultimate Macroeconomics Dashboard")
st.markdown("Dati reali in tempo reale dai server FRED (Federal Reserve)")

# Sidebar per selezione
country_map = {"USA": "USA", "Eurozona": "Euro Area", "Giappone": "JPN", "UK": "GBR"}
target_country = st.sidebar.selectbox("Seleziona Area Geografica", list(country_map.keys()))

# Indicatori Macro Core (Codici FRED REALI)
metrics = {
    "PIL (GDP)": "GDP" if target_country == "USA" else "CPMNACSCAB1GQEL",
    "Inflazione (CPI)": "CPIAUCSL" if target_country == "USA" else "CPALTT01EZM657N",
    "Tasso Disoccupazione": "UNRATE" if target_country == "USA" else "LRHUTTTTEZM156S",
    "Tassi Interesse": "FEDFUNDS" if target_country == "USA" else "ECBASWIND"
}

selected_metric = st.sidebar.radio("Scegli Indicatore", list(metrics.keys()))

# Recupero Dati Live
with st.spinner('Scaricamento dati da FRED in corso...'):
    try:
        series_id = metrics[selected_metric]
        data = fred.get_series(series_id)
        df = pd.DataFrame(data, columns=["Valore"])
        df.index.name = "Data"
        
        # Grafico Plotly
        fig = px.line(df, y="Valore", title=f"{selected_metric} - {target_country}", 
                     template="plotly_dark", color_discrete_sequence=['#00d1ff'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Ultimi rilascio dati:")
        st.write(df.tail(10))
        
    except Exception as e:
        st.error(f"Errore nel caricamento della serie {selected_metric}: {e}")
