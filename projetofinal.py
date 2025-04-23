# bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.title("Previsão de Renda")
st.subheader('As principais análises do Projeto')

#@st.cache_data()
renda = pd.read_csv('renda_tratada.csv')

st.write(f'O tamanho do conjunto de dados usado na análise: {renda.shape}')

st.markdown("Plotagem com as variáveis dependentes e target original")

fig, ax = plt.subplots()

sns.regplot(data=renda[renda.tipo_renda == 'Assalariado'], x='tempo_emprego', y='renda', ax=ax)
sns.regplot(data=renda[renda.tipo_renda == 'Empresário'], x='tempo_emprego', y='renda', ax=ax)
sns.regplot(data=renda[renda.tipo_renda == 'Pensionista'], x='tempo_emprego', y='renda', ax=ax)
sns.regplot(data=renda[renda.tipo_renda == 'Servidor público'], x='tempo_emprego', y='renda', ax=ax)
sns.regplot(data=renda[renda.tipo_renda == 'Bolsista'], x='tempo_emprego', y='renda', ax=ax)

st.pyplot(fig)

# transformação de variáveis núméricas essenciais
renda['tempo_emprego_log'] = np.log(renda.tempo_emprego)
renda['renda_log'] = np.log(renda.renda)

st.markdown("Plotagem com as variáveis dependentes e target transformada")

fig, ax = plt.subplots()

sns.regplot(data=renda[renda.tipo_renda == 'Assalariado'], x='tempo_emprego', y='renda_log', ax=ax)
sns.regplot(data=renda[renda.tipo_renda == 'Empresário'], x='tempo_emprego', y='renda_log', ax=ax)
sns.regplot(data=renda[renda.tipo_renda == 'Pensionista'], x='tempo_emprego', y='renda_log', ax=ax)
sns.regplot(data=renda[renda.tipo_renda == 'Servidor público'], x='tempo_emprego', y='renda_log', ax=ax)
sns.regplot(data=renda[renda.tipo_renda == 'Bolsista'], x='tempo_emprego', y='renda_log', ax=ax)

st.pyplot(fig)

# correlação com variável target original e transformada
st.text("Correlação com as variáveis resposta original e transformada")
st.dataframe(renda.corr(numeric_only=True).loc[:,['renda','renda_log']].T)

