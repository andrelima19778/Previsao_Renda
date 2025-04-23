# bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

st.title("Análise de Renda")
st.subheader('Use esse aplicativo para obter a predição da renda')

pipefinal = joblib.load("pipe_m16_v.final.joblib")

with st.form('user_inputs'):
    tempo = st.number_input('Tempo de emprego', 
                             min_value=0.12, max_value=40.0, value=1.0)
    sexo = st.radio("Selecione o sexo", ['M', 'F'])
    st.form_submit_button()

tempo=float(tempo)

estimativa = round(np.exp(pipefinal
                          .predict(pd.DataFrame([[sexo, tempo]], 
                                                columns=['sexo', 'tempo_emprego'])))[0], 2)
st.subheader(f':blue[A estimativa de Renda é: $ {estimativa:,.2f}]', divider=True)

unseen_data = st.file_uploader('Selecione sua base local CSV (arquivo padrão provido)')

@st.cache_data()
def load_file(unseen_data):
    time.sleep(3)
    if unseen_data is not None:
        df = pd.read_csv(unseen_data, usecols=['tempo_emprego', 'sexo', 'renda'])
    else:
        df = pd.read_csv('unseen_data3.csv', usecols=['tempo_emprego', 'sexo', 'renda']) # alterar para definitivo

    return(df)
    
renda = load_file(unseen_data)



novos_dados = pd.concat([renda, pd.DataFrame(np.exp(pipefinal.predict(renda)), columns=['renda_predita'])], axis=1)
novos_dados['renda'] = round(novos_dados['renda'], 2)
novos_dados['renda_predita'] = round(novos_dados['renda_predita'], 2)
novos_dados['renda'] = novos_dados['renda'].apply(lambda x: '$ {:,.2f}'.format(x))
novos_dados['renda_predita'] = novos_dados['renda_predita'].apply(lambda x: '$ {:,.2f}'.format(x))
novos_dados
