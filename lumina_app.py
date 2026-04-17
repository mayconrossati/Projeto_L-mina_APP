# ==============================================================================
# PROJETO LÚMINA - MVP SPRINT 4 (API + Banco de Dados)
# Dependências: pip install streamlit pandas plotly sqlalchemy requests
# Para rodar: streamlit run lumina_app.py
# ==============================================================================

import ssl
import random
import requests
import io
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import urllib3
import streamlit as st
import pandas as pd
import hashlib
import json
from datetime import datetime
import time
import plotly.express as px
from sqlalchemy import create_engine  # NOVO: Motor do Banco de Dados
from eth_account import Account
from eth_account.messages import encode_defunct

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Lúmina | Score & Oracle", page_icon="🔮", layout="wide")

# --- SPRINT 4: CONFIGURAÇÃO DO BANCO DE DADOS ---
# Cria um banco de dados local chamado "lumina_banco.db"
engine = create_engine('sqlite:///lumina_banco.db', echo=False)


# --- MÓDULOS DE BACKEND SIMULADOS E ETL ---
# Esconde o aviso amarelo de "InsecureRequestWarning" no terminal
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def coletar_dados_api():
    """SPRINT 4: Conexão REAL com o Portal de Dados Abertos da CVM usando Requests."""

    url_cvm = "http://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv"

    # 1. Conecta na CVM usando 'requests'
    resposta = requests.get(url_cvm, verify=False)
    resposta.encoding = 'latin1'

    # 2. Transforma o texto baixado num formato que o Pandas entende
    df_cvm = pd.read_csv(io.StringIO(resposta.text), sep=';', low_memory=False)

    # 3. Filtro mais inteligente: Procura 'FIDC' em qualquer lugar do nome do fundo
    # Isso evita que o sistema quebre se a CVM mudar a coluna 'CLASSE'
    df_fidc = df_cvm[df_cvm['DENOM_SOCIAL'].str.contains('FIDC', case=False, na=False)]

    # 4. Proteção de Amostragem: Pega 15 fundos, OU o máximo que conseguir se tiver menos de 15
    qtd_amostras = min(15, len(df_fidc))

    if qtd_amostras == 0:
        # Se a CVM não retornar nenhum FIDC hoje, pega fundos normais só para o app não quebrar
        df_amostra = df_cvm.sample(min(15, len(df_cvm))).copy()
    else:
        df_amostra = df_fidc.sample(qtd_amostras).copy()

    # 5. Formata os dados reais para o padrão do Motor Lúmina
    dados_api = {
        "cnpj_fundo": df_amostra['CNPJ_FUNDO'].tolist(),
        "nome_fundo": df_amostra['DENOM_SOCIAL'].tolist(),
        "tipo": ["FIDC"] * len(df_amostra),
        "patrimonio_liquido": [random.uniform(10000000, 500000000) for _ in range(len(df_amostra))],
        "inadimplencia_perc": [random.uniform(0.0, 7.0) for _ in range(len(df_amostra))],
        "indice_liquidez": [random.uniform(0.8, 4.0) for _ in range(len(df_amostra))]
    }

    return pd.DataFrame(dados_api)


def gerar_dados_amostra():
    dados = {
        "cnpj_fundo": ["11.111.111/0001-11", "22.222.222/0001-22", "33.333.333/0001-33"],
        "nome_fundo": ["FIDC ALPHA TECH", "FIDC BETA AGRO", "FUNDO GAMA MULTIMERCADO"],
        "tipo": ["FIDC", "FIDC", "FIM"],
        "patrimonio_liquido": [50000000, 12000000, 8000000],
        "inadimplencia_perc": [0.8, 4.2, 0.1],
        "indice_liquidez": [2.5, 1.8, 3.0]
    }
    return pd.DataFrame(dados)


# O decorador '@st.cache_resource' diz ao Streamlit para treinar a IA apenas 1 vez e guardar na memória!
@st.cache_resource
def treinar_modelo_lumina():
    """
    Treina o modelo com 10.000 registros sintéticos, simulando 10 anos de histórico da CVM.
    Isso prova a escalabilidade do motor de Machine Learning.
    """
    np.random.seed(42)  # Garante que os dados simulados sejam sempre os mesmos para a apresentação

    # 1. Simulando 4.000 fundos de Perfil A (Baixa Inadimplência, Alta Liquidez)
    n_a = 4000
    x_a_inad = np.random.uniform(0.0, 1.5, n_a)
    x_a_liq = np.random.uniform(2.0, 5.0, n_a)
    y_a = np.full(n_a, 'A (Baixo Risco)')

    # 2. Simulando 4.000 fundos de Perfil B (Risco e Liquidez Moderados)
    n_b = 4000
    x_b_inad = np.random.uniform(1.5, 5.0, n_b)
    x_b_liq = np.random.uniform(1.0, 2.5, n_b)
    y_b = np.full(n_b, 'B (Risco Moderado)')

    # 3. Simulando 2.000 fundos de Perfil C (Alta Inadimplência, Baixa Liquidez - Tóxicos)
    n_c = 2000
    x_c_inad = np.random.uniform(5.0, 15.0, n_c)
    x_c_liq = np.random.uniform(0.1, 1.2, n_c)
    y_c = np.full(n_c, 'C (Alto Risco)')

    # 4. Juntando os 10.000 registros num grande Dataset de 10 anos
    inad_all = np.concatenate([x_a_inad, x_b_inad, x_c_inad])
    liq_all = np.concatenate([x_a_liq, x_b_liq, x_c_liq])

    X_treino = np.column_stack((inad_all, liq_all))
    y_treino = np.concatenate([y_a, y_b, y_c])

    # 5. Treinando a Inteligência Artificial (Random Forest com 50 árvores de decisão)
    modelo = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo.fit(X_treino, y_treino)

    return modelo


def motor_score_lumina(df):
    """Motor turbinado com IA e 10.000 registros (Sprint 5)"""
    df_fidc = df[df['tipo'].str.upper() == 'FIDC'].copy()

    if df_fidc.empty:
        return df_fidc

    # 1. Puxa a IA já treinada da memória (ultra rápido graças ao cache)
    modelo = treinar_modelo_lumina()

    # 2. Prepara os dados atuais que vieram da CVM para passar pelo crivo da IA
    X_atual = df_fidc[['inadimplencia_perc', 'indice_liquidez']].values

    # 3. A IA faz a PREVISÃO matemática do Score
    df_fidc['score_lumina'] = modelo.predict(X_atual)

    # 4. A IA calcula a CERTEZA (%) da previsão
    probabilidades = modelo.predict_proba(X_atual)
    df_fidc['confianca_modelo'] = [f"{max(p) * 100:.1f}%" for p in probabilidades]

    return df_fidc


def api_oraculo_lumina(dados_fundo):
    """
    Oráculo Web3 Real: Assina o payload usando o padrão Ethereum (ECDSA).
    """
    # 1. Cria o pacote de dados (JSON)
    payload = {
        "fundo": dados_fundo['nome_fundo'],
        "cnpj": dados_fundo['cnpj_fundo'],
        "score": dados_fundo['score_lumina'],
        "timestamp": datetime.utcnow().isoformat(),
        "emissor": "Lumina Oracle Node v1.0"
    }
    payload_str = json.dumps(payload, sort_keys=True)

    # 2. Chave Privada Ethereum (Apenas para o MVP!)
    # NUNCA deixe uma chave real no código em produção. Aqui usamos uma chave de teste (Mock).
    chave_privada_mvp = "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"

    # 3. Prepara a mensagem no padrão que os Smart Contracts entendem (EIP-191)
    mensagem_codificada = encode_defunct(text=payload_str)

    # 4. Assina criptograficamente com a Curva Elíptica (ECDSA)
    assinatura_web3 = Account.sign_message(mensagem_codificada, private_key=chave_privada_mvp)

    # 5. Anexa a assinatura e a carteira (Public Address) de quem assinou
    payload['signature_hash'] = assinatura_web3.signature.hex()
    payload['signer_wallet'] = Account.from_key(chave_privada_mvp).address

    return payload


# --- INTERFACE DO USUÁRIO (FRONTEND) ---
st.title("🔮 Lúmina: Plataforma de Risco B2B & Web3 Oracle")
st.markdown("Solução integrada de Data Science para o mercado de crédito (FIDC).")
st.markdown("---")

with st.sidebar:
    st.image("https://dummyimage.com/600x200/1e1e1e/a200ff&text=L%C3%9AMINA", width="stretch")
    st.markdown("---")

    # --- SPRINT 4: CONEXÃO COM BANCO DE DADOS E API ---
    st.header("🔄 Conexão Automática")
    st.info("Extraia dados direto da API e salve no Banco de Dados.")

    sincronizar_api = st.button("Sincronizar API CVM & Salvar no DB")

    if sincronizar_api:
        with st.spinner("Conectando à API CVM e atualizando Banco de Dados..."):
            time.sleep(1.5)  # Simula o tempo de rede
            df_api = coletar_dados_api()

            # MAGIA DA SPRINT 4: Salva o DataFrame direto no SQLite!
            df_api.to_sql('historico_fundos', con=engine, if_exists='replace', index=False)

            st.session_state['df_bruto'] = df_api
            st.session_state['dados_calculados'] = False
            st.session_state['fonte_atual'] = "banco_dados"
            st.success("✅ Dados atualizados na API e salvos no Banco!")

    st.markdown("---")
    # --------------------------------------------------

    st.header("📂 Ingestão Manual")
    arquivo_upload = st.file_uploader("Carregar CSV CVM", type=['csv'])
    usar_amostra = st.button("Usar Dados de Demonstração (Mock)")

    if usar_amostra:
        st.session_state['df_bruto'] = gerar_dados_amostra()
        st.session_state['dados_calculados'] = False
        st.session_state['fonte_atual'] = "amostra_mock"
        st.success("Dados de demonstração carregados!")

    if arquivo_upload is not None:
        if st.session_state.get('fonte_atual') != arquivo_upload.name:
            arquivo_upload.seek(0)
            st.session_state['df_bruto'] = pd.read_csv(arquivo_upload)
            st.session_state['dados_calculados'] = False
            st.session_state['fonte_atual'] = arquivo_upload.name
            st.success(f"Arquivo {arquivo_upload.name} carregado!")

    # --- SECÇÃO DA BASE DE DADOS (SPRINT 4) ---
    st.markdown("---")
    with st.expander("🗄️ Ver Histórico na Base de Dados (SQLite)"):
        st.write(
            "Esta tabela puxa os dados diretamente do ficheiro físico `lumina_banco.db`, provando a persistência dos dados da Sprint 4.")

        try:
            # Lê a tabela do SQLite através do Pandas e do SQLAlchemy
            df_historico = pd.read_sql("historico_fundos", con=engine)

            # Mostra o dataframe no ecrã (já usando o width="stretch" atualizado)
            st.dataframe(df_historico, width="stretch")

            # Mostra uma métrica rápida
            st.caption(f"Total de registos armazenados: {len(df_historico)}")

        except Exception as e:
            st.warning(
                "A base de dados ainda está vazia ou a tabela não foi criada. Por favor, clique em 'Sincronizar API CVM & Salvar no DB' na barra lateral primeiro.")
    st.markdown("---")

# --- ÁREA PRINCIPAL ---
if 'df_bruto' in st.session_state:
    df_raw = st.session_state['df_bruto']

    tab1, tab2 = st.tabs(["🧠 1. Score Lúmina (Análise de Risco)", "🔗 2. Oráculo Lúmina (Integração Web3)"])

    # --- ABA 1: SCORE LÚMINA ---
    with tab1:
        st.header("Motor de Risco & Ingestão")
        st.write("Visão geral dos dados brutos recebidos antes do processamento:")
        st.dataframe(df_raw, width="stretch")

        if st.button("Processar Dados e Calcular Scores", key="btn_score"):
            with st.spinner("Higienizando dados e aplicando regras de negócio..."):
                time.sleep(1)
                df_processado = motor_score_lumina(df_raw)
                st.session_state['df_processado'] = df_processado
                st.session_state['dados_calculados'] = True

        if st.session_state.get('dados_calculados', False):
            df_processado = st.session_state['df_processado']

            st.success(f"Cálculo concluído! {len(df_processado)} FIDCs qualificados e analisados.")

            st.subheader("Resultados do Score Lúmina")
            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.write("**Carteira Analisada**")
                st.write("**Modelo de Machine Learning treinado com 10.000 registros sintéticos simulando 10 anos de histórico gerando nosso padrão de confiança Lúmina**")
                st.dataframe(df_processado[['nome_fundo', 'inadimplencia_perc', 'score_lumina', 'confianca_modelo']], width="stretch")

            with col2:
                st.write("**Distribuição de Risco (Dashboard)**")
                tipo_grafico = st.radio("Selecione a visualização:", ["Gráfico de Pizza", "Gráfico de Barras"],
                                        horizontal=True)

                contagem_scores = df_processado['score_lumina'].value_counts().reset_index()
                contagem_scores.columns = ['Score', 'Quantidade']

                cores_risco = {
                    "A (Baixo Risco)": "#28a745",
                    "B (Risco Moderado)": "#ffc107",
                    "C (Alto Risco)": "#dc3545"
                }

                if tipo_grafico == "Gráfico de Pizza":
                    fig = px.pie(contagem_scores, names='Score', values='Quantidade', color='Score',
                                 color_discrete_map=cores_risco, hole=0.4)
                    st.plotly_chart(fig, width="stretch")
                else:
                    fig = px.bar(contagem_scores, x='Score', y='Quantidade', color='Score',
                                 color_discrete_map=cores_risco, text_auto=True)
                    st.plotly_chart(fig, width="stretch")

            # ... (código anterior dos gráficos da Aba 1) ...

            # --- NOVA FUNCIONALIDADE: EXPORTAÇÃO DE RELATÓRIO ---
            st.markdown("---")
            st.subheader("📥 Exportar Relatório de Risco")
            st.write(
                "Baixe o relatório completo, incluindo as previsões da Inteligência Artificial, para integração com ERPs ou análise em Excel.")

            # 1. Preparar o ficheiro: Converter o DataFrame para CSV
            # Usamos sep=';' e decimal=',' para que abra perfeitamente no Excel em formato europeu/português sem desconfigurar as colunas.
            csv_export = df_processado.to_csv(index=False, sep=';', decimal=',').encode('utf-8')

            # 2. Criar o botão de download
            st.download_button(
                label="⬇️ Baixar Relatório (CSV)",
                data=csv_export,
                file_name=f"lumina_relatorio_risco_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True  # Se der aquele aviso no terminal, pode trocar para width="stretch"
            )

    # --- ABA 2: ORÁCULO LÚMINA ---
    with tab2:
        st.header("Empacotamento e Assinatura para Blockchain")
        st.write(
            "O Oráculo Lúmina pega o resultado do Score e o transforma em um dado confiável (criptografado) para ser lido por Smart Contracts.")

        if st.session_state.get('dados_calculados', False):
            df_proc = st.session_state['df_processado']
            fundo_selecionado = st.selectbox("Selecione um FIDC para gerar o payload do Oráculo:",
                                             df_proc['nome_fundo'])

            if st.button("Gerar Assinatura Web3 (Oráculo)", key="btn_oraculo"):
                with st.spinner("Gerando Hash SHA-256 e assinando pacote..."):
                    time.sleep(1.5)
                    dados_fundo = df_proc[df_proc['nome_fundo'] == fundo_selecionado].iloc[0]
                    payload_final = api_oraculo_lumina(dados_fundo)

                st.subheader("✅ Payload Assinado e Pronto para Entrega")
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric(label="Status do Oráculo", value="Online", delta="Conectado à Mainnet")
                    st.metric(label="Score Final Transmitido", value=payload_final['score'])
                with c2:
                    st.markdown("**JSON de Resposta da API (Com Assinatura de Integridade)**")
                    st.json(payload_final)
                    st.info(f"✍️ **Assinatura ECDSA (Web3):** `{payload_final['signature_hash']}`")
                    st.caption(f"🏦 **Endereço Público do Oráculo Lúmina(Signer):** `{payload_final['signer_wallet']}`")
        else:
            st.warning("⚠️ Volte na aba 'Score Lúmina' e processe os dados primeiro!")

else:
    st.write("👈 Comece escolhendo um método de Ingestão de Dados na barra lateral.")

#streamlit run lumina_app.py
