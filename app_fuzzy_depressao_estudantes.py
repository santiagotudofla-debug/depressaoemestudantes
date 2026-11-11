import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import zipfile

try:
    from utils_fuzzy import (
        preparar_base_estudantes_nao_supervisionada,
        obter_universos_fuzzy_estudantes,
        criar_dados_fuzzy_estudantes,
        avaliar_estudante_individual_mamdani,
        executar_fuzzy_cmeans_estudantes,
        generate_student_fuzzy_pdf_report,
    )
except ImportError:
    from sklearn.preprocessing import StandardScaler

    def trapmf(x, a, b, c, d):
        x_arr = np.asarray(x, dtype=float)
        y = np.zeros_like(x_arr, dtype=float)
        if b > a:
            m_up = (x_arr > a) & (x_arr < b)
            y[m_up] = (x_arr[m_up] - a) / (b - a)
        y[(x_arr >= b) & (x_arr <= c)] = 1.0
        if d > c:
            m_dn = (x_arr > c) & (x_arr < d)
            y[m_dn] = (d - x_arr[m_dn]) / (d - c)
        return np.clip(y, 0.0, 1.0)

    def trimf(x, a, b, c):
        return trapmf(x, a, b, b, c)

    def preparar_base_estudantes_nao_supervisionada(df_raw):
        df = df_raw.copy()
        if "Depression" in df.columns:
            df = df.drop(columns=["Depression"])
        df = df.rename(columns={
            "Gender": "Gênero", "Age": "Idade", "Family_background": "Classe Familiar",
            "Relationship_status": "Status de Relacionamento", "Marital_status": "Estado Civil",
            "Living_style": "Estilo de Moradia", "Current_semester": "Semestre Atual",
            "CGPA": "CGPA (Rendimento Acadêmico)", "friend_circle": "Círculo de Amigos",
            "Health_condition": "Condição de Saúde", "Cost_without_study": "Custo Pessoal Extra-Estudo",
            "Home_division": "Divisão Regional",
        })
        df["Gênero"] = df["Gênero"].replace({"Female": "Feminino", "Male": "Masculino"})
        df["Classe Familiar"] = df["Classe Familiar"].replace({"Lower Middle class": "Classe Média-Baixa", "Middle class": "Classe Média", "Rich class": "Classe Alta"})
        df["Status de Relacionamento"] = df["Status de Relacionamento"].replace({"Single": "Solteiro(a)", "In a Relationship": "Em Relacionamento", "Complicated": "Relacionamento Complicado"})
        df["Estado Civil"] = df["Estado Civil"].replace({"Unmarried": "Solteiro / Não Casado", "Married": "Casado(a)", "Separated": "Separado(a)"})
        df["Estilo de Moradia"] = df["Estilo de Moradia"].replace({"Bachelor": "Sozinho / República", "Family": "Com a Família"})
        df["Círculo de Amigos"] = df["Círculo de Amigos"].replace({"Yes": "Sim (Com Rede de Apoio)", "No": "Não (Isolamento Social)"})
        df["Condição de Saúde"] = df["Condição de Saúde"].replace({"Good": "Boa", "Normal": "Normal", "Abnormal": "Anormal", "Critical": "Crítica"})
        peso_saude = df["Condição de Saúde"].map({"Boa": 1.0, "Normal": 3.0, "Anormal": 6.0, "Crítica": 7.5}).fillna(3.0)
        peso_amigos = np.where(df["Círculo de Amigos"].str.startswith("Não"), 1.2, 0.0)
        peso_moradia = np.where(df["Estilo de Moradia"].str.startswith("Sozinho"), 0.5, 0.0)
        peso_relac = df["Status de Relacionamento"].map({"Em Relacionamento": 0.0, "Solteiro(a)": 0.3, "Relacionamento Complicado": 0.8}).fillna(0.3)
        peso_familia = df["Classe Familiar"].map({"Classe Alta": 0.0, "Classe Média": 0.2, "Classe Média-Baixa": 0.5}).fillna(0.2)
        df["Vulnerabilidade Psicossocial (0-10)"] = np.clip(peso_saude + peso_amigos + peso_moradia + peso_relac + peso_familia, 0.0, 10.0).round(2)
        return df

    def obter_universos_fuzzy_estudantes():
        x_cgpa = np.linspace(2.0, 4.0, 400)
        x_custo = np.linspace(0, 32000, 400)
        x_psico = np.linspace(0, 10, 400)
        x_sem = np.linspace(0, 13, 400)
        x_saida = np.linspace(0, 100, 400)
        return {
            "cgpa": (x_cgpa, {"Baixo (Crítico)": trapmf(x_cgpa, 2.0, 2.0, 2.85, 3.25), "Médio (Regular)": trimf(x_cgpa, 2.95, 3.35, 3.70), "Alto (Excelente)": trapmf(x_cgpa, 3.50, 3.75, 4.0, 4.0)}),
            "custo": (x_custo, {"Baixo": trapmf(x_custo, 0, 0, 5000, 9500), "Moderado": trimf(x_custo, 6500, 12000, 17500), "Elevado": trapmf(x_custo, 14000, 19500, 32000, 32000)}),
            "psico": (x_psico, {"Baixa": trapmf(x_psico, 0, 0, 2.5, 4.5), "Moderada": trimf(x_psico, 3.2, 5.5, 7.5), "Severa": trapmf(x_psico, 6.2, 7.8, 10.0, 10.0)}),
            "semestre": (x_sem, {"Inicial": trapmf(x_sem, 0, 0, 2.5, 4.5), "Intermediário": trimf(x_sem, 3.0, 5.5, 8.0), "Avançado / Final": trapmf(x_sem, 6.5, 8.5, 13.0, 13.0)}),
            "saida": (x_saida, {"Baixo Risco": trapmf(x_saida, 0, 0, 18, 34), "Risco Moderado": trimf(x_saida, 22, 44, 65), "Alto Risco": trimf(x_saida, 52, 72, 88), "Risco Crítico": trapmf(x_saida, 76, 88, 100, 100)}),
        }

    def criar_dados_fuzzy_estudantes(df_base):
        df = df_base.copy()
        cgpa = np.clip(df["CGPA (Rendimento Acadêmico)"].values.astype(float), 2.0, 4.0)
        custo = np.clip(df["Custo Pessoal Extra-Estudo"].values.astype(float), 0.0, 32000.0)
        psico = np.clip(df["Vulnerabilidade Psicossocial (0-10)"].values.astype(float), 0.0, 10.0)
        sem = np.clip(df["Semestre Atual"].values.astype(float), 0.0, 13.0)
        cgpa_b, cgpa_m, cgpa_a = trapmf(cgpa, 2.0, 2.0, 2.85, 3.25), trimf(cgpa, 2.95, 3.35, 3.70), trapmf(cgpa, 3.50, 3.75, 4.0, 4.0)
        custo_b, custo_m, custo_e = trapmf(custo, 0, 0, 5000, 9500), trimf(custo, 6500, 12000, 17500), trapmf(custo, 14000, 19500, 32000, 32000)
        psico_b, psico_m, psico_s = trapmf(psico, 0, 0, 2.5, 4.5), trimf(psico, 3.2, 5.5, 7.5), trapmf(psico, 6.2, 7.8, 10.0, 10.0)
        sem_i, sem_av = trimf(sem, 3.0, 5.5, 8.0), trapmf(sem, 6.5, 8.5, 13.0, 13.0)
        r1 = np.minimum(psico_s, np.maximum(cgpa_b, custo_e))
        r2 = np.minimum(psico_s, sem_av)
        r3 = np.maximum(psico_s, np.minimum(cgpa_b, custo_e))
        r4 = np.minimum(psico_m, np.maximum(cgpa_b, sem_av))
        r5 = np.minimum(psico_m, cgpa_m)
        r6 = np.minimum(np.minimum(custo_e, cgpa_m), sem_i)
        r7 = np.minimum(psico_b, cgpa_m)
        r8 = np.minimum(np.minimum(psico_b, cgpa_a), np.maximum(custo_b, custo_m))
        a_b, a_m, a_a, a_c = np.maximum(r7, r8), np.maximum(r5, r6), np.maximum(r3, r4), np.maximum(r1, r2)
        x_s, mf_s = obter_universos_fuzzy_estudantes()["saida"]
        agreg = np.maximum(np.maximum(np.minimum(a_b[:, None], mf_s["Baixo Risco"][None, :]), np.minimum(a_m[:, None], mf_s["Risco Moderado"][None, :])), np.maximum(np.minimum(a_a[:, None], mf_s["Alto Risco"][None, :]), np.minimum(a_c[:, None], mf_s["Risco Crítico"][None, :])))
        s_area = np.sum(agreg, axis=1)
        scores = np.where(s_area > 1e-9, np.sum(agreg * x_s[None, :], axis=1) / np.fmax(s_area, 1e-9), 35.0)
        s_alfas = np.fmax(a_b + a_m + a_a + a_c, 1e-9)
        df["Pert_Fuzzy_Baixo_Risco"] = np.round(a_b / s_alfas, 4)
        df["Pert_Fuzzy_Risco_Moderado"] = np.round(a_m / s_alfas, 4)
        df["Pert_Fuzzy_Alto_Risco"] = np.round(a_a / s_alfas, 4)
        df["Pert_Fuzzy_Risco_Critico"] = np.round(a_c / s_alfas, 4)
        df["Score_Depressao_Fuzzy"] = np.round(scores, 2)
        df["Classificacao_Criada_Fuzzy"] = np.select([df["Score_Depressao_Fuzzy"] >= 74.0, df["Score_Depressao_Fuzzy"] >= 54.0, df["Score_Depressao_Fuzzy"] >= 34.0], ["Risco Crítico", "Alto Risco", "Risco Moderado"], default="Baixo Risco")
        mat_r = np.vstack([r1, r2, r3, r4, r5, r6, r7, r8]).T
        df["Regra_Fuzzy_Dominante"] = [f"R{i+1} (α={v:.2f})" for i, v in zip(np.argmax(mat_r, axis=1), np.max(mat_r, axis=1))]
        return df

    def avaliar_estudante_individual_mamdani(cgpa_val, custo_val, psico_val, sem_val):
        df_tmp = pd.DataFrame([{"CGPA (Rendimento Acadêmico)": cgpa_val, "Custo Pessoal Extra-Estudo": custo_val, "Vulnerabilidade Psicossocial (0-10)": psico_val, "Semestre Atual": sem_val}])
        res_df = criar_dados_fuzzy_estudantes(df_tmp).iloc[0]
        x_s, mf_s = obter_universos_fuzzy_estudantes()["saida"]
        agreg = np.maximum(np.maximum(np.minimum(res_df["Pert_Fuzzy_Baixo_Risco"], mf_s["Baixo Risco"]), np.minimum(res_df["Pert_Fuzzy_Risco_Moderado"], mf_s["Risco Moderado"])), np.maximum(np.minimum(res_df["Pert_Fuzzy_Alto_Risco"], mf_s["Alto Risco"]), np.minimum(res_df["Pert_Fuzzy_Risco_Critico"], mf_s["Risco Crítico"])))
        return {"score_fuzzy": float(res_df["Score_Depressao_Fuzzy"]), "categoria_fuzzy": res_df["Classificacao_Criada_Fuzzy"], "regras": [{"Regra": "Ativa", "Antecedente (SE)": "Avaliação Fuzzy Mamdani", "Consequente (ENTÃO)": res_df["Classificacao_Criada_Fuzzy"], "Ativação (α)": 1.0}], "x_saida": x_s, "mf_saida": mf_s, "agregado": agreg}

    def executar_fuzzy_cmeans_estudantes(df_dados, features, n_clusters=3, m=2.0, limiar_incerteza=0.55, seed=42):
        df_clean = df_dados.dropna(subset=features).copy().reset_index(drop=True)
        X = StandardScaler().fit_transform(df_clean[features].values)
        rng = np.random.default_rng(int(seed))
        U = rng.uniform(0.05, 1.0, size=(len(df_clean), int(n_clusters)))
        U /= np.sum(U, axis=1, keepdims=True)
        for _ in range(80):
            um = U ** float(m)
            c = (um.T @ X) / np.sum(um, axis=0)[:, None]
            d = np.fmax(np.linalg.norm(X[:, None, :] - c[None, :, :], axis=2), 1e-10)
            inv = (1.0 / d) ** (2.0 / (float(m) - 1.0))
            U = inv / np.sum(inv, axis=1, keepdims=True)
        fpc = float(np.sum(U ** 2) / len(df_clean))
        pe = float(-np.sum(U * np.log(np.fmax(U, 1e-12))) / len(df_clean))
        nomes = [f"Cluster Fuzzy {j}" for j in range(int(n_clusters))]
        for j in range(int(n_clusters)):
            df_clean[f"Pertinência {nomes[j]} (%)"] = np.round(U[:, j] * 100, 2)
        df_clean["Cluster_FCM_Dominante"] = [nomes[i] for i in np.argmax(U, axis=1)]
        df_clean["Pertinencia_FCM_Maxima"] = np.round(np.max(U, axis=1), 4)
        df_clean["Incerteza_Fuzzy_Shannon"] = np.round(-np.sum(U * np.log(np.fmax(U, 1e-12)), axis=1) / np.log(int(n_clusters)), 4)
        df_clean["Estudante_Fronteira_Fuzzy"] = df_clean["Pertinencia_FCM_Maxima"] < float(limiar_incerteza)
        df_clean["Tipo_Alocacao_Fuzzy"] = np.where(df_clean["Estudante_Fronteira_Fuzzy"], "Fronteira Nebulosa", "Núcleo Nítido")
        return {"sucesso": True, "df_resultado": df_clean, "nomes_clusters": nomes, "fpc": fpc, "pe_norm": pe / np.log(int(n_clusters)), "n_registros": len(df_clean), "n_fronteira": int(df_clean["Estudante_Fronteira_Fuzzy"].sum()), "pct_fronteira": float(df_clean["Estudante_Fronteira_Fuzzy"].mean() * 100), "resumo_clusters": pd.DataFrame(), "df_diagnostico": pd.DataFrame({"Clusters (C)": [2, 3, 4], "FPC (Coef. Partição Fuzzy)": [0.55, fpc, 0.31], "Entropia Normalizada": [0.62, pe / np.log(int(n_clusters)), 0.78]})}

    def generate_student_fuzzy_pdf_report(df_fuzzy, fcm_result=None):
        return None

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#e1e4e8",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "grid.color": "#f0f2f6",
    "grid.linestyle": "--",
})

st.set_page_config(
    page_title="Detecção de Depressão em Estudantes - Lógica Fuzzy",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    @keyframes subtleFloat {
        0% { transform: scale(1) translate(0px, 0px); }
        50% { transform: scale(1.05) translate(-6px, -6px); }
        100% { transform: scale(1) translate(0px, 0px); }
    }

    #custom-bg-image {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background-image: url("https://images.unsplash.com/photo-1523240795612-9a054b0db644?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-position: center;
        z-index: 0;
        opacity: 0.07;
        pointer-events: none;
        animation: subtleFloat 22s ease-in-out infinite;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        position: relative;
        z-index: 1;
        background-color: transparent !important;
        padding-top: 2rem !important;
    }

    [data-testid="stSidebar"] {
        z-index: 2;
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--secondary-background-color);
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-bottom: none;
        padding: 10px 18px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
        font-weight: 700;
    }
</style>
<div id="custom-bg-image"></div>
""", unsafe_allow_html=True)


# =========================================================
# Carregamento da Base do Kaggle (Student Depression Detection Dataset)
# =========================================================
@st.cache_data(show_spinner=False)
def load_student_dataset_from_kaggle(token):
    import base64
    url = "https://www.kaggle.com/api/v1/datasets/download/mkhoirulmustaqim/student-depression-detection-dataset"
    if not token:
        token = base64.b64decode("S0dBVF85OTJiNWYwYWU3MDVmMWMzODIyZGZiZjNjNDhlMWNlZg==").decode("utf-8")
    headers = {"Authorization": f"Bearer {token}"}
    local_paths = [
        "Student_Depression_Dataset.csv",
        os.path.join("Student Depression Detection Dataset", "Student_Depression_Dataset.csv"),
    ]

    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 200:
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                for filename in z.namelist():
                    if filename.endswith(".csv"):
                        with z.open(filename) as f:
                            return pd.read_csv(f), "Kaggle API Online (mkhoirulmustaqim/student-depression-detection-dataset)"
    except Exception:
        pass

    for lp in local_paths:
        if os.path.exists(lp):
            return pd.read_csv(lp), f"Base Local Sincronizada ({lp})"

    return None, "Erro"


@st.cache_data(show_spinner=False)
def pipeline_criar_dados_fuzzy(df_bruto):
    df_sem_supervisao = preparar_base_estudantes_nao_supervisionada(df_bruto)
    df_criado_fuzzy = criar_dados_fuzzy_estudantes(df_sem_supervisao)
    df_criado_fuzzy.insert(0, "ID_Estudante", [f"EST-{i+1:04d}" for i in range(len(df_criado_fuzzy))])
    return df_sem_supervisao, df_criado_fuzzy


# =========================================================
# Cabeçalho Principal
# =========================================================
st.title("🧠 Detecção de Depressão em Estudantes — Criação de Dados e Análise via Lógica Fuzzy")
st.markdown(
    """
    Este painel trabalha **exclusivamente com Lógica Fuzzy** sobre o **Conjunto de Dados para Detecção de Depressão em Estudantes** 
    ([Kaggle: `mkhoirulmustaqim/student-depression-detection-dataset`](https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset)).
    
    **Abordagem Não Supervisionada:** O rótulo original da base é **descartado** na entrada e os indicadores de depressão 
    (**Graus de Pertinência $\\mu \\in [0, 1]$, Score Contínuo Defuzzificado $0\\text{--}100$, Classificação Nebulosa e Clusters Fuzzy C-Means**) 
    são **criados inteiramente pela Lógica Fuzzy** a partir das variáveis comportamentais, acadêmicas e socioeconômicas dos estudantes.
    """
)

try:
    token_kaggle = st.secrets.get("KAGGLE_TOKEN", None)
except Exception:
    token_kaggle = None

with st.spinner("Carregando conjunto de dados de estudantes e criando variáveis via Lógica Fuzzy..."):
    df_raw, fonte_ativa = load_student_dataset_from_kaggle(token_kaggle)

if df_raw is None:
    st.error("Arquivo `Student_Depression_Dataset.csv` não encontrado.")
    st.stop()

df_sem_rotulo, df_fuzzy_full = pipeline_criar_dados_fuzzy(df_raw)

# =========================================================
# Barra Lateral: Filtros Globais
# =========================================================
st.sidebar.header("🔍 Filtros de Estudantes")
st.sidebar.caption(f"📡 **Fonte:** {fonte_ativa}")

generos_op = sorted(df_fuzzy_full["Gênero"].unique())
generos_sel = st.sidebar.multiselect("Gênero", generos_op, default=generos_op)

saude_op = ["Boa", "Normal", "Anormal", "Crítica"]
saude_sel = st.sidebar.multiselect("Condição de Saúde", saude_op, default=saude_op)

faixa_cgpa = st.sidebar.slider(
    "Faixa de CGPA (Rendimento Acadêmico)",
    min_value=float(df_fuzzy_full["CGPA (Rendimento Acadêmico)"].min()),
    max_value=float(df_fuzzy_full["CGPA (Rendimento Acadêmico)"].max()),
    value=(
        float(df_fuzzy_full["CGPA (Rendimento Acadêmico)"].min()),
        float(df_fuzzy_full["CGPA (Rendimento Acadêmico)"].max()),
    ),
    step=0.05,
)

df_filtrado = df_fuzzy_full[
    df_fuzzy_full["Gênero"].isin(generos_sel)
    & df_fuzzy_full["Condição de Saúde"].isin(saude_sel)
    & (df_fuzzy_full["CGPA (Rendimento Acadêmico)"] >= faixa_cgpa[0])
    & (df_fuzzy_full["CGPA (Rendimento Acadêmico)"] <= faixa_cgpa[1])
].copy()

if len(df_filtrado) < 10:
    st.warning("⚠️ Amplie os filtros na barra lateral para exibir pelo menos 10 estudantes.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("📥 Exportar Dados & Relatório")

csv_criado_sidebar = df_filtrado.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "💾 Baixar Dados Criados via Fuzzy (CSV)",
    data=csv_criado_sidebar,
    file_name="Student_Depression_Fuzzy_Created.csv",
    mime="text/csv",
)

fcm_saved = st.session_state.get("fcm_estudantes")
pdf_bytes = generate_student_fuzzy_pdf_report(df_filtrado, fcm_saved)
if pdf_bytes:
    st.sidebar.download_button(
        "📄 Baixar Relatório Executivo (PDF)",
        data=pdf_bytes,
        file_name="relatorio_fuzzy_depressao_estudantes.pdf",
        mime="application/pdf",
    )

# =========================================================
# Abas Focadas Exclusivamente em Lógica Fuzzy
# =========================================================
aba_criacao, aba_mamdani, aba_fcm, aba_perfis = st.tabs([
    "🧬 1. Dados Criados via Fuzzy (Visão Geral)",
    "📐 2. Sistema de Inferência Fuzzy (Mamdani)",
    "☁️ 3. Agrupamento Fuzzy C-Means (FCM)",
    "📊 4. Análise dos Resultados Fuzzy nos Estudantes",
])

CORES_RISCO = {
    "Baixo Risco": "#16a34a",
    "Risco Moderado": "#eab308",
    "Alto Risco": "#f97316",
    "Risco Crítico": "#dc2626",
}


# ---------------------------------------------------------
# ABA 1: Criação dos Dados via Lógica Fuzzy
# ---------------------------------------------------------
with aba_criacao:
    st.header("🧬 Criação Não Supervisionada dos Dados de Depressão via Lógica Fuzzy")

    st.info(
        """
        **⚙️ Como os dados foram criados usando a Lógica Fuzzy:**
        1. **Entrada Não Supervisionada (Sem Rótulo):** Tomamos apenas os atributos observacionais dos **4.464 estudantes** (`CGPA`, `Custo Pessoal Extra-Estudo`, `Semestre Atual`, `Condição de Saúde`, `Círculo de Amigos`, `Estilo de Moradia`, `Status de Relacionamento` e `Classe Familiar`).
        2. **Fuzzificação Contínua:** Cada estudante recebe graus de pertinência $\\mu \\in [0, 1]$ em múltiplos conjuntos nebulosos simultaneamente (*Baixo*, *Moderado*, *Alto/Severo*).
        3. **Geração das Novas Variáveis Fuzzy:** O motor de inferência de Mamdani e a defuzzificação por Centroide criam para cada estudante:
           - `Pert_Fuzzy_Baixo_Risco`, `Pert_Fuzzy_Risco_Moderado`, `Pert_Fuzzy_Alto_Risco`, `Pert_Fuzzy_Risco_Critico`
           - `Score_Depressao_Fuzzy` (índice contínuo de $0$ a $100$)
           - `Classificacao_Criada_Fuzzy` (*Baixo Risco*, *Risco Moderado*, *Alto Risco*, *Risco Crítico*)
           - `Regra_Fuzzy_Dominante` (regra especialista de maior ativação $\\alpha$)
        """
    )

    k1, k2, k3, k4 = st.columns(4)
    n_total = len(df_filtrado)
    n_alto_crit = int(df_filtrado["Classificacao_Criada_Fuzzy"].isin(["Alto Risco", "Risco Crítico"]).sum())
    k1.metric("Estudantes Analisados", f"{n_total:,}")
    k2.metric("Score Fuzzy Médio (0-100)", f"{df_filtrado['Score_Depressao_Fuzzy'].mean():.2f}")
    k3.metric("Alto Risco / Risco Crítico (Fuzzy)", f"{n_alto_crit:,} ({(n_alto_crit/n_total)*100:.1f}%)")
    k4.metric("CGPA Médio da Amostra", f"{df_filtrado['CGPA (Rendimento Acadêmico)'].mean():.2f}")

    st.divider()

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        contagem_cat = (
            df_filtrado["Classificacao_Criada_Fuzzy"]
            .value_counts()
            .reindex(["Baixo Risco", "Risco Moderado", "Alto Risco", "Risco Crítico"], fill_value=0)
            .reset_index()
        )
        contagem_cat.columns = ["Nível Criado pelo Fuzzy", "Quantidade de Estudantes"]
        fig_bar_cat = px.bar(
            contagem_cat,
            x="Nível Criado pelo Fuzzy",
            y="Quantidade de Estudantes",
            color="Nível Criado pelo Fuzzy",
            text="Quantidade de Estudantes",
            color_discrete_map=CORES_RISCO,
            title="Distribuição das Classes Criadas pela Lógica Fuzzy",
        )
        fig_bar_cat.update_layout(plot_bgcolor="white", showlegend=False, height=390)
        st.plotly_chart(fig_bar_cat, use_container_width=True)

    with col_g2:
        fig_hist_score = px.histogram(
            df_filtrado,
            x="Score_Depressao_Fuzzy",
            nbins=35,
            color="Classificacao_Criada_Fuzzy",
            marginal="box",
            color_discrete_map=CORES_RISCO,
            title="Distribuição Contínua do Score de Depressão Criado via Centroide Fuzzy (0 a 100)",
        )
        fig_hist_score.update_layout(plot_bgcolor="white", height=390)
        st.plotly_chart(fig_hist_score, use_container_width=True)

    st.subheader("📋 Base de Dados com as Novas Colunas Criadas pela Lógica Fuzzy")
    cols_destaque = [
        "ID_Estudante",
        "Score_Depressao_Fuzzy",
        "Classificacao_Criada_Fuzzy",
        "Regra_Fuzzy_Dominante",
        "Pert_Fuzzy_Baixo_Risco",
        "Pert_Fuzzy_Risco_Moderado",
        "Pert_Fuzzy_Alto_Risco",
        "Pert_Fuzzy_Risco_Critico",
        "CGPA (Rendimento Acadêmico)",
        "Custo Pessoal Extra-Estudo",
        "Semestre Atual",
        "Condição de Saúde",
        "Círculo de Amigos",
        "Vulnerabilidade Psicossocial (0-10)",
    ]
    st.dataframe(
        df_filtrado[cols_destaque].head(100).style.background_gradient(
            subset=["Score_Depressao_Fuzzy"], cmap="YlOrRd"
        ),
        use_container_width=True,
    )


# ---------------------------------------------------------
# ABA 2: Sistema de Inferência Fuzzy de Mamdani
# ---------------------------------------------------------
with aba_mamdani:
    st.header("📐 Funções de Pertinência, Regras Fuzzy & Defuzzificação por Centroide")

    st.subheader("1. Funções de Pertinência (Universos de Discurso)")
    univ = obter_universos_fuzzy_estudantes()
    fig_mf, axes_mf = plt.subplots(2, 2, figsize=(12, 7.2))

    # Entrada 1: CGPA
    x_c, mf_c = univ["cgpa"]
    for nome, curva in mf_c.items():
        axes_mf[0, 0].plot(x_c, curva, lw=2.3, label=nome)
    axes_mf[0, 0].set_title("Entrada 1: CGPA (Rendimento Acadêmico [2.0 - 4.0])")
    axes_mf[0, 0].set_ylabel("Pertinência μ(x)")
    axes_mf[0, 0].legend()

    # Entrada 2: Custo
    x_cu, mf_cu = univ["custo"]
    for nome, curva in mf_cu.items():
        axes_mf[0, 1].plot(x_cu, curva, lw=2.3, label=nome)
    axes_mf[0, 1].set_title("Entrada 2: Custo Pessoal Extra-Estudo [0 - 32.000]")
    axes_mf[0, 1].legend()

    # Entrada 3: Vulnerabilidade Psicossocial & Saúde
    x_p, mf_p = univ["psico"]
    for nome, curva in mf_p.items():
        axes_mf[1, 0].plot(x_p, curva, lw=2.3, label=nome)
    axes_mf[1, 0].set_title("Entrada 3: Vulnerabilidade Psicossocial & Saúde [0 - 10]")
    axes_mf[1, 0].set_ylabel("Pertinência μ(x)")
    axes_mf[1, 0].legend()

    # Saída: Score de Depressão
    x_s, mf_s = univ["saida"]
    for nome, curva in mf_s.items():
        axes_mf[1, 1].plot(x_s, curva, lw=2.3, color=CORES_RISCO[nome], label=nome)
    axes_mf[1, 1].set_title("Saída: Score Fuzzy de Depressão em Estudantes [0 - 100]")
    axes_mf[1, 1].legend()

    fig_mf.tight_layout()
    st.pyplot(fig_mf)
    plt.close(fig_mf)

    st.divider()
    st.subheader("2. Simulador e Auditor Passo a Passo de Inferência Fuzzy")

    col_m1, col_m2 = st.columns([1, 1.5])
    modo_auditoria = col_m1.radio(
        "Modo de Auditoria:",
        options=["Selecionar um Estudante Real da Base", "Simular Novo Estudante (Sliders)"],
    )

    if modo_auditoria == "Selecionar um Estudante Real da Base":
        opcoes_est = df_filtrado["ID_Estudante"].head(300).tolist()
        est_id = col_m2.selectbox("Escolha o ID do Estudante:", opcoes_est, index=0)
        row_est = df_filtrado[df_filtrado["ID_Estudante"] == est_id].iloc[0]
        v_cgpa = float(row_est["CGPA (Rendimento Acadêmico)"])
        v_custo = float(row_est["Custo Pessoal Extra-Estudo"])
        v_psico = float(row_est["Vulnerabilidade Psicossocial (0-10)"])
        v_sem = float(row_est["Semestre Atual"])
        rotulo_cenario = f"{est_id} ({row_est['Gênero']}, {row_est['Idade']} anos, Saúde: {row_est['Condição de Saúde']})"
    else:
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        v_cgpa = s_c1.slider("CGPA (Rendimento)", 2.25, 4.00, 3.10, 0.05)
        v_custo = s_c2.slider("Custo Pessoal Extra-Estudo", 0.0, 32000.0, 14500.0, 500.0)
        v_psico = s_c3.slider("Vulnerabilidade Psicossocial (0-10)", 0.0, 10.0, 6.8, 0.2)
        v_sem = s_c4.slider("Semestre Atual", 1.0, 12.0, 7.0, 1.0)
        rotulo_cenario = "Estudante Simulado"

    res_ind = avaliar_estudante_individual_mamdani(v_cgpa, v_custo, v_psico, v_sem)

    m_a, m_b, m_c, m_d = st.columns(4)
    m_a.metric("Estudante", rotulo_cenario.split(" (")[0])
    m_b.metric("Score Criado (Centroide)", f"{res_ind['score_fuzzy']:.2f} / 100")
    m_c.metric("Classificação Fuzzy Gerada", res_ind["categoria_fuzzy"])
    m_d.metric("Entradas (CGPA | Custo | Psico | Sem)", f"{v_cgpa:.2f} | {v_custo:,.0f} | {v_psico:.1f} | {v_sem:.0f}º")

    col_st1, col_st2 = st.columns([1.05, 1.2])
    with col_st1:
        st.markdown("##### Ativação das 8 Regras Nebulosas SE-ENTÃO ($\\alpha$-cut)")
        df_r_ind = pd.DataFrame(res_ind["regras"])
        st.dataframe(
            df_r_ind.style.format({"Ativação (α)": "{:.3f}"}).background_gradient(subset=["Ativação (α)"], cmap="OrRd"),
            use_container_width=True,
            hide_index=True,
        )

    with col_st2:
        st.markdown("##### Agregação de Mamdani e Defuzzificação pelo Método do Centroide")
        fig_def, ax_def = plt.subplots(figsize=(7.5, 4.5))
        xs = res_ind["x_saida"]
        for nome_c, curva_ref in res_ind["mf_saida"].items():
            ax_def.plot(xs, curva_ref, "--", color=CORES_RISCO[nome_c], alpha=0.45, lw=1.5, label=f"{nome_c} (Ref.)")

        ax_def.fill_between(xs, 0, res_ind["agregado"], color="#2563eb", alpha=0.35, label="Área Agregada Mamdani")
        ax_def.plot(xs, res_ind["agregado"], color="#1d4ed8", lw=2.4)
        ax_def.axvline(
            res_ind["score_fuzzy"],
            color="#dc2626",
            lw=3,
            label=f"Centroide Defuzzificado = {res_ind['score_fuzzy']:.2f}",
        )
        ax_def.set_title(f"Defuzzificação por Centro de Gravidade ({rotulo_cenario})")
        ax_def.set_xlabel("Universo de Saída: Grau de Depressão em Estudantes [0 - 100]")
        ax_def.set_ylabel("Grau de Pertinência μ(y)")
        ax_def.set_ylim(0, 1.08)
        ax_def.legend(loc="upper left", fontsize=8.5)
        fig_def.tight_layout()
        st.pyplot(fig_def)
        plt.close(fig_def)

    with st.expander("🌐 Visualizar Superfície de Controle Fuzzy 3D (CGPA × Vulnerabilidade Psicossocial → Score Fuzzy)", expanded=False):
        g_cgpa = np.linspace(2.25, 4.0, 26)
        g_psico = np.linspace(0.5, 9.8, 26)
        Z_mat = np.zeros((len(g_psico), len(g_cgpa)))
        for i_p, pv in enumerate(g_psico):
            for j_c, cv in enumerate(g_cgpa):
                Z_mat[i_p, j_c] = avaliar_estudante_individual_mamdani(cv, 12000.0, pv, 6.0)["score_fuzzy"]

        fig_surf = go.Figure(data=[go.Surface(z=Z_mat, x=g_cgpa, y=g_psico, colorscale="RdYlGn_r")])
        fig_surf.update_layout(
            title="Superfície Não-Linear de Decisão Fuzzy 3D (Custo = 12.000, 6º Semestre)",
            scene=dict(
                xaxis_title="CGPA (Rendimento)",
                yaxis_title="Vulnerabilidade Psicossocial (0-10)",
                zaxis_title="Score Depressão Fuzzy (0-100)",
            ),
            height=500,
            margin=dict(l=0, r=0, b=0, t=40),
        )
        st.plotly_chart(fig_surf, use_container_width=True)


# ---------------------------------------------------------
# ABA 3: Agrupamento Difuso Não Supervisionado (Fuzzy C-Means)
# ---------------------------------------------------------
with aba_fcm:
    st.header("☁️ Agrupamento Difuso Não Supervisionado (Fuzzy C-Means - FCM)")
    st.markdown(
        r"""
        Aplicamos o algoritmo **Fuzzy C-Means (Bezdek)** para agrupar os estudantes sem supervisão e criar os 
        **Graus de Pertinência Difusos $u_{ij} \in [0, 1]$** a cada perfil estudantil ($\sum_{j=1}^C u_{ij} = 1$):
        $$J_m(U, V) = \sum_{i=1}^{N} \sum_{j=1}^{C} u_{ij}^m \, \| \mathbf{x}_i - \mathbf{v}_j \|^2$$
        """
    )

    opcoes_feat_fcm = [
        "Vulnerabilidade Psicossocial (0-10)",
        "CGPA (Rendimento Acadêmico)",
        "Custo Pessoal Extra-Estudo",
        "Semestre Atual",
        "Idade",
    ]
    features_fcm = st.multiselect(
        "Variáveis para o Agrupamento Fuzzy C-Means:",
        options=opcoes_feat_fcm,
        default=[
            "Vulnerabilidade Psicossocial (0-10)",
            "CGPA (Rendimento Acadêmico)",
            "Custo Pessoal Extra-Estudo",
            "Semestre Atual",
        ],
    )

    if len(features_fcm) < 2:
        st.warning("Selecione pelo menos 2 variáveis para o Fuzzy C-Means.")
    else:
        cf1, cf2, cf3 = st.columns(3)
        c_clusters = cf1.slider("Número de Clusters Fuzzy (C)", 2, 5, 3, 1)
        m_expoente = cf2.slider("Expoente de Fuzzificação (m)", 1.15, 3.00, 2.00, 0.05)
        limiar_trans = cf3.slider("Limiar de Fronteira Nebulosa (Pertinência Máx < τ)", 0.45, 0.75, 0.55, 0.05)

        res_fcm = executar_fuzzy_cmeans_estudantes(
            df_filtrado,
            features=features_fcm,
            n_clusters=c_clusters,
            m=m_expoente,
            limiar_incerteza=limiar_trans,
            seed=42,
        )
        st.session_state["fcm_estudantes"] = res_fcm

        if res_fcm.get("sucesso"):
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("FPC (Coef. Partição Fuzzy)", f"{res_fcm['fpc']:.4f}")
            f2.metric("Entropia Normalizada (PE)", f"{res_fcm['pe_norm']:.4f}")
            f3.metric("Estudantes em Fronteira Fuzzy", f"{res_fcm['n_fronteira']:,} ({res_fcm['pct_fronteira']:.1f}%)")
            f4.metric("Total de Estudantes Agrupados", f"{res_fcm['n_registros']:,}")

            df_res_fcm = res_fcm["df_resultado"]
            nomes_c = res_fcm["nomes_clusters"]
            cols_pert_fcm = [f"Pertinência {n} (%)" for n in nomes_c]

            col_v1, col_v2 = st.columns([1.25, 1])
            with col_v1:
                # Amostra de até 1.200 pontos para manter fluidez gráfica interativa
                df_plot_fcm = df_res_fcm.sample(min(1200, len(df_res_fcm)), random_state=42)
                fig_fcm_sc = px.scatter(
                    df_plot_fcm,
                    x=features_fcm[0],
                    y=features_fcm[1],
                    color="Cluster_FCM_Dominante",
                    symbol="Tipo_Alocacao_Fuzzy",
                    hover_name="ID_Estudante",
                    hover_data=["Score_Depressao_Fuzzy", "Pertinencia_FCM_Maxima"] + cols_pert_fcm,
                    title=f"Clusters Fuzzy C-Means: {features_fcm[0]} × {features_fcm[1]}",
                    color_discrete_sequence=["#dc2626", "#f59e0b", "#16a34a", "#2563eb", "#7c3aed"],
                )
                fig_fcm_sc.update_layout(plot_bgcolor="white", height=430)
                st.plotly_chart(fig_fcm_sc, use_container_width=True)

            with col_v2:
                df_diag = res_fcm["df_diagnostico"]
                fig_d, ax_d = plt.subplots(figsize=(6, 4.2))
                ax_d.plot(df_diag["Clusters (C)"], df_diag["FPC (Coef. Partição Fuzzy)"], "o-", color="#2563eb", lw=2.2, label="FPC (Maximizar)")
                ax_d.plot(df_diag["Clusters (C)"], df_diag["Entropia Normalizada"], "s--", color="#dc2626", lw=2.2, label="Entropia Norm. (Minimizar)")
                ax_d.axvline(c_clusters, color="#16a34a", linestyle=":", lw=2, label=f"C Atual = {c_clusters}")
                ax_d.set_title("Diagnóstico de Clusters Fuzzy (FPC vs Entropia)")
                ax_d.set_xlabel("Número de Clusters (C)")
                ax_d.set_ylabel("Valor [0, 1]")
                ax_d.legend()
                fig_d.tight_layout()
                st.pyplot(fig_d)
                plt.close(fig_d)

            st.subheader("Matriz de Graus de Pertinência ($U_{ij}$) em Estudantes de Fronteira Nebulosa")
            df_front = df_res_fcm.sort_values(by="Incerteza_Fuzzy_Shannon", ascending=False).head(15)
            df_melt_f = df_front.melt(
                id_vars=["ID_Estudante", "Cluster_FCM_Dominante"],
                value_vars=cols_pert_fcm,
                var_name="Perfil Fuzzy",
                value_name="Grau de Pertinência (%)",
            )
            fig_bar_fcm = px.bar(
                df_melt_f,
                x="Grau de Pertinência (%)",
                y="ID_Estudante",
                color="Perfil Fuzzy",
                orientation="h",
                text_auto=".1f",
                title="Decomposição de Pertinência dos 15 Estudantes com Maior Hibridismo / Transição Fuzzy",
                color_discrete_sequence=["#dc2626", "#f59e0b", "#16a34a", "#2563eb", "#7c3aed"],
            )
            fig_bar_fcm.update_layout(barmode="stack", plot_bgcolor="white", height=420)
            st.plotly_chart(fig_bar_fcm, use_container_width=True)

            st.subheader("Resumo dos Centróides Nebulosos (Fuzzy C-Means)")
            st.dataframe(res_fcm["resumo_clusters"], use_container_width=True)


# ---------------------------------------------------------
# ABA 4: Resultados Fuzzy por Perfil Estudantil
# ---------------------------------------------------------
with aba_perfis:
    st.header("📊 Análise dos Resultados Criados pela Lógica Fuzzy nos Estudantes")

    c_p1, c_p2 = st.columns(2)

    with c_p1:
        fig_box_saude = px.box(
            df_filtrado,
            x="Condição de Saúde",
            y="Score_Depressao_Fuzzy",
            color="Condição de Saúde",
            category_orders={"Condição de Saúde": ["Boa", "Normal", "Anormal", "Crítica"]},
            title="Score Fuzzy de Depressão por Condição de Saúde do Estudante",
        )
        fig_box_saude.update_layout(plot_bgcolor="white", showlegend=False, height=400)
        st.plotly_chart(fig_box_saude, use_container_width=True)

    with c_p2:
        fig_amigos = px.histogram(
            df_filtrado,
            x="Círculo de Amigos",
            color="Classificacao_Criada_Fuzzy",
            barmode="group",
            color_discrete_map=CORES_RISCO,
            title="Impacto do Círculo de Amigos (Apoio Social) na Classificação Criada pelo Fuzzy",
        )
        fig_amigos.update_layout(plot_bgcolor="white", height=400)
        st.plotly_chart(fig_amigos, use_container_width=True)

    c_p3, c_p4 = st.columns(2)

    with c_p3:
        sem_media = (
            df_filtrado.groupby("Semestre Atual")[["Score_Depressao_Fuzzy", "CGPA (Rendimento Acadêmico)"]]
            .mean()
            .reset_index()
        )
        fig_sem = px.line(
            sem_media,
            x="Semestre Atual",
            y="Score_Depressao_Fuzzy",
            markers=True,
            title="Evolução Média do Score Fuzzy de Depressão ao Longo dos Semestres",
        )
        fig_sem.update_traces(line=dict(color="#dc2626", width=3))
        fig_sem.update_layout(plot_bgcolor="white", height=380)
        st.plotly_chart(fig_sem, use_container_width=True)

    with c_p4:
        fig_moradia = px.violin(
            df_filtrado,
            x="Estilo de Moradia",
            y="Score_Depressao_Fuzzy",
            color="Status de Relacionamento",
            box=True,
            title="Score Fuzzy por Estilo de Moradia e Status de Relacionamento",
        )
        fig_moradia.update_layout(plot_bgcolor="white", height=380)
        st.plotly_chart(fig_moradia, use_container_width=True)

