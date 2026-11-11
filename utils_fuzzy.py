import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# 1. FUNÇÕES DE PERTINÊNCIA FUZZY (TRAPEZOIDAL E TRIANGULAR VETORIZADAS)
# ==============================================================================
def trapmf(x, a, b, c, d):
    """Função de pertinência Trapezoidal vetorizada μ(x; a, b, c, d) ∈ [0, 1]."""
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr, dtype=float)

    if b > a:
        mask_up = (x_arr > a) & (x_arr < b)
        y[mask_up] = (x_arr[mask_up] - a) / (b - a)

    mask_top = (x_arr >= b) & (x_arr <= c)
    y[mask_top] = 1.0

    if d > c:
        mask_down = (x_arr > c) & (x_arr < d)
        y[mask_down] = (d - x_arr[mask_down]) / (d - c)

    return np.clip(y, 0.0, 1.0)


def trimf(x, a, b, c):
    """Função de pertinência Triangular vetorizada μ(x; a, b, c) ∈ [0, 1]."""
    return trapmf(x, a, b, b, c)


# ==============================================================================
# 2. PREPARAÇÃO NÃO SUPERVISIONADA E CRIAÇÃO DO ÍNDICE PSICOSSOCIAL
# ==============================================================================
MAPA_GENERO = {"Female": "Feminino", "Male": "Masculino"}
MAPA_FAMILIA = {
    "Lower Middle class": "Classe Média-Baixa",
    "Middle class": "Classe Média",
    "Rich class": "Classe Alta",
}
MAPA_RELACIONAMENTO = {
    "Single": "Solteiro(a)",
    "In a Relationship": "Em Relacionamento",
    "Complicated": "Relacionamento Complicado",
}
MAPA_ESTADO_CIVIL = {
    "Unmarried": "Solteiro / Não Casado",
    "Married": "Casado(a)",
    "Separated": "Separado(a)",
}
MAPA_MORADIA = {
    "Bachelor": "Sozinho / República",
    "Family": "Com a Família",
}
MAPA_AMIGOS = {"Yes": "Sim (Com Rede de Apoio)", "No": "Não (Isolamento Social)"}
MAPA_SAUDE = {
    "Good": "Boa",
    "Normal": "Normal",
    "Abnormal": "Anormal",
    "Critical": "Crítica",
}


def preparar_base_estudantes_nao_supervisionada(df_raw):
    """
    Remove qualquer variável target supervisionada ('Depression') da base original do Kaggle
    e estrutura os atributos observacionais dos estudantes para criação dos dados via Lógica Fuzzy.
    """
    df = df_raw.copy()

    if "Depression" in df.columns:
        df = df.drop(columns=["Depression"])

    df = df.rename(columns={
        "Gender": "Gênero",
        "Age": "Idade",
        "Family_background": "Classe Familiar",
        "Relationship_status": "Status de Relacionamento",
        "Marital_status": "Estado Civil",
        "Living_style": "Estilo de Moradia",
        "Current_semester": "Semestre Atual",
        "CGPA": "CGPA (Rendimento Acadêmico)",
        "friend_circle": "Círculo de Amigos",
        "Health_condition": "Condição de Saúde",
        "Cost_without_study": "Custo Pessoal Extra-Estudo",
        "Home_division": "Divisão Regional",
    })

    df["Gênero"] = df["Gênero"].replace(MAPA_GENERO)
    df["Classe Familiar"] = df["Classe Familiar"].replace(MAPA_FAMILIA)
    df["Status de Relacionamento"] = df["Status de Relacionamento"].replace(MAPA_RELACIONAMENTO)
    df["Estado Civil"] = df["Estado Civil"].replace(MAPA_ESTADO_CIVIL)
    df["Estilo de Moradia"] = df["Estilo de Moradia"].replace(MAPA_MORADIA)
    df["Círculo de Amigos"] = df["Círculo de Amigos"].replace(MAPA_AMIGOS)
    df["Condição de Saúde"] = df["Condição de Saúde"].replace(MAPA_SAUDE)

    peso_saude = df["Condição de Saúde"].map({
        "Boa": 1.0,
        "Normal": 3.0,
        "Anormal": 6.0,
        "Crítica": 7.5,
    }).fillna(3.0)

    peso_amigos = np.where(df["Círculo de Amigos"].str.startswith("Não"), 1.2, 0.0)
    peso_moradia = np.where(df["Estilo de Moradia"].str.startswith("Sozinho"), 0.5, 0.0)
    peso_relac = df["Status de Relacionamento"].map({
        "Em Relacionamento": 0.0,
        "Solteiro(a)": 0.3,
        "Relacionamento Complicado": 0.8,
    }).fillna(0.3)
    peso_familia = df["Classe Familiar"].map({
        "Classe Alta": 0.0,
        "Classe Média": 0.2,
        "Classe Média-Baixa": 0.5,
    }).fillna(0.2)

    df["Vulnerabilidade Psicossocial (0-10)"] = np.clip(
        peso_saude + peso_amigos + peso_moradia + peso_relac + peso_familia,
        0.0,
        10.0,
    ).round(2)

    return df


# ==============================================================================
# 3. UNIVERSOS DE DISCURSO E FUNÇÕES DE PERTINÊNCIA (ESTUDANTES)
# ==============================================================================
def obter_universos_fuzzy_estudantes():
    """
    Define os universos de discurso e funções de pertinência para detecção de depressão
    em estudantes via Lógica Fuzzy.
    """
    x_cgpa = np.linspace(2.0, 4.0, 400)
    mf_cgpa = {
        "Baixo (Crítico)": trapmf(x_cgpa, 2.0, 2.0, 2.85, 3.25),
        "Médio (Regular)": trimf(x_cgpa, 2.95, 3.35, 3.70),
        "Alto (Excelente)": trapmf(x_cgpa, 3.50, 3.75, 4.0, 4.0),
    }

    x_custo = np.linspace(0, 32000, 400)
    mf_custo = {
        "Baixo": trapmf(x_custo, 0, 0, 5000, 9500),
        "Moderado": trimf(x_custo, 6500, 12000, 17500),
        "Elevado": trapmf(x_custo, 14000, 19500, 32000, 32000),
    }

    x_psico = np.linspace(0, 10, 400)
    mf_psico = {
        "Baixa": trapmf(x_psico, 0, 0, 2.5, 4.5),
        "Moderada": trimf(x_psico, 3.2, 5.5, 7.5),
        "Severa": trapmf(x_psico, 6.2, 7.8, 10.0, 10.0),
    }

    x_sem = np.linspace(0, 13, 400)
    mf_sem = {
        "Inicial": trapmf(x_sem, 0, 0, 2.5, 4.5),
        "Intermediário": trimf(x_sem, 3.0, 5.5, 8.0),
        "Avançado / Final": trapmf(x_sem, 6.5, 8.5, 13.0, 13.0),
    }

    x_saida = np.linspace(0, 100, 400)
    mf_saida = {
        "Baixo Risco": trapmf(x_saida, 0, 0, 18, 34),
        "Risco Moderado": trimf(x_saida, 22, 44, 65),
        "Alto Risco": trimf(x_saida, 52, 72, 88),
        "Risco Crítico": trapmf(x_saida, 76, 88, 100, 100),
    }

    return {
        "cgpa": (x_cgpa, mf_cgpa),
        "custo": (x_custo, mf_custo),
        "psico": (x_psico, mf_psico),
        "semestre": (x_sem, mf_sem),
        "saida": (x_saida, mf_saida),
    }


DESCRICAO_REGRAS_MAMDANI = [
    ("R1", "SE Vulnerabilidade é Severa E (CGPA é Baixo OU Custo é Elevado)", "Risco Crítico"),
    ("R2", "SE Vulnerabilidade é Severa E Semestre é Avançado/Final", "Risco Crítico"),
    ("R3", "SE Vulnerabilidade é Severa OU (CGPA é Baixo E Custo é Elevado)", "Alto Risco"),
    ("R4", "SE Vulnerabilidade é Moderada E (CGPA é Baixo OU Semestre é Avançado)", "Alto Risco"),
    ("R5", "SE Vulnerabilidade é Moderada E CGPA é Médio", "Risco Moderado"),
    ("R6", "SE Custo é Elevado E CGPA é Médio E Semestre é Intermediário", "Risco Moderado"),
    ("R7", "SE Vulnerabilidade é Baixa E CGPA é Médio", "Baixo Risco"),
    ("R8", "SE Vulnerabilidade é Baixa E CGPA é Alto E Custo é Baixo/Moderado", "Baixo Risco"),
]


# ==============================================================================
# 4. CRIAÇÃO COMPLETA DOS DADOS VIA INFERÊNCIA FUZZY DE MAMDANI
# ==============================================================================
def criar_dados_fuzzy_estudantes(df_base):
    """
    Aplica a Lógica Fuzzy completa para CRIAR todos os dados de detecção de depressão:
    - 12 colunas de Graus de Pertinência Fuzzificados de Entrada (Mu_*)
    - 8 colunas de Ativação das Regras de Mamdani (Regra_R1_Alfa .. Regra_R8_Alfa)
    - 4 colunas de Pertinência de Saída (Pert_Fuzzy_*)
    - Score_Depressao_Fuzzy (0 a 100 via Centroide)
    - Classificacao_Criada_Fuzzy e Regra_Fuzzy_Dominante
    """
    df = df_base.copy()

    cgpa = np.clip(df["CGPA (Rendimento Acadêmico)"].values.astype(float), 2.0, 4.0)
    custo = np.clip(df["Custo Pessoal Extra-Estudo"].values.astype(float), 0.0, 32000.0)
    psico = np.clip(df["Vulnerabilidade Psicossocial (0-10)"].values.astype(float), 0.0, 10.0)
    sem = np.clip(df["Semestre Atual"].values.astype(float), 0.0, 13.0)

    # 1. Fuzzificação das entradas
    cgpa_baixo = trapmf(cgpa, 2.0, 2.0, 2.85, 3.25)
    cgpa_medio = trimf(cgpa, 2.95, 3.35, 3.70)
    cgpa_alto = trapmf(cgpa, 3.50, 3.75, 4.0, 4.0)

    custo_baixo = trapmf(custo, 0, 0, 5000, 9500)
    custo_mod = trimf(custo, 6500, 12000, 17500)
    custo_elev = trapmf(custo, 14000, 19500, 32000, 32000)

    psico_baixa = trapmf(psico, 0, 0, 2.5, 4.5)
    psico_mod = trimf(psico, 3.2, 5.5, 7.5)
    psico_sev = trapmf(psico, 6.2, 7.8, 10.0, 10.0)

    sem_ini = trapmf(sem, 0, 0, 2.5, 4.5)
    sem_int = trimf(sem, 3.0, 5.5, 8.0)
    sem_avanc = trapmf(sem, 6.5, 8.5, 13.0, 13.0)

    # Salvar todos os 12 graus de pertinência de entrada no DataFrame
    df["Mu_CGPA_Baixo"] = np.round(cgpa_baixo, 4)
    df["Mu_CGPA_Medio"] = np.round(cgpa_medio, 4)
    df["Mu_CGPA_Alto"] = np.round(cgpa_alto, 4)

    df["Mu_Custo_Baixo"] = np.round(custo_baixo, 4)
    df["Mu_Custo_Moderado"] = np.round(custo_mod, 4)
    df["Mu_Custo_Elevado"] = np.round(custo_elev, 4)

    df["Mu_Psico_Baixa"] = np.round(psico_baixa, 4)
    df["Mu_Psico_Moderada"] = np.round(psico_mod, 4)
    df["Mu_Psico_Severa"] = np.round(psico_sev, 4)

    df["Mu_Semestre_Inicial"] = np.round(sem_ini, 4)
    df["Mu_Semestre_Intermediario"] = np.round(sem_int, 4)
    df["Mu_Semestre_Avancado"] = np.round(sem_avanc, 4)

    # 2. Avaliação das 8 Regras Fuzzy
    r1 = np.minimum(psico_sev, np.maximum(cgpa_baixo, custo_elev))
    r2 = np.minimum(psico_sev, sem_avanc)
    r3 = np.maximum(psico_sev, np.minimum(cgpa_baixo, custo_elev))
    r4 = np.minimum(psico_mod, np.maximum(cgpa_baixo, sem_avanc))
    r5 = np.minimum(psico_mod, cgpa_medio)
    r6 = np.minimum(np.minimum(custo_elev, cgpa_medio), sem_int)
    r7 = np.minimum(psico_baixa, cgpa_medio)
    r8 = np.minimum(np.minimum(psico_baixa, cgpa_alto), np.maximum(custo_baixo, custo_mod))

    df["Regra_R1_Alfa"] = np.round(r1, 4)
    df["Regra_R2_Alfa"] = np.round(r2, 4)
    df["Regra_R3_Alfa"] = np.round(r3, 4)
    df["Regra_R4_Alfa"] = np.round(r4, 4)
    df["Regra_R5_Alfa"] = np.round(r5, 4)
    df["Regra_R6_Alfa"] = np.round(r6, 4)
    df["Regra_R7_Alfa"] = np.round(r7, 4)
    df["Regra_R8_Alfa"] = np.round(r8, 4)

    alfa_baixo = np.maximum(r7, r8)
    alfa_moderado = np.maximum(r5, r6)
    alfa_alto = np.maximum(r3, r4)
    alfa_critico = np.maximum(r1, r2)

    # 3. Agregação de Mamdani e Defuzzificação por Centroide
    univ = obter_universos_fuzzy_estudantes()
    x_saida, mf_saida = univ["saida"]

    c_baixo = np.minimum(alfa_baixo[:, None], mf_saida["Baixo Risco"][None, :])
    c_mod = np.minimum(alfa_moderado[:, None], mf_saida["Risco Moderado"][None, :])
    c_alto = np.minimum(alfa_alto[:, None], mf_saida["Alto Risco"][None, :])
    c_crit = np.minimum(alfa_critico[:, None], mf_saida["Risco Crítico"][None, :])

    agregado = np.maximum(np.maximum(c_baixo, c_mod), np.maximum(c_alto, c_crit))
    soma_area = np.sum(agregado, axis=1)
    soma_area_safe = np.fmax(soma_area, 1e-9)
    scores_centroide = np.where(
        soma_area > 1e-9,
        np.sum(agregado * x_saida[None, :], axis=1) / soma_area_safe,
        35.0,
    )

    soma_alfas = np.fmax(alfa_baixo + alfa_moderado + alfa_alto + alfa_critico, 1e-9)
    df["Pert_Fuzzy_Baixo_Risco"] = np.round(alfa_baixo / soma_alfas, 4)
    df["Pert_Fuzzy_Risco_Moderado"] = np.round(alfa_moderado / soma_alfas, 4)
    df["Pert_Fuzzy_Alto_Risco"] = np.round(alfa_alto / soma_alfas, 4)
    df["Pert_Fuzzy_Risco_Critico"] = np.round(alfa_critico / soma_alfas, 4)

    df["Score_Depressao_Fuzzy"] = np.round(scores_centroide, 2)

    conds = [
        df["Score_Depressao_Fuzzy"] >= 74.0,
        df["Score_Depressao_Fuzzy"] >= 54.0,
        df["Score_Depressao_Fuzzy"] >= 34.0,
    ]
    choices = ["Risco Crítico", "Alto Risco", "Risco Moderado"]
    df["Classificacao_Criada_Fuzzy"] = np.select(conds, choices, default="Baixo Risco")

    matriz_regras = np.vstack([r1, r2, r3, r4, r5, r6, r7, r8]).T
    idx_regra = np.argmax(matriz_regras, axis=1)
    val_regra = np.max(matriz_regras, axis=1)
    df["Regra_Fuzzy_Dominante"] = [f"R{idx + 1} (α={val:.2f})" for idx, val in zip(idx_regra, val_regra)]

    return df


def avaliar_estudante_individual_mamdani(cgpa_val, custo_val, psico_val, sem_val):
    """Executa a inferência Fuzzy passo a passo para um único estudante."""
    cgpa_v = float(np.clip(cgpa_val, 2.0, 4.0))
    custo_v = float(np.clip(custo_val, 0.0, 32000.0))
    psico_v = float(np.clip(psico_val, 0.0, 10.0))
    sem_v = float(np.clip(sem_val, 0.0, 13.0))

    mu_cgpa = {
        "Baixo (Crítico)": float(trapmf(cgpa_v, 2.0, 2.0, 2.85, 3.25)),
        "Médio (Regular)": float(trimf(cgpa_v, 2.95, 3.35, 3.70)),
        "Alto (Excelente)": float(trapmf(cgpa_v, 3.50, 3.75, 4.0, 4.0)),
    }
    mu_custo = {
        "Baixo": float(trapmf(custo_v, 0, 0, 5000, 9500)),
        "Moderado": float(trimf(custo_v, 6500, 12000, 17500)),
        "Elevado": float(trapmf(custo_v, 14000, 19500, 32000, 32000)),
    }
    mu_psico = {
        "Baixa": float(trapmf(psico_v, 0, 0, 2.5, 4.5)),
        "Moderada": float(trimf(psico_v, 3.2, 5.5, 7.5)),
        "Severa": float(trapmf(psico_v, 6.2, 7.8, 10.0, 10.0)),
    }
    mu_sem = {
        "Inicial": float(trapmf(sem_v, 0, 0, 2.5, 4.5)),
        "Intermediário": float(trimf(sem_v, 3.0, 5.5, 8.0)),
        "Avançado / Final": float(trapmf(sem_v, 6.5, 8.5, 13.0, 13.0)),
    }

    r1 = min(mu_psico["Severa"], max(mu_cgpa["Baixo (Crítico)"], mu_custo["Elevado"]))
    r2 = min(mu_psico["Severa"], mu_sem["Avançado / Final"])
    r3 = max(mu_psico["Severa"], min(mu_cgpa["Baixo (Crítico)"], mu_custo["Elevado"]))
    r4 = min(mu_psico["Moderada"], max(mu_cgpa["Baixo (Crítico)"], mu_sem["Avançado / Final"]))
    r5 = min(mu_psico["Moderada"], mu_cgpa["Médio (Regular)"])
    r6 = min(mu_custo["Elevado"], mu_cgpa["Médio (Regular)"], mu_sem["Intermediário"])
    r7 = min(mu_psico["Baixa"], mu_cgpa["Médio (Regular)"])
    r8 = min(mu_psico["Baixa"], mu_cgpa["Alto (Excelente)"], max(mu_custo["Baixo"], mu_custo["Moderado"]))

    alfas_r = [r1, r2, r3, r4, r5, r6, r7, r8]
    regras_detalhe = [
        {
            "Regra": cod,
            "Antecedente (SE)": ant,
            "Consequente (ENTÃO)": cons,
            "Ativação (α)": alfa_v,
        }
        for (cod, ant, cons), alfa_v in zip(DESCRICAO_REGRAS_MAMDANI, alfas_r)
    ]

    univ = obter_universos_fuzzy_estudantes()
    x_saida, mf_saida = univ["saida"]

    ativacoes = {
        "Baixo Risco": max(r7, r8),
        "Risco Moderado": max(r5, r6),
        "Alto Risco": max(r3, r4),
        "Risco Crítico": max(r1, r2),
    }

    agregado = np.zeros_like(x_saida)
    for nome_c, alfa in ativacoes.items():
        agregado = np.maximum(agregado, np.minimum(alfa, mf_saida[nome_c]))

    soma_area = float(np.sum(agregado))
    score_centroide = float(np.sum(x_saida * agregado) / soma_area) if soma_area > 1e-9 else 35.0

    if score_centroide >= 74.0:
        categoria = "Risco Crítico"
    elif score_centroide >= 54.0:
        categoria = "Alto Risco"
    elif score_centroide >= 34.0:
        categoria = "Risco Moderado"
    else:
        categoria = "Baixo Risco"

    return {
        "score_fuzzy": score_centroide,
        "categoria_fuzzy": categoria,
        "mu_entradas": {
            "CGPA (Rendimento)": mu_cgpa,
            "Custo Extra-Estudo": mu_custo,
            "Vulnerabilidade Psicossocial": mu_psico,
            "Semestre Atual": mu_sem,
        },
        "regras": regras_detalhe,
        "ativacoes": ativacoes,
        "x_saida": x_saida,
        "mf_saida": mf_saida,
        "agregado": agregado,
    }


# ==============================================================================
# 5. AGRUPAMENTO DIFUSO FUZZY C-MEANS (FCM) PARA ESTUDANTES
# ==============================================================================
def _calcular_fcm_core(X_scaled, n_clusters=3, m=2.0, max_iter=120, tol=1e-5, seed=42):
    rng = np.random.default_rng(int(seed))
    N, _ = X_scaled.shape
    C = int(n_clusters)
    m = float(m)

    U = rng.uniform(0.05, 1.0, size=(N, C))
    U = U / np.sum(U, axis=1, keepdims=True)

    jm_history = []
    exp_power = 2.0 / (m - 1.0)

    for _ in range(max_iter):
        U_old = U.copy()
        um = U ** m
        centers = (um.T @ X_scaled) / np.sum(um, axis=0)[:, None]

        dist = np.linalg.norm(X_scaled[:, None, :] - centers[None, :, :], axis=2)
        dist = np.fmax(dist, 1e-10)

        jm = float(np.sum(um * (dist ** 2)))
        jm_history.append(jm)

        inv_dist = (1.0 / dist) ** exp_power
        U = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)

        if np.linalg.norm(U - U_old, ord="fro") < tol:
            break

    fpc = float(np.sum(U ** 2) / N)
    u_safe = np.fmax(U, 1e-12)
    pe = float(-np.sum(U * np.log(u_safe)) / N)
    pe_norm = pe / np.log(C) if C > 1 else 0.0

    return U, centers, jm_history, fpc, pe, pe_norm


def executar_fuzzy_cmeans_estudantes(df_dados, features, n_clusters=3, m=2.0, limiar_incerteza=0.55, seed=42):
    """
    Executa o agrupamento não supervisionado Fuzzy C-Means sobre os atributos dos estudantes,
    anexando todas as colunas de pertinência U_ij, cluster dominante, entropia e fronteira nebulosa.
    """
    df_clean = df_dados.dropna(subset=features).copy().reset_index(drop=True)
    n_registros = len(df_clean)

    if n_registros < n_clusters + 2:
        return {"sucesso": False, "erro": "Registros insuficientes para o agrupamento Fuzzy C-Means."}

    X = df_clean[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    U, centers_scaled, jm_history, fpc, pe, pe_norm = _calcular_fcm_core(
        X_scaled, n_clusters=n_clusters, m=m, seed=seed
    )

    if "Score_Depressao_Fuzzy" in df_clean.columns:
        scores_c = [
            float(np.sum((U[:, j] ** m) * df_clean["Score_Depressao_Fuzzy"].values) / np.sum(U[:, j] ** m))
            for j in range(n_clusters)
        ]
        ordem = np.argsort(scores_c)[::-1]
    else:
        ordem = np.argsort(centers_scaled[:, 0])[::-1]

    U = U[:, ordem]
    centers_scaled = centers_scaled[ordem, :]

    if n_clusters == 3:
        nomes_clusters = [
            "Cluster 0 (Alta Vulnerabilidade / Depressão)",
            "Cluster 1 (Transição / Estresse Moderado)",
            "Cluster 2 (Equilíbrio / Baixo Risco)",
        ]
    elif n_clusters == 4:
        nomes_clusters = [
            "Cluster 0 (Risco Crítico)",
            "Cluster 1 (Alto Risco)",
            "Cluster 2 (Risco Moderado)",
            "Cluster 3 (Baixo Risco)",
        ]
    else:
        nomes_clusters = [f"Cluster Fuzzy {j}" for j in range(n_clusters)]

    cluster_dom = np.argmax(U, axis=1)
    pert_max = np.max(U, axis=1)
    u_safe = np.fmax(U, 1e-12)
    incerteza_shannon = -np.sum(U * np.log(u_safe), axis=1) / np.log(n_clusters)

    for j in range(n_clusters):
        df_clean[f"Pertinencia_FCM_C{j}"] = np.round(U[:, j], 4)
        df_clean[f"Pertinência {nomes_clusters[j]} (%)"] = np.round(U[:, j] * 100, 2)

    df_clean["Cluster_FCM_ID"] = cluster_dom
    df_clean["Cluster_FCM_Dominante"] = [nomes_clusters[c] for c in cluster_dom]
    df_clean["Pertinencia_FCM_Maxima"] = np.round(pert_max, 4)
    df_clean["Incerteza_Fuzzy_Shannon"] = np.round(incerteza_shannon, 4)
    df_clean["Estudante_Fronteira_Fuzzy"] = pert_max < float(limiar_incerteza)
    df_clean["Tipo_Alocacao_Fuzzy"] = np.where(
        df_clean["Estudante_Fronteira_Fuzzy"],
        "Fronteira Nebulosa (Transição entre Perfis)",
        "Núcleo Nítido (Alta Pertinência)",
    )

    resumo_list = []
    for j in range(n_clusters):
        sub_dom = df_clean[df_clean["Cluster_FCM_ID"] == j]
        pesos = U[:, j] ** m
        soma_p = np.sum(pesos)
        item = {
            "Perfil Fuzzy C-Means": nomes_clusters[j],
            "Qtd Estudantes (Dominante)": len(sub_dom),
            "% Dominante": f"{(len(sub_dom) / n_registros) * 100:.1f}%",
            "Pertinência Média Global (%)": f"{np.mean(U[:, j]) * 100:.1f}%",
            "Em Zona de Transição": int(sub_dom["Estudante_Fronteira_Fuzzy"].sum()),
        }
        for idx_f, f in enumerate(features):
            item[f"Centróide ({f})"] = round(float(np.sum(pesos * X[:, idx_f]) / soma_p), 2) if soma_p > 0 else 0.0
        if "Score_Depressao_Fuzzy" in df_clean.columns:
            item["Score Mamdani Médio"] = round(float(np.sum(pesos * df_clean["Score_Depressao_Fuzzy"].values) / soma_p), 2)
        resumo_list.append(item)
    resumo_df = pd.DataFrame(resumo_list)

    diag_c = []
    for c_test in range(2, 7):
        _, _, _, fpc_t, pe_t, pe_norm_t = _calcular_fcm_core(X_scaled, n_clusters=c_test, m=m, max_iter=60, seed=seed)
        diag_c.append({
            "Clusters (C)": c_test,
            "FPC (Coef. Partição Fuzzy)": round(fpc_t, 4),
            "Entropia da Partição (PE)": round(pe_t, 4),
            "Entropia Normalizada": round(pe_norm_t, 4),
        })
    df_diag = pd.DataFrame(diag_c)

    return {
        "sucesso": True,
        "df_resultado": df_clean,
        "U": U,
        "nomes_clusters": nomes_clusters,
        "jm_history": jm_history,
        "fpc": fpc,
        "pe": pe,
        "pe_norm": pe_norm,
        "n_clusters": int(n_clusters),
        "m": float(m),
        "limiar_incerteza": float(limiar_incerteza),
        "n_registros": n_registros,
        "n_fronteira": int(df_clean["Estudante_Fronteira_Fuzzy"].sum()),
        "pct_fronteira": float(df_clean["Estudante_Fronteira_Fuzzy"].mean() * 100),
        "resumo_clusters": resumo_df,
        "df_diagnostico": df_diag,
        "features": features,
    }


# ==============================================================================
# 6. GERADOR DE RELATÓRIO PDF COMPLETO DE TODO O PROJETO (COM GRÁFICOS E TABELAS)
# ==============================================================================
def _latin1_safe(texto):
    """
    Garante compatibilidade total com a codificação Latin-1 (cp1252) nativa do FPDF2
    em qualquer sistema operacional (Windows e Linux/Streamlit Cloud), preservando todos
    os acentos do português (á, é, í, ó, ú, ã, õ, ç, â, ê, ô) e substituindo símbolos gregos.
    """
    s = str(texto)
    substituicoes = {
        "μ": "u",
        "α": "alfa",
        "τ": "tau",
        "≥": ">=",
        "≤": "<=",
        "×": "x",
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
        "•": "*",
        "‰": "o/oo",
    }
    for k, v in substituicoes.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def generate_student_fuzzy_pdf_report(df_completo, fcm_result=None):
    """
    Gera um Relatório Técnico-Executivo em PDF COMPLETO (múltiplas páginas, todas as tabelas,
    todas as regras, estatísticas descritivas, centróides Fuzzy C-Means, cruzamentos de perfil
    e 4 painéis gráficos de alta resolução embutidos), sem deixar nenhuma informação de fora.
    """
    try:
        from fpdf import FPDF

        # Se fcm_result não foi passado ou falhou, executa com configuração padrão
        if not fcm_result or not fcm_result.get("sucesso"):
            feats_default = [
                "Vulnerabilidade Psicossocial (0-10)",
                "CGPA (Rendimento Acadêmico)",
                "Custo Pessoal Extra-Estudo",
                "Semestre Atual",
            ]
            fcm_result = executar_fuzzy_cmeans_estudantes(df_completo, feats_default, n_clusters=3, m=2.0, limiar_incerteza=0.55)

        df = fcm_result["df_resultado"] if (fcm_result and fcm_result.get("sucesso")) else df_completo.copy()
        n_total = len(df)

        CORES_RISCO = {
            "Baixo Risco": "#16a34a",
            "Risco Moderado": "#eab308",
            "Alto Risco": "#f97316",
            "Risco Crítico": "#dc2626",
        }

        # Criar arquivos temporários para os 4 painéis gráficos do PDF
        tmp_files = []

        # --- PAINEL GRÁFICO 1: Distribuição das Classes Criadas + Histograma do Score Fuzzy ---
        f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=160)
        contagem = df["Classificacao_Criada_Fuzzy"].value_counts().reindex(
            ["Baixo Risco", "Risco Moderado", "Alto Risco", "Risco Crítico"], fill_value=0
        )
        bars = ax1.bar(contagem.index, contagem.values, color=[CORES_RISCO[k] for k in contagem.index], edgecolor="#222")
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h + max(1, n_total * 0.01), f"{int(h)}\n({(h/n_total)*100:.1f}%)", ha="center", fontsize=8, weight="bold")
        ax1.set_ylim(0, max(contagem.values) * 1.25 if len(contagem) > 0 else 10)
        ax1.set_title("Distribuição das Classes Criadas pela Lógica Fuzzy", weight="bold", fontsize=10)
        ax1.set_ylabel("Qtd. de Estudantes")
        ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

        ax2.hist(df["Score_Depressao_Fuzzy"], bins=30, color="#2563eb", edgecolor="white", alpha=0.85)
        ax2.axvline(df["Score_Depressao_Fuzzy"].mean(), color="#dc2626", linestyle="--", lw=2, label=f"Média = {df['Score_Depressao_Fuzzy'].mean():.2f}")
        ax2.axvline(df["Score_Depressao_Fuzzy"].median(), color="#16a34a", linestyle=":", lw=2, label=f"Mediana = {df['Score_Depressao_Fuzzy'].median():.2f}")
        ax2.set_title("Distribuição Contínua do Score de Depressão Fuzzy (0-100)", weight="bold", fontsize=10)
        ax2.set_xlabel("Score Criado via Centroide Mamdani")
        ax2.set_ylabel("Frequência")
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.6)
        f1.tight_layout()
        p1_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        f1.savefig(p1_path, dpi=160)
        plt.close(f1)
        tmp_files.append(p1_path)

        # --- PAINEL GRÁFICO 2: As 4 Funções de Pertinência + Defuzzificação por Centroide ---
        f2, axes2 = plt.subplots(2, 2, figsize=(10.5, 5.2), dpi=160)
        univ = obter_universos_fuzzy_estudantes()

        xc, mfc = univ["cgpa"]
        for k, v in mfc.items():
            axes2[0, 0].plot(xc, v, lw=2, label=k)
        axes2[0, 0].set_title("Entrada 1: CGPA (Rendimento Acadêmico [2.0 - 4.0])", weight="bold", fontsize=9.5)
        axes2[0, 0].legend(fontsize=7.5)
        axes2[0, 0].grid(True, linestyle=":", alpha=0.6)

        xcu, mfcu = univ["custo"]
        for k, v in mfcu.items():
            axes2[0, 1].plot(xcu, v, lw=2, label=k)
        axes2[0, 1].set_title("Entrada 2: Custo Pessoal Extra-Estudo [0 - 32.000]", weight="bold", fontsize=9.5)
        axes2[0, 1].legend(fontsize=7.5)
        axes2[0, 1].grid(True, linestyle=":", alpha=0.6)

        xp, mfp = univ["psico"]
        for k, v in mfp.items():
            axes2[1, 0].plot(xp, v, lw=2, label=k)
        axes2[1, 0].set_title("Entrada 3: Vulnerabilidade Psicossocial e Saúde [0 - 10]", weight="bold", fontsize=9.5)
        axes2[1, 0].legend(fontsize=7.5)
        axes2[1, 0].grid(True, linestyle=":", alpha=0.6)

        row0 = df.iloc[0]
        res0 = avaliar_estudante_individual_mamdani(
            row0["CGPA (Rendimento Acadêmico)"],
            row0["Custo Pessoal Extra-Estudo"],
            row0["Vulnerabilidade Psicossocial (0-10)"],
            row0["Semestre Atual"],
        )
        xs = res0["x_saida"]
        for k, v in res0["mf_saida"].items():
            axes2[1, 1].plot(xs, v, "--", color=CORES_RISCO[k], alpha=0.5, lw=1.3, label=k)
        axes2[1, 1].fill_between(xs, 0, res0["agregado"], color="#2563eb", alpha=0.35)
        axes2[1, 1].axvline(res0["score_fuzzy"], color="#dc2626", lw=2.3, label=f"Centroide = {res0['score_fuzzy']:.2f}")
        axes2[1, 1].set_title(f"Saída e Defuzzificação por Centroide (Exemplo: {row0['ID_Estudante']})", weight="bold", fontsize=9.5)
        axes2[1, 1].legend(fontsize=7.2, loc="upper left")
        axes2[1, 1].grid(True, linestyle=":", alpha=0.6)
        f2.tight_layout()
        p2_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        f2.savefig(p2_path, dpi=160)
        plt.close(f2)
        tmp_files.append(p2_path)

        # --- PAINEL GRÁFICO 3: Fuzzy C-Means (Espaço 2D + Diagnóstico FPC vs Entropia) ---
        f3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=160)
        feats_fcm = fcm_result["features"]
        pal_fcm = ["#dc2626", "#f59e0b", "#16a34a", "#2563eb", "#7c3aed"]
        for idx_c, nome_c in enumerate(fcm_result["nomes_clusters"]):
            sub_c = df[df["Cluster_FCM_Dominante"] == nome_c]
            ax3_1.scatter(
                sub_c[feats_fcm[0]],
                sub_c[feats_fcm[1]],
                s=22,
                alpha=0.7,
                color=pal_fcm[idx_c % len(pal_fcm)],
                label=f"{nome_c[:28]} (n={len(sub_c)})",
            )
        ax3_1.set_title(f"Agrupamento Fuzzy C-Means: {feats_fcm[0][:20]} x {feats_fcm[1][:20]}", weight="bold", fontsize=9.5)
        ax3_1.set_xlabel(feats_fcm[0])
        ax3_1.set_ylabel(feats_fcm[1])
        ax3_1.legend(fontsize=7.2)
        ax3_1.grid(True, linestyle=":", alpha=0.6)

        df_diag = fcm_result["df_diagnostico"]
        ax3_2.plot(df_diag["Clusters (C)"], df_diag["FPC (Coef. Partição Fuzzy)"], "o-", color="#2563eb", lw=2, label="FPC (Maximizar)")
        ax3_2.plot(df_diag["Clusters (C)"], df_diag["Entropia Normalizada"], "s--", color="#dc2626", lw=2, label="Entropia Norm. (Minimizar)")
        ax3_2.axvline(fcm_result["n_clusters"], color="#16a34a", linestyle=":", lw=2, label=f"C Atual = {fcm_result['n_clusters']}")
        ax3_2.set_title("Validação Matemática Fuzzy C-Means (FPC vs Entropia)", weight="bold", fontsize=9.5)
        ax3_2.set_xlabel("Número de Clusters (C)")
        ax3_2.set_ylabel("Índice [0, 1]")
        ax3_2.legend(fontsize=8)
        ax3_2.grid(True, linestyle=":", alpha=0.6)
        f3.tight_layout()
        p3_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        f3.savefig(p3_path, dpi=160)
        plt.close(f3)
        tmp_files.append(p3_path)

        # --- PAINEL GRÁFICO 4: Análise por Perfil Estudantil (Saúde, Amigos, Semestre e Moradia) ---
        f4, axes4 = plt.subplots(2, 2, figsize=(10.5, 5.0), dpi=160)
        ordem_saude = [s for s in ["Boa", "Normal", "Anormal", "Crítica"] if s in df["Condição de Saúde"].unique()]
        dados_saude = [df[df["Condição de Saúde"] == s]["Score_Depressao_Fuzzy"].values for s in ordem_saude]
        if dados_saude:
            axes4[0, 0].boxplot(dados_saude, tick_labels=ordem_saude, patch_artist=True, boxprops=dict(facecolor="#93c5fd"))
        axes4[0, 0].set_title("Score Fuzzy por Condição de Saúde", weight="bold", fontsize=9.5)
        axes4[0, 0].set_ylabel("Score Fuzzy (0-100)")
        axes4[0, 0].grid(True, linestyle=":", alpha=0.6)

        med_amigos = df.groupby("Círculo de Amigos")["Score_Depressao_Fuzzy"].mean()
        axes4[0, 1].bar(med_amigos.index, med_amigos.values, color=["#dc2626", "#16a34a"][:len(med_amigos)], edgecolor="#222")
        axes4[0, 1].set_title("Score Fuzzy Médio por Círculo de Amigos (Apoio Social)", weight="bold", fontsize=9.5)
        axes4[0, 1].set_ylabel("Score Fuzzy Médio")
        axes4[0, 1].grid(True, axis="y", linestyle=":", alpha=0.6)

        sem_m = df.groupby("Semestre Atual")["Score_Depressao_Fuzzy"].mean()
        axes4[1, 0].plot(sem_m.index, sem_m.values, "o-", color="#dc2626", lw=2)
        axes4[1, 0].set_title("Evolução do Score Fuzzy Médio por Semestre Atual", weight="bold", fontsize=9.5)
        axes4[1, 0].set_xlabel("Semestre Atual")
        axes4[1, 0].set_ylabel("Score Fuzzy Médio")
        axes4[1, 0].grid(True, linestyle=":", alpha=0.6)

        fam_m = df.groupby("Classe Familiar")["Score_Depressao_Fuzzy"].mean()
        axes4[1, 1].bar(fam_m.index, fam_m.values, color="#7c3aed", alpha=0.85, edgecolor="#222")
        axes4[1, 1].set_title("Score Fuzzy Médio por Classe Socioeconômica Familiar", weight="bold", fontsize=9.5)
        axes4[1, 1].set_ylabel("Score Fuzzy Médio")
        axes4[1, 1].grid(True, axis="y", linestyle=":", alpha=0.6)
        f4.tight_layout()
        p4_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        f4.savefig(p4_path, dpi=160)
        plt.close(f4)
        tmp_files.append(p4_path)

        # ======================================================================
        # CONSTRUÇÃO DO DOCUMENTO PDF COMPLETO (FPDF2)
        # ======================================================================
        class PDFCompleto(FPDF):
            def header(self):
                self.set_font("helvetica", "B", 8.5)
                self.set_text_color(90, 90, 90)
                self.cell(
                    0, 5,
                    _latin1_safe("FATEC - Teoria do Aprendizado Estatístico | Relatório Integral do Projeto: Lógica Fuzzy em Estudantes"),
                    align="R", new_x="LMARGIN", new_y="NEXT"
                )
                self.set_draw_color(200, 200, 200)
                self.line(10, self.get_y() + 1, 200, self.get_y() + 1)
                self.ln(4)

            def footer(self):
                self.set_y(-12)
                self.set_font("helvetica", "I", 8)
                self.set_text_color(130, 130, 130)
                self.cell(0, 8, _latin1_safe(f"Página {self.page_no()}/{{nb}} - Exportação Integral sem Omissão de Dados"), align="C")

        pdf = PDFCompleto()
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=13)

        def sec_title(texto):
            pdf.set_font("helvetica", "B", 11.5)
            pdf.set_text_color(25, 55, 95)
            pdf.cell(0, 7, _latin1_safe(texto), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(35, 35, 35)

        # ---------------- PÁGINA 1: CAPA EXECUTIVA, METODOLOGIA, KPIs & ESTATÍSTICAS DESCRITIVAS ----------------
        pdf.add_page()
        pdf.set_font("helvetica", "B", 15)
        pdf.set_text_color(25, 55, 95)
        pdf.cell(0, 8, _latin1_safe("Relatório Técnico Integral: Detecção de Depressão em Estudantes via Lógica Fuzzy"), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("helvetica", "I", 9.5)
        pdf.set_text_color(70, 70, 70)
        pdf.cell(0, 5.5, _latin1_safe("Base Kaggle: Student Depression Detection Dataset (mkhoirulmustaqim/student-depression-detection-dataset)"), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(3)

        sec_title("1. Contexto Metodológico Não Supervisionado e Criação dos Dados via Fuzzy")
        pdf.set_font("helvetica", "", 9)
        pdf.multi_cell(
            0, 4.5,
            _latin1_safe(
                "Este documento consolida 100% das informações, métricas, equações, regras nebulosas, centróides e gráficos do painel "
                "de Detecção de Depressão em Estudantes. Para atender ao requisito de aprendizado não supervisionado e trabalhar "
                "exclusivamente com a Lógica Fuzzy na criação dos dados, o rótulo original da base ('Depression') foi descartado na "
                "entrada. Todas as variáveis de risco, graus de pertinência u(x) em [0, 1], Score Contínuo Defuzzificado (0 a 100), "
                "Classificação Nebulosa e Agrupamento Difuso (Fuzzy C-Means) foram criados matematicamente a partir dos atributos "
                "observacionais dos estudantes (CGPA, Custo Pessoal Extra-Estudo, Semestre Atual, Idade, Condição de Saúde, "
                "Círculo de Amigos, Estilo de Moradia, Status de Relacionamento e Classe Familiar)."
            )
        )
        pdf.ln(2)

        sec_title("2. Indicadores Globais (KPIs) e Distribuição das Classes Criadas pela Lógica Fuzzy")
        pdf.set_font("helvetica", "", 9)
        n_alto_crit = int(df["Classificacao_Criada_Fuzzy"].isin(["Alto Risco", "Risco Crítico"]).sum())
        pdf.cell(0, 5, _latin1_safe(f"- Total de Estudantes Processados: {n_total:,} estudantes | CGPA Médio: {df['CGPA (Rendimento Acadêmico)'].mean():.2f}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, _latin1_safe(f"- Score Fuzzy Médio (Centroide 0-100): {df['Score_Depressao_Fuzzy'].mean():.2f} (Desvio Padrão: {df['Score_Depressao_Fuzzy'].std():.2f} | Mediana: {df['Score_Depressao_Fuzzy'].median():.2f})"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, _latin1_safe(f"- Estudantes em Alto Risco ou Risco Crítico (Criado via Fuzzy): {n_alto_crit:,} ({(n_alto_crit/n_total)*100:.1f}%)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.5)

        # Tabela das 4 Classes Criadas
        pdf.set_font("helvetica", "B", 8.5)
        pdf.set_fill_color(25, 55, 95)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(48, 6, _latin1_safe("Classe Criada pelo Fuzzy"), border=1, fill=True)
        pdf.cell(32, 6, _latin1_safe("Faixa do Centroide"), border=1, fill=True, align="C")
        pdf.cell(28, 6, _latin1_safe("Qtd. Alunos"), border=1, fill=True, align="C")
        pdf.cell(26, 6, _latin1_safe("% do Total"), border=1, fill=True, align="C")
        pdf.cell(28, 6, _latin1_safe("CGPA Médio"), border=1, fill=True, align="R")
        pdf.cell(28, 6, _latin1_safe("Score Médio"), border=1, fill=True, align="R", new_x="LMARGIN", new_y="NEXT")

        faixas_txt = {
            "Baixo Risco": "0.0 a 33.9",
            "Risco Moderado": "34.0 a 53.9",
            "Alto Risco": "54.0 a 73.9",
            "Risco Crítico": "74.0 a 100.0",
        }
        pdf.set_font("helvetica", "", 8.5)
        pdf.set_text_color(35, 35, 35)
        for cat in ["Baixo Risco", "Risco Moderado", "Alto Risco", "Risco Crítico"]:
            sub_c = df[df["Classificacao_Criada_Fuzzy"] == cat]
            qtd = len(sub_c)
            pct = (qtd / n_total) * 100 if n_total > 0 else 0.0
            cgpa_m = sub_c["CGPA (Rendimento Acadêmico)"].mean() if qtd > 0 else 0.0
            sc_m = sub_c["Score_Depressao_Fuzzy"].mean() if qtd > 0 else 0.0
            pdf.cell(48, 5.5, _latin1_safe(cat), border=1)
            pdf.cell(32, 5.5, _latin1_safe(faixas_txt[cat]), border=1, align="C")
            pdf.cell(28, 5.5, f"{qtd:,}", border=1, align="C")
            pdf.cell(26, 5.5, f"{pct:.1f}%", border=1, align="C")
            pdf.cell(28, 5.5, f"{cgpa_m:.2f}", border=1, align="R")
            pdf.cell(28, 5.5, f"{sc_m:.2f}", border=1, align="R", new_x="LMARGIN", new_y="NEXT")

        pdf.ln(2)
        y_img1 = pdf.get_y()
        pdf.image(p1_path, x=12, y=y_img1, w=186)
        pdf.set_y(y_img1 + 65)

        sec_title("3. Estatísticas Descritivas Completas das Variáveis Numéricas e Criadas pelo Fuzzy")
        cols_num_stat = [
            "Idade",
            "Semestre Atual",
            "CGPA (Rendimento Acadêmico)",
            "Custo Pessoal Extra-Estudo",
            "Vulnerabilidade Psicossocial (0-10)",
            "Score_Depressao_Fuzzy",
            "Pertinencia_FCM_Maxima",
            "Incerteza_Fuzzy_Shannon",
        ]
        cols_num_exist = [c for c in cols_num_stat if c in df.columns]
        pdf.set_font("helvetica", "B", 7.8)
        pdf.set_fill_color(25, 55, 95)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(54, 5.8, _latin1_safe("Variável Analisada"), border=1, fill=True)
        pdf.cell(22, 5.8, _latin1_safe("Média"), border=1, fill=True, align="R")
        pdf.cell(22, 5.8, _latin1_safe("Desv.Pad"), border=1, fill=True, align="R")
        pdf.cell(22, 5.8, _latin1_safe("Mínimo"), border=1, fill=True, align="R")
        pdf.cell(23, 5.8, _latin1_safe("Quartil 25%"), border=1, fill=True, align="R")
        pdf.cell(23, 5.8, _latin1_safe("Mediana"), border=1, fill=True, align="R")
        pdf.cell(24, 5.8, _latin1_safe("Máximo"), border=1, fill=True, align="R", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 7.8)
        pdf.set_text_color(35, 35, 35)
        for col_n in cols_num_exist:
            serie = df[col_n].dropna()
            pdf.cell(54, 5.2, _latin1_safe(col_n[:32]), border=1)
            pdf.cell(22, 5.2, f"{serie.mean():,.2f}", border=1, align="R")
            pdf.cell(22, 5.2, f"{serie.std():,.2f}", border=1, align="R")
            pdf.cell(22, 5.2, f"{serie.min():,.2f}", border=1, align="R")
            pdf.cell(23, 5.2, f"{serie.quantile(0.25):,.2f}", border=1, align="R")
            pdf.cell(23, 5.2, f"{serie.median():,.2f}", border=1, align="R")
            pdf.cell(24, 5.2, f"{serie.max():,.2f}", border=1, align="R", new_x="LMARGIN", new_y="NEXT")

        # ---------------- PÁGINA 2: SISTEMA DE INFERÊNCIA DE MAMDANI, FUNÇÕES DE PERTINÊNCIA & 8 REGRAS ----------------
        pdf.add_page()
        sec_title("4. Sistema de Inferência Fuzzy de Mamdani: Funções de Pertinência e Regras")
        pdf.set_font("helvetica", "", 8.8)
        pdf.multi_cell(
            0, 4.3,
            _latin1_safe(
                "Abaixo estão especificadas todas as funções de pertinência (Trapezoidais [a, b, c, d] e Triangulares [a, b, c]) "
                "utilizadas na etapa de Fuzzificação e as 8 Regras Especialistas SE-ENTÃO avaliadas com operadores de Zadeh "
                "(T-norma MIN para conjunção 'E'; S-norma MAX para disjunção 'OU'), seguidas pela Defuzzificação via Centroide: "
                "y* = Soma(y * u_agregado(y)) / Soma(u_agregado(y))."
            )
        )
        pdf.ln(1.5)

        # Tabela de Parâmetros das Funções de Pertinência
        params_mf = [
            ("Entrada 1: CGPA [2.0 - 4.0]", "Baixo (Crítico)", "Trapezoidal", "[2.00, 2.00, 2.85, 3.25]"),
            ("Entrada 1: CGPA [2.0 - 4.0]", "Médio (Regular)", "Triangular", "[2.95, 3.35, 3.70]"),
            ("Entrada 1: CGPA [2.0 - 4.0]", "Alto (Excelente)", "Trapezoidal", "[3.50, 3.75, 4.00, 4.00]"),
            ("Entrada 2: Custo [0 - 32.000]", "Baixo / Moderado / Elevado", "Trap / Tri / Trap", "[0..9.5k] / [6.5k..17.5k] / [14k..32k]"),
            ("Entrada 3: Vuln. Psico [0 - 10]", "Baixa / Moderada / Severa", "Trap / Tri / Trap", "[0..4.5] / [3.2..7.5] / [6.2..10.0]"),
            ("Entrada 4: Semestre [0 - 13]", "Inicial / Interm. / Avançado", "Trap / Tri / Trap", "[0..4.5] / [3.0..8.0] / [6.5..13.0]"),
            ("Saída: Score Fuzzy [0 - 100]", "Baixo / Mod. / Alto / Crítico", "Trap / Tri / Tri / Trap", "[0..34] / [22..65] / [52..88] / [76..100]"),
        ]
        pdf.set_font("helvetica", "B", 7.8)
        pdf.set_fill_color(25, 55, 95)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(52, 5.5, _latin1_safe("Variável / Universo"), border=1, fill=True)
        pdf.cell(48, 5.5, _latin1_safe("Conjuntos Nebulosos"), border=1, fill=True)
        pdf.cell(32, 5.5, _latin1_safe("Tipo de Curva"), border=1, fill=True, align="C")
        pdf.cell(58, 5.5, _latin1_safe("Parâmetros Matemáticos"), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 7.6)
        pdf.set_text_color(35, 35, 35)
        for v_u, conj_n, tipo_c, par_m in params_mf:
            pdf.cell(52, 5, _latin1_safe(v_u), border=1)
            pdf.cell(48, 5, _latin1_safe(conj_n), border=1)
            pdf.cell(32, 5, _latin1_safe(tipo_c), border=1, align="C")
            pdf.cell(58, 5, _latin1_safe(par_m), border=1, new_x="LMARGIN", new_y="NEXT")

        pdf.ln(2)
        # Tabela das 8 Regras Nebulosas com Ativação Média na Base
        pdf.set_font("helvetica", "B", 8)
        pdf.set_fill_color(25, 55, 95)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(12, 5.5, "Cod", border=1, fill=True, align="C")
        pdf.cell(108, 5.5, _latin1_safe("Antecedente da Regra Fuzzy (SE ...)"), border=1, fill=True)
        pdf.cell(32, 5.5, _latin1_safe("Consequente"), border=1, fill=True, align="C")
        pdf.cell(20, 5.5, _latin1_safe("Alfa Médio"), border=1, fill=True, align="R")
        pdf.cell(18, 5.5, _latin1_safe("Dom. (n)"), border=1, fill=True, align="R", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 7.5)
        pdf.set_text_color(35, 35, 35)
        for idx_r, (cod_r, ant_r, cons_r) in enumerate(DESCRICAO_REGRAS_MAMDANI):
            col_alfa = f"Regra_R{idx_r+1}_Alfa"
            alfa_med = df[col_alfa].mean() if col_alfa in df.columns else 0.0
            n_dom = int(df["Regra_Fuzzy_Dominante"].str.startswith(f"{cod_r} ").sum())
            pdf.cell(12, 5, cod_r, border=1, align="C")
            pdf.cell(108, 5, _latin1_safe(ant_r[:68]), border=1)
            pdf.cell(32, 5, _latin1_safe(cons_r), border=1, align="C")
            pdf.cell(20, 5, f"{alfa_med:.3f}", border=1, align="R")
            pdf.cell(18, 5, f"{n_dom}", border=1, align="R", new_x="LMARGIN", new_y="NEXT")

        pdf.ln(2)
        y_img2 = pdf.get_y()
        pdf.image(p2_path, x=12, y=y_img2, w=186)

        # ---------------- PÁGINA 3: AGRUPAMENTO DIFUSO FUZZY C-MEANS (FCM) COMPLETO ----------------
        pdf.add_page()
        sec_title("5. Agrupamento Não Supervisionado Fuzzy C-Means (FCM) - Métricas e Centróides")
        pdf.set_font("helvetica", "", 8.8)
        pdf.multi_cell(
            0, 4.3,
            _latin1_safe(
                f"O algoritmo Fuzzy C-Means (Bezdek) foi executado com C = {fcm_result['n_clusters']} clusters, expoente de "
                f"nebulosidade m = {fcm_result['m']} e limiar de transição tau = {fcm_result['limiar_incerteza']}. "
                f"Resultados globais: FPC (Fuzzy Partition Coefficient) = {fcm_result['fpc']:.4f} | Entropia da Partição (PE) = "
                f"{fcm_result['pe']:.4f} (Entropia Normalizada = {fcm_result['pe_norm']:.4f}) | Estudantes em Fronteira Nebulosa "
                f"(Zona de Transição entre Perfis): {fcm_result['n_fronteira']:,} estudantes ({fcm_result['pct_fronteira']:.1f}%)."
            )
        )
        pdf.ln(2)

        # Tabela Resumo dos Centróides Fuzzy C-Means
        res_fcm_df = fcm_result["resumo_clusters"]
        pdf.set_font("helvetica", "B", 7.8)
        pdf.set_fill_color(25, 55, 95)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(56, 5.8, _latin1_safe("Perfil Fuzzy C-Means"), border=1, fill=True)
        pdf.cell(20, 5.8, _latin1_safe("Alunos"), border=1, fill=True, align="C")
        pdf.cell(20, 5.8, _latin1_safe("Pert.Méd"), border=1, fill=True, align="C")
        pdf.cell(22, 5.8, _latin1_safe("Transição"), border=1, fill=True, align="C")
        pdf.cell(24, 5.8, _latin1_safe("Vuln.Psico"), border=1, fill=True, align="R")
        pdf.cell(24, 5.8, _latin1_safe("CGPA Centr."), border=1, fill=True, align="R")
        pdf.cell(24, 5.8, _latin1_safe("Score Fuzzy"), border=1, fill=True, align="R", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 7.8)
        pdf.set_text_color(35, 35, 35)
        for _, r_row in res_fcm_df.iterrows():
            v_psico_c = r_row.get("Centróide (Vulnerabilidade Psicossocial (0-10))", 0.0)
            v_cgpa_c = r_row.get("Centróide (CGPA (Rendimento Acadêmico))", 0.0)
            v_sc_c = r_row.get("Score Mamdani Médio", 0.0)
            pdf.cell(56, 5.4, _latin1_safe(str(r_row["Perfil Fuzzy C-Means"])[:33]), border=1)
            pdf.cell(20, 5.4, f"{r_row['Qtd Estudantes (Dominante)']:,}", border=1, align="C")
            pdf.cell(20, 5.4, _latin1_safe(r_row["Pertinência Média Global (%)"]), border=1, align="C")
            pdf.cell(22, 5.4, f"{r_row['Em Zona de Transição']:,}", border=1, align="C")
            pdf.cell(24, 5.4, f"{v_psico_c:.2f}", border=1, align="R")
            pdf.cell(24, 5.4, f"{v_cgpa_c:.2f}", border=1, align="R")
            pdf.cell(24, 5.4, f"{v_sc_c:.2f}", border=1, align="R", new_x="LMARGIN", new_y="NEXT")

        pdf.ln(2)
        # Tabela de Diagnóstico C = 2..6
        pdf.set_font("helvetica", "B", 8)
        pdf.cell(0, 5.5, _latin1_safe("Tabela de Diagnóstico Matemático por Quantidade de Clusters (C = 2 a 6):"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_fill_color(235, 240, 248)
        for col_d in fcm_result["df_diagnostico"].columns:
            pdf.cell(47, 5.2, _latin1_safe(col_d), border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_font("helvetica", "", 7.8)
        for _, d_row in fcm_result["df_diagnostico"].iterrows():
            pdf.cell(47, 5, f"C = {int(d_row['Clusters (C)'])}", border=1, align="C")
            pdf.cell(47, 5, f"{d_row['FPC (Coef. Partição Fuzzy)']:.4f}", border=1, align="C")
            pdf.cell(47, 5, f"{d_row['Entropia da Partição (PE)']:.4f}", border=1, align="C")
            pdf.cell(47, 5, f"{d_row['Entropia Normalizada']:.4f}", border=1, align="C", new_x="LMARGIN", new_y="NEXT")

        pdf.ln(2)
        y_img3 = pdf.get_y()
        pdf.image(p3_path, x=12, y=y_img3, w=186)
        pdf.set_y(y_img3 + 66)

        # Tabela dos Top 10 Estudantes em Fronteira Nebulosa (Maior Hibridismo / Incerteza de Shannon)
        pdf.set_font("helvetica", "B", 8.5)
        pdf.set_text_color(25, 55, 95)
        pdf.cell(0, 5.5, _latin1_safe("Top 10 Estudantes em Fronteira Nebulosa (Maior Incerteza de Transição entre Perfis):"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "B", 7.5)
        pdf.set_fill_color(25, 55, 95)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(24, 5.2, "ID", border=1, fill=True, align="C")
        pdf.cell(52, 5.2, _latin1_safe("Cluster Dominante"), border=1, fill=True)
        pdf.cell(24, 5.2, _latin1_safe("Pert. C0 (%)"), border=1, fill=True, align="R")
        pdf.cell(24, 5.2, _latin1_safe("Pert. C1 (%)"), border=1, fill=True, align="R")
        pdf.cell(24, 5.2, _latin1_safe("Pert. C2 (%)"), border=1, fill=True, align="R")
        pdf.cell(22, 5.2, _latin1_safe("Entropia"), border=1, fill=True, align="R")
        pdf.cell(20, 5.2, _latin1_safe("Score"), border=1, fill=True, align="R", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 7.5)
        pdf.set_text_color(35, 35, 35)
        top_front = df.sort_values(by="Incerteza_Fuzzy_Shannon", ascending=False).head(10)
        for _, f_row in top_front.iterrows():
            c0_p = f_row.get("Pertinencia_FCM_C0", 0.0) * 100
            c1_p = f_row.get("Pertinencia_FCM_C1", 0.0) * 100
            c2_p = f_row.get("Pertinencia_FCM_C2", 0.0) * 100
            pdf.cell(24, 4.8, _latin1_safe(f_row["ID_Estudante"]), border=1, align="C")
            pdf.cell(52, 4.8, _latin1_safe(str(f_row["Cluster_FCM_Dominante"])[:30]), border=1)
            pdf.cell(24, 4.8, f"{c0_p:.1f}%", border=1, align="R")
            pdf.cell(24, 4.8, f"{c1_p:.1f}%", border=1, align="R")
            pdf.cell(24, 4.8, f"{c2_p:.1f}%", border=1, align="R")
            pdf.cell(22, 4.8, f"{f_row['Incerteza_Fuzzy_Shannon']:.4f}", border=1, align="R")
            pdf.cell(20, 4.8, f"{f_row['Score_Depressao_Fuzzy']:.2f}", border=1, align="R", new_x="LMARGIN", new_y="NEXT")

        # ---------------- PÁGINA 4: ANÁLISE CRUZADA POR PERFIL ESTUDANTIL & RANKING CRÍTICO ----------------
        pdf.add_page()
        sec_title("6. Análise Cruzada dos Resultados Fuzzy por Perfil Estudantil (Saúde, Apoio, Moradia e Família)")

        def add_group_summary_table(col_agrup, titulo_t):
            pdf.set_font("helvetica", "B", 8.2)
            pdf.set_text_color(25, 55, 95)
            pdf.cell(0, 5.2, _latin1_safe(titulo_t), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "B", 7.5)
            pdf.set_fill_color(25, 55, 95)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(54, 5, _latin1_safe(col_agrup), border=1, fill=True)
            pdf.cell(22, 5, _latin1_safe("Alunos"), border=1, fill=True, align="C")
            pdf.cell(26, 5, _latin1_safe("Score Médio"), border=1, fill=True, align="R")
            pdf.cell(26, 5, _latin1_safe("CGPA Médio"), border=1, fill=True, align="R")
            pdf.cell(30, 5, _latin1_safe("Custo Médio"), border=1, fill=True, align="R")
            pdf.cell(32, 5, _latin1_safe("% Alto/Crítico"), border=1, fill=True, align="R", new_x="LMARGIN", new_y="NEXT")

            pdf.set_font("helvetica", "", 7.5)
            pdf.set_text_color(35, 35, 35)
            for grp_val, sub_g in df.groupby(col_agrup):
                pct_ac = sub_g["Classificacao_Criada_Fuzzy"].isin(["Alto Risco", "Risco Crítico"]).mean() * 100
                pdf.cell(54, 4.8, _latin1_safe(str(grp_val)[:32]), border=1)
                pdf.cell(22, 4.8, f"{len(sub_g):,}", border=1, align="C")
                pdf.cell(26, 4.8, f"{sub_g['Score_Depressao_Fuzzy'].mean():.2f}", border=1, align="R")
                pdf.cell(26, 4.8, f"{sub_g['CGPA (Rendimento Acadêmico)'].mean():.2f}", border=1, align="R")
                pdf.cell(30, 4.8, f"{sub_g['Custo Pessoal Extra-Estudo'].mean():,.0f}", border=1, align="R")
                pdf.cell(32, 4.8, f"{pct_ac:.1f}%", border=1, align="R", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)

        add_group_summary_table("Condição de Saúde", "a) Desempenho e Score Fuzzy por Condição de Saúde:")
        add_group_summary_table("Círculo de Amigos", "b) Desempenho e Score Fuzzy por Círculo de Amigos (Apoio Social):")
        add_group_summary_table("Estilo de Moradia", "c) Desempenho e Score Fuzzy por Estilo de Moradia:")
        add_group_summary_table("Classe Familiar", "d) Desempenho e Score Fuzzy por Classe Socioeconômica Familiar:")

        y_img4 = pdf.get_y() + 1
        pdf.image(p4_path, x=12, y=y_img4, w=186)
        pdf.set_y(y_img4 + 88)

        sec_title("7. Top 12 Estudantes Prioritários Identificados com Maior Risco Crítico Fuzzy")
        pdf.set_font("helvetica", "B", 7.4)
        pdf.set_fill_color(25, 55, 95)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(20, 5, "ID", border=1, fill=True, align="C")
        pdf.cell(20, 5, _latin1_safe("Gênero"), border=1, fill=True, align="C")
        pdf.cell(16, 5, "CGPA", border=1, fill=True, align="R")
        pdf.cell(22, 5, "Custo", border=1, fill=True, align="R")
        pdf.cell(22, 5, _latin1_safe("Saúde"), border=1, fill=True, align="C")
        pdf.cell(36, 5, _latin1_safe("Círculo de Amigos"), border=1, fill=True, align="C")
        pdf.cell(26, 5, _latin1_safe("Regra Dom."), border=1, fill=True, align="C")
        pdf.cell(28, 5, _latin1_safe("Score Fuzzy"), border=1, fill=True, align="R", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 7.3)
        pdf.set_text_color(35, 35, 35)
        top_crit = df.sort_values(by=["Score_Depressao_Fuzzy", "Vulnerabilidade Psicossocial (0-10)"], ascending=False).head(12)
        for _, c_row in top_crit.iterrows():
            pdf.cell(20, 4.7, _latin1_safe(c_row["ID_Estudante"]), border=1, align="C")
            pdf.cell(20, 4.7, _latin1_safe(c_row["Gênero"]), border=1, align="C")
            pdf.cell(16, 4.7, f"{c_row['CGPA (Rendimento Acadêmico)']:.2f}", border=1, align="R")
            pdf.cell(22, 4.7, f"{c_row['Custo Pessoal Extra-Estudo']:,.0f}", border=1, align="R")
            pdf.cell(22, 4.7, _latin1_safe(c_row["Condição de Saúde"]), border=1, align="C")
            pdf.cell(36, 4.7, _latin1_safe(str(c_row["Círculo de Amigos"])[:20]), border=1, align="C")
            pdf.cell(26, 4.7, _latin1_safe(c_row["Regra_Fuzzy_Dominante"]), border=1, align="C")
            pdf.cell(28, 4.7, f"{c_row['Score_Depressao_Fuzzy']:.2f} ({_latin1_safe(c_row['Classificacao_Criada_Fuzzy'])[:7]})", border=1, align="R", new_x="LMARGIN", new_y="NEXT")

        pdf_bytes = bytes(pdf.output())

        for tf in tmp_files:
            try:
                os.remove(tf)
            except Exception:
                pass

        return pdf_bytes
    except Exception:
        return None
