import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


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

    # Remover coluna supervisionada original para garantir tratamento 100% não supervisionado
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

    # Construção de indicador contínuo de Vulnerabilidade Psicossocial & Saúde [0 a 10]
    # baseado exclusivamente nos fatores observacionais não supervisionados do estudante
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
    em estudantes via Lógica Fuzzy:
      1. CGPA (Rendimento Acadêmico): [2.00 a 4.00] -> Baixo, Médio, Alto
      2. Custo Pessoal Extra-Estudo: [0 a 32.000] -> Baixo, Moderado, Elevado
      3. Vulnerabilidade Psicossocial & Saúde: [0 a 10] -> Baixa, Moderada, Severa
      4. Semestre Atual (Desgaste Acadêmico): [0 a 13] -> Inicial, Intermediário, Avançado
      5. Saída (Grau de Depressão Fuzzy): [0 a 100] -> Baixo Risco, Risco Moderado, Alto Risco, Risco Crítico
    """
    # 1. CGPA (2.0 a 4.0)
    x_cgpa = np.linspace(2.0, 4.0, 400)
    mf_cgpa = {
        "Baixo (Crítico)": trapmf(x_cgpa, 2.0, 2.0, 2.85, 3.25),
        "Médio (Regular)": trimf(x_cgpa, 2.95, 3.35, 3.70),
        "Alto (Excelente)": trapmf(x_cgpa, 3.50, 3.75, 4.0, 4.0),
    }

    # 2. Custo Pessoal Extra-Estudo (0 a 32.000)
    x_custo = np.linspace(0, 32000, 400)
    mf_custo = {
        "Baixo": trapmf(x_custo, 0, 0, 5000, 9500),
        "Moderado": trimf(x_custo, 6500, 12000, 17500),
        "Elevado": trapmf(x_custo, 14000, 19500, 32000, 32000),
    }

    # 3. Vulnerabilidade Psicossocial & Saúde (0 a 10)
    x_psico = np.linspace(0, 10, 400)
    mf_psico = {
        "Baixa": trapmf(x_psico, 0, 0, 2.5, 4.5),
        "Moderada": trimf(x_psico, 3.2, 5.5, 7.5),
        "Severa": trapmf(x_psico, 6.2, 7.8, 10.0, 10.0),
    }

    # 4. Semestre Atual (0 a 13)
    x_sem = np.linspace(0, 13, 400)
    mf_sem = {
        "Inicial": trapmf(x_sem, 0, 0, 2.5, 4.5),
        "Intermediário": trimf(x_sem, 3.0, 5.5, 8.0),
        "Avançado / Final": trapmf(x_sem, 6.5, 8.5, 13.0, 13.0),
    }

    # 5. Saída: Score Fuzzy de Depressão em Estudantes [0 a 100]
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


# ==============================================================================
# 4. CRIAÇÃO DOS DADOS VIA INFERÊNCIA FUZZY DE MAMDANI (VETORIZADA E INDIVIDUAL)
# ==============================================================================
def criar_dados_fuzzy_estudantes(df_base):
    """
    Aplica a Lógica Fuzzy (Fuzzificação + 8 Regras de Mamdani + Defuzzificação por Centroide)
    para CRIAR os dados de detecção de depressão de forma 100% não supervisionada.
    Gera novas colunas com os graus de pertinência nebulosos, o Score Fuzzy (0-100),
    a regra dominante e a classificação Fuzzy gerada.
    """
    df = df_base.copy()

    cgpa = np.clip(df["CGPA (Rendimento Acadêmico)"].values.astype(float), 2.0, 4.0)
    custo = np.clip(df["Custo Pessoal Extra-Estudo"].values.astype(float), 0.0, 32000.0)
    psico = np.clip(df["Vulnerabilidade Psicossocial (0-10)"].values.astype(float), 0.0, 10.0)
    sem = np.clip(df["Semestre Atual"].values.astype(float), 0.0, 13.0)

    # 1. Fuzzificação vetorizada (graus de pertinência das entradas)
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

    # 2. Avaliação das 8 Regras Fuzzy (Operadores de Zadeh: MIN = E, MAX = OU)
    # R1: SE Vulnerabilidade Psicossocial é Severa E (CGPA é Baixo OU Custo é Elevado) -> Risco Crítico
    r1 = np.minimum(psico_sev, np.maximum(cgpa_baixo, custo_elev))
    # R2: SE Vulnerabilidade Psicossocial é Severa E Semestre é Avançado -> Risco Crítico
    r2 = np.minimum(psico_sev, sem_avanc)
    # R3: SE Vulnerabilidade Psicossocial é Severa OU (CGPA é Baixo E Custo é Elevado) -> Alto Risco
    r3 = np.maximum(psico_sev, np.minimum(cgpa_baixo, custo_elev))
    # R4: SE Vulnerabilidade Psicossocial é Moderada E (CGPA é Baixo OU Semestre é Avançado) -> Alto Risco
    r4 = np.minimum(psico_mod, np.maximum(cgpa_baixo, sem_avanc))
    # R5: SE Vulnerabilidade Psicossocial é Moderada E CGPA é Médio -> Risco Moderado
    r5 = np.minimum(psico_mod, cgpa_medio)
    # R6: SE Custo é Elevado E CGPA é Médio E Semestre é Intermediário -> Risco Moderado
    r6 = np.minimum(np.minimum(custo_elev, cgpa_medio), sem_int)
    # R7: SE Vulnerabilidade Psicossocial é Baixa E CGPA é Médio -> Baixo Risco
    r7 = np.minimum(psico_baixa, cgpa_medio)
    # R8: SE Vulnerabilidade Psicossocial é Baixa E CGPA é Alto E Custo é Baixo/Moderado -> Baixo Risco
    r8 = np.minimum(np.minimum(psico_baixa, cgpa_alto), np.maximum(custo_baixo, custo_mod))

    # Força de ativação para cada conjunto de saída (Agregação MAX por consequente)
    alfa_baixo = np.maximum(r7, r8)          # (N,)
    alfa_moderado = np.maximum(r5, r6)       # (N,)
    alfa_alto = np.maximum(r3, r4)           # (N,)
    alfa_critico = np.maximum(r1, r2)        # (N,)

    # 3. Agregação de Mamdani e Defuzzificação por Centroide (Vetorizada N x 400)
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

    # Normalizar ativações de saída para representar graus de pertinência relativos
    soma_alfas = np.fmax(alfa_baixo + alfa_moderado + alfa_alto + alfa_critico, 1e-9)
    df["Pert_Fuzzy_Baixo_Risco"] = np.round(alfa_baixo / soma_alfas, 4)
    df["Pert_Fuzzy_Risco_Moderado"] = np.round(alfa_moderado / soma_alfas, 4)
    df["Pert_Fuzzy_Alto_Risco"] = np.round(alfa_alto / soma_alfas, 4)
    df["Pert_Fuzzy_Risco_Critico"] = np.round(alfa_critico / soma_alfas, 4)

    df["Score_Depressao_Fuzzy"] = np.round(scores_centroide, 2)

    # Rótulo categórico criado exclusivamente pelo Centroide Fuzzy
    conds = [
        df["Score_Depressao_Fuzzy"] >= 74.0,
        df["Score_Depressao_Fuzzy"] >= 54.0,
        df["Score_Depressao_Fuzzy"] >= 34.0,
    ]
    choices = ["Risco Crítico", "Alto Risco", "Risco Moderado"]
    df["Classificacao_Criada_Fuzzy"] = np.select(conds, choices, default="Baixo Risco")

    # Identificar a regra dominante de cada estudante
    matriz_regras = np.vstack([r1, r2, r3, r4, r5, r6, r7, r8]).T  # (N, 8)
    idx_regra = np.argmax(matriz_regras, axis=1)
    val_regra = np.max(matriz_regras, axis=1)
    df["Regra_Fuzzy_Dominante"] = [f"R{idx + 1} (α={val:.2f})" for idx, val in zip(idx_regra, val_regra)]

    return df


def avaliar_estudante_individual_mamdani(cgpa_val, custo_val, psico_val, sem_val):
    """
    Executa a inferência Fuzzy passo a passo para um único estudante (para auditoria visual
    das funções de pertinência, regras ativadas e gráfico do Centroide).
    """
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

    regras_detalhe = [
        {"Regra": "R1", "Antecedente (SE)": "Vulnerabilidade é Severa E (CGPA Baixo OU Custo Elevado)", "Consequente (ENTÃO)": "Risco Crítico", "Ativação (α)": r1},
        {"Regra": "R2", "Antecedente (SE)": "Vulnerabilidade é Severa E Semestre é Avançado/Final", "Consequente (ENTÃO)": "Risco Crítico", "Ativação (α)": r2},
        {"Regra": "R3", "Antecedente (SE)": "Vulnerabilidade é Severa OU (CGPA Baixo E Custo Elevado)", "Consequente (ENTÃO)": "Alto Risco", "Ativação (α)": r3},
        {"Regra": "R4", "Antecedente (SE)": "Vulnerabilidade é Moderada E (CGPA Baixo OU Semestre Avançado)", "Consequente (ENTÃO)": "Alto Risco", "Ativação (α)": r4},
        {"Regra": "R5", "Antecedente (SE)": "Vulnerabilidade é Moderada E CGPA é Médio", "Consequente (ENTÃO)": "Risco Moderado", "Ativação (α)": r5},
        {"Regra": "R6", "Antecedente (SE)": "Custo é Elevado E CGPA é Médio E Semestre Intermediário", "Consequente (ENTÃO)": "Risco Moderado", "Ativação (α)": r6},
        {"Regra": "R7", "Antecedente (SE)": "Vulnerabilidade é Baixa E CGPA é Médio", "Consequente (ENTÃO)": "Baixo Risco", "Ativação (α)": r7},
        {"Regra": "R8", "Antecedente (SE)": "Vulnerabilidade é Baixa E CGPA é Alto E Custo Baixo/Mod.", "Consequente (ENTÃO)": "Baixo Risco", "Ativação (α)": r8},
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
    criando os graus de pertinência nebulosos U_ij a cada perfil estudantil.
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

    # Ordenar os clusters pelo maior Score_Depressao_Fuzzy médio ponderado
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
    centers_orig = scaler.inverse_transform(centers_scaled)

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

    # Resumo ponderado dos clusters
    resumo_list = []
    for j in range(n_clusters):
        sub_dom = df_clean[df_clean["Cluster_FCM_ID"] == j]
        pesos = U[:, j] ** m
        soma_p = np.sum(pesos)
        item = {
            "Perfil Fuzzy C-Means": nomes_clusters[j],
            "Qtd Estudantes (Dominante)": len(sub_dom),
            "Pertinência Média Global (%)": f"{np.mean(U[:, j]) * 100:.1f}%",
            "Em Zona de Transição": int(sub_dom["Estudante_Fronteira_Fuzzy"].sum()),
        }
        for idx_f, f in enumerate(features):
            item[f"Centróide ({f})"] = float(np.sum(pesos * X[:, idx_f]) / soma_p) if soma_p > 0 else 0.0
        if "Score_Depressao_Fuzzy" in df_clean.columns:
            item["Score Mamdani Médio"] = float(np.sum(pesos * df_clean["Score_Depressao_Fuzzy"].values) / soma_p)
        resumo_list.append(item)
    resumo_df = pd.DataFrame(resumo_list)

    # Diagnóstico rápido para C = 2..6
    diag_c = []
    for c_test in range(2, 7):
        _, _, _, fpc_t, pe_t, pe_norm_t = _calcular_fcm_core(X_scaled, n_clusters=c_test, m=m, max_iter=60, seed=seed)
        diag_c.append({
            "Clusters (C)": c_test,
            "FPC (Coef. Partição Fuzzy)": fpc_t,
            "Entropia da Partição (PE)": pe_t,
            "Entropia Normalizada": pe_norm_t,
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


def generate_student_fuzzy_pdf_report(df_fuzzy, fcm_result=None):
    """Gera o Relatório Executivo em PDF da criação de dados Fuzzy para Detecção de Depressão em Estudantes."""
    try:
        from fpdf import FPDF
        import unicodedata

        def _ascii(texto):
            return unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 9, _ascii("Relatorio Executivo: Deteccao de Depressao em Estudantes via Logica Fuzzy"), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("helvetica", "I", 9.5)
        pdf.cell(0, 6, _ascii("Base Kaggle (Nao Supervisionada): Student Depression Detection Dataset"), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(5)

        pdf.set_font("helvetica", "B", 11.5)
        pdf.cell(0, 7, _ascii("1. Sintese dos Dados Criados pelo Sistema de Inferencia Fuzzy (Mamdani)"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 5.5, _ascii(f"- Total de Estudantes Processados (Sem Rotulo Supervisionado): {len(df_fuzzy)}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5.5, _ascii(f"- Score Medio de Depressao Fuzzy (Centroide 0-100): {df_fuzzy['Score_Depressao_Fuzzy'].mean():.2f}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5.5, _ascii(f"- CGPA Medio: {df_fuzzy['CGPA (Rendimento Acadêmico)'].mean():.2f} | Custo Pessoal Medio: {df_fuzzy['Custo Pessoal Extra-Estudo'].mean():,.0f}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        contagem = df_fuzzy["Classificacao_Criada_Fuzzy"].value_counts()
        for cat in ["Risco Crítico", "Alto Risco", "Risco Moderado", "Baixo Risco"]:
            qtd = int(contagem.get(cat, 0))
            pct = (qtd / len(df_fuzzy)) * 100 if len(df_fuzzy) > 0 else 0
            pdf.cell(0, 5.5, _ascii(f"  * {cat}: {qtd} estudantes ({pct:.1f}%)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        if fcm_result and fcm_result.get("sucesso"):
            pdf.set_font("helvetica", "B", 11.5)
            pdf.cell(0, 7, _ascii("2. Agrupamento Difuso Nao Supervisionado (Fuzzy C-Means)"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 10)
            pdf.cell(0, 5.5, _ascii(f"- Clusters (C): {fcm_result['n_clusters']} | Expoente de Fuzzificacao (m): {fcm_result['m']}"), new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 5.5, _ascii(f"- Coeficiente de Particao Fuzzy (FPC): {fcm_result['fpc']:.4f} | Entropia Norm.: {fcm_result['pe_norm']:.4f}"), new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 5.5, _ascii(f"- Estudantes em Fronteira Nebulosa (Zona de Transicao): {fcm_result['n_fronteira']} ({fcm_result['pct_fronteira']:.1f}%)"), new_x="LMARGIN", new_y="NEXT")

        return bytes(pdf.output())
    except Exception:
        return None
