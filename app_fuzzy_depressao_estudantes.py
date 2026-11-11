import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
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
        DESCRICAO_REGRAS_MAMDANI,
    )
except ImportError:
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

    try:
        import lzma
        return pd.read_csv(BytesIO(lzma.decompress(base64.b64decode('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')))), 'Kaggle Dataset Sincronizado'
    except Exception:
        return None, 'Erro'


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
# Barra Lateral: Filtros Globais & Hiperparâmetros
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

# =========================================================
# Execução Antecipada do Fuzzy C-Means para que o CSV e o PDF
# na Barra Lateral já contenham 100% dos Dados do Projeto!
# =========================================================
DEFAULT_FCM_FEATURES = [
    "Vulnerabilidade Psicossocial (0-10)",
    "CGPA (Rendimento Acadêmico)",
    "Custo Pessoal Extra-Estudo",
    "Semestre Atual",
]
fcm_feats_ativas = st.session_state.get("fcm_feats_sel", DEFAULT_FCM_FEATURES)
if len(fcm_feats_ativas) < 2:
    fcm_feats_ativas = DEFAULT_FCM_FEATURES
fcm_c_ativo = st.session_state.get("fcm_c_sel", 3)
fcm_m_ativo = st.session_state.get("fcm_m_sel", 2.0)
fcm_tau_ativo = st.session_state.get("fcm_tau_sel", 0.55)

res_fcm_global = executar_fuzzy_cmeans_estudantes(
    df_filtrado,
    features=fcm_feats_ativas,
    n_clusters=fcm_c_ativo,
    m=fcm_m_ativo,
    limiar_incerteza=fcm_tau_ativo,
    seed=42,
)
st.session_state["fcm_estudantes"] = res_fcm_global

# DataFrame consolidado com TODAS as colunas do projeto (Originais + Mamdani + Regras + Fuzzy C-Means)
df_exportacao_completa = res_fcm_global["df_resultado"] if res_fcm_global.get("sucesso") else df_filtrado

st.sidebar.divider()
st.sidebar.subheader("📥 Exportar Dados & Relatório")
st.sidebar.caption(
    f"✅ Exportação Integral: **{len(df_exportacao_completa):,} registros** × **{len(df_exportacao_completa.columns)} colunas** "
    "(Atributos + Fuzzificação + 8 Regras + Mamdani + Fuzzy C-Means)."
)

# utf-8-sig garante abertura perfeita com acentos tanto no Excel quanto em editores de texto
csv_criado_sidebar = df_exportacao_completa.to_csv(index=False).encode("utf-8-sig")
st.sidebar.download_button(
    "💾 Baixar Dados Criados via Fuzzy (CSV)",
    data=csv_criado_sidebar,
    file_name="Student_Depression_Fuzzy_Completo.csv",
    mime="text/csv",
)

with st.sidebar:
    with st.spinner("Gerando PDF Integral com Tabelas e Gráficos..."):
        pdf_bytes = generate_student_fuzzy_pdf_report(df_exportacao_completa, res_fcm_global)

if pdf_bytes:
    st.sidebar.download_button(
        "📄 Baixar Relatório Executivo (PDF)",
        data=pdf_bytes,
        file_name="Relatorio_Integral_Fuzzy_Depressao_Estudantes.pdf",
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
        **⚙️ Como os dados foram criados usando a Lógica Fuzzy (Exportação Integral):**
        1. **Entrada Não Supervisionada (Sem Rótulo):** Tomamos apenas os atributos observacionais dos **4.464 estudantes** (`CGPA`, `Custo Pessoal Extra-Estudo`, `Semestre Atual`, `Condição de Saúde`, `Círculo de Amigos`, `Estilo de Moradia`, `Status de Relacionamento` e `Classe Familiar`).
        2. **Fuzzificação Contínua (12 Colunas $\\mu$):** Cada estudante recebe graus de pertinência $\\mu \\in [0, 1]$ em todos os conjuntos nebulosos de entrada (`Mu_CGPA_*`, `Mu_Custo_*`, `Mu_Psico_*`, `Mu_Semestre_*`).
        3. **Ativação das 8 Regras e Defuzzificação Mamdani:** Gera as colunas `Regra_R1_Alfa` a `Regra_R8_Alfa`, `Pert_Fuzzy_Baixo_Risco` a `Pert_Fuzzy_Risco_Critico`, `Score_Depressao_Fuzzy` ($0$ a $100$) e `Classificacao_Criada_Fuzzy`.
        4. **Agrupamento Fuzzy C-Means (FCM):** Anexa na mesma base os graus de pertinência aos clusters (`Pertinencia_FCM_C0..C2`), `Cluster_FCM_Dominante`, `Incerteza_Fuzzy_Shannon` e `Estudante_Fronteira_Fuzzy`.
        """
    )

    k1, k2, k3, k4 = st.columns(4)
    n_total = len(df_exportacao_completa)
    n_alto_crit = int(df_exportacao_completa["Classificacao_Criada_Fuzzy"].isin(["Alto Risco", "Risco Crítico"]).sum())
    k1.metric("Estudantes Analisados", f"{n_total:,}")
    k2.metric("Score Fuzzy Médio (0-100)", f"{df_exportacao_completa['Score_Depressao_Fuzzy'].mean():.2f}")
    k3.metric("Alto Risco / Risco Crítico (Fuzzy)", f"{n_alto_crit:,} ({(n_alto_crit/n_total)*100:.1f}%)")
    k4.metric("Total de Colunas na Base Fuzzy", f"{len(df_exportacao_completa.columns)} variáveis")

    st.divider()

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        contagem_cat = (
            df_exportacao_completa["Classificacao_Criada_Fuzzy"]
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
            df_exportacao_completa,
            x="Score_Depressao_Fuzzy",
            nbins=35,
            color="Classificacao_Criada_Fuzzy",
            marginal="box",
            color_discrete_map=CORES_RISCO,
            title="Distribuição Contínua do Score de Depressão Criado via Centroide Fuzzy (0 a 100)",
        )
        fig_hist_score.update_layout(plot_bgcolor="white", height=390)
        st.plotly_chart(fig_hist_score, use_container_width=True)

    st.subheader("📋 Base Completa com Todas as Colunas Criadas pela Lógica Fuzzy (Mamdani + FCM)")
    st.dataframe(
        df_exportacao_completa.head(100).style.background_gradient(
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

    x_c, mf_c = univ["cgpa"]
    for nome, curva in mf_c.items():
        axes_mf[0, 0].plot(x_c, curva, lw=2.3, label=nome)
    axes_mf[0, 0].set_title("Entrada 1: CGPA (Rendimento Acadêmico [2.0 - 4.0])")
    axes_mf[0, 0].set_ylabel("Pertinência μ(x)")
    axes_mf[0, 0].legend()

    x_cu, mf_cu = univ["custo"]
    for nome, curva in mf_cu.items():
        axes_mf[0, 1].plot(x_cu, curva, lw=2.3, label=nome)
    axes_mf[0, 1].set_title("Entrada 2: Custo Pessoal Extra-Estudo [0 - 32.000]")
    axes_mf[0, 1].legend()

    x_p, mf_p = univ["psico"]
    for nome, curva in mf_p.items():
        axes_mf[1, 0].plot(x_p, curva, lw=2.3, label=nome)
    axes_mf[1, 0].set_title("Entrada 3: Vulnerabilidade Psicossocial & Saúde [0 - 10]")
    axes_mf[1, 0].set_ylabel("Pertinência μ(x)")
    axes_mf[1, 0].legend()

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
        opcoes_est = df_exportacao_completa["ID_Estudante"].head(300).tolist()
        est_id = col_m2.selectbox("Escolha o ID do Estudante:", opcoes_est, index=0)
        row_est = df_exportacao_completa[df_exportacao_completa["ID_Estudante"] == est_id].iloc[0]
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
        default=DEFAULT_FCM_FEATURES,
        key="fcm_feats_sel",
    )

    if len(features_fcm) < 2:
        st.warning("Selecione pelo menos 2 variáveis para o Fuzzy C-Means.")
    else:
        cf1, cf2, cf3 = st.columns(3)
        c_clusters = cf1.slider("Número de Clusters Fuzzy (C)", 2, 5, 3, 1, key="fcm_c_sel")
        m_expoente = cf2.slider("Expoente de Fuzzificação (m)", 1.15, 3.00, 2.00, 0.05, key="fcm_m_sel")
        limiar_trans = cf3.slider("Limiar de Fronteira Nebulosa (Pertinência Máx < τ)", 0.45, 0.75, 0.55, 0.05, key="fcm_tau_sel")

        res_fcm = st.session_state["fcm_estudantes"]

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
            df_exportacao_completa,
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
            df_exportacao_completa,
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
            df_exportacao_completa.groupby("Semestre Atual")[["Score_Depressao_Fuzzy", "CGPA (Rendimento Acadêmico)"]]
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
            df_exportacao_completa,
            x="Estilo de Moradia",
            y="Score_Depressao_Fuzzy",
            color="Status de Relacionamento",
            box=True,
            title="Score Fuzzy por Estilo de Moradia e Status de Relacionamento",
        )
        fig_moradia.update_layout(plot_bgcolor="white", height=380)
        st.plotly_chart(fig_moradia, use_container_width=True)
