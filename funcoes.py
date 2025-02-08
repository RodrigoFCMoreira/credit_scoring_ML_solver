from sklearn.metrics import roc_curve, auc
from typing import Dict
from typing import Tuple, List
from pycaret.classification import setup, create_model
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import seaborn as sns
from IPython.display import display
from typing import Any
from pycaret.classification import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from typing import Dict, Tuple
from typing import List, Literal
from pycaret.classification import ClassificationExperiment
from collections import Counter
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from optbinning import OptimalBinning
from sklearn.cluster import KMeans


def perfil_base(base_modelo: pd.DataFrame, id_col: str, target_col: str, safra_col: str) -> dict:
    """
    Calcula métricas básicas do perfil da base de dados.

    Parâmetros:
    - base_modelo (pd.DataFrame): DataFrame contendo os dados a serem analisados.
    - id_col (str): Nome da coluna que representa o identificador único (ID).
    - target_col (str): Nome da coluna que representa a variável alvo (Y).
    - safra_col (str): Nome da coluna que representa a safra.

    Retorna:
    - dict: Dicionário contendo as seguintes métricas:
        - shape: Tupla com a quantidade de linhas e colunas.
        - tipos_variaveis: Contagem dos tipos das variáveis.
        - ids_unicos: Quantidade de IDs únicos.
        - bad_rate: Taxa de maus (bad rate) da base.
        - volumetria_safras: Quantidade de registros por safra.
    """

    perfil = {}

    # 1. Verificando a volumetria de linhas e colunas
    perfil['shape'] = f"Essa base possui {base_modelo.shape[0]} linhas e {base_modelo.shape[1]} colunas"

    # 2. Verificando a tipagem das variáveis
    perfil['tipos_variaveis'] = base_modelo.dtypes.value_counts().to_dict()

    # 3. Verificando a quantidade de IDs únicos (possíveis duplicatas)
    perfil['ids_unicos'] = base_modelo[id_col].nunique()

    # 4. Verificando a taxa de maus (bad rate)
    if target_col in base_modelo.columns:
        total_bons_maus = base_modelo[target_col].value_counts()
        bad_rate = total_bons_maus / perfil['ids_unicos']
        perfil['bad_rate'] = f"Bons: {total_bons_maus[0]}({round(bad_rate[0] * 100, 1)} %), Maus: {total_bons_maus[1]} ({round(bad_rate[1] * 100, 1)}%)"
    else:
        perfil['bad_rate'] = "Coluna alvo não encontrada."

    # 5. Verificando a quantidade de safras e suas volumetrias
    if safra_col in base_modelo.columns:
        perfil['volumetria_safras'] = dict(
            sorted(base_modelo[safra_col].value_counts().to_dict().items()))
    else:
        perfil['volumetria_safras'] = "Coluna safra não encontrada."

    print("Calcula métricas básicas do perfil da base de dados.")
    print(f"Shape da base: {perfil['shape']}")
    print(f"Tipos de variáveis: {perfil['tipos_variaveis']}")
    print(f"IDs únicos: {perfil['ids_unicos']}")
    print(f"Taxa de maus (bad rate): {perfil['bad_rate']}")
    print(f"Volumetria das safras: {perfil['volumetria_safras']}")
    print("\n")

    return perfil


def plot_safra_bad_rate(df: pd.DataFrame, safra_col: str = "safra", inadimplente_col: str = "y",
                        bad_rate_min: Optional[float] = None, bad_rate_max: Optional[float] = None) -> pd.DataFrame:
    """
    Gera um gráfico de barras com a contagem de contratos por safra e 
    um gráfico de linha com a taxa de inadimplência (bad rate) no eixo secundário.

    Retorna:
    - DataFrame com as colunas: safra, contagem, total_maus, total_bons, badrate.

    Parâmetros:
    - df: DataFrame do pandas contendo os dados.
    - safra_col: Nome da coluna que representa a safra.
    - inadimplente_col: Nome da coluna que indica se o contrato foi inadimplente (1) ou não (0).
    - bad_rate_min: Valor mínimo para o eixo secundário (bad rate).
    - bad_rate_max: Valor máximo para o eixo secundário (bad rate).
    """
    # Garantir que a safra seja tratada como string para exibição correta no gráfico
    df[safra_col] = df[safra_col].astype(str)

    # Agrupando por safra para calcular os totais
    safra_stats = df.groupby(safra_col).agg(
        contagem=(inadimplente_col, "count"),
        total_maus=(inadimplente_col, "sum")
    ).reset_index()

    # Calculando os bons e a bad rate corretamente
    safra_stats["total_bons"] = safra_stats["contagem"] - \
        safra_stats["total_maus"]
    safra_stats["badrate"] = safra_stats["total_maus"] / \
        safra_stats["contagem"]

    # Criando a figura e os eixos
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Eixo principal (barras) - Total de contratos
    ax1.bar(safra_stats[safra_col], safra_stats["contagem"],
            color="blue", alpha=0.6, label="Total de Contratos")
    ax1.set_xlabel("Safra")
    ax1.set_ylabel("Total de IDs", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Ajustar rótulos do eixo X
    ax1.set_xticks(range(len(safra_stats[safra_col])))
    ax1.set_xticklabels(safra_stats[safra_col], rotation=45)

    # Criando eixo secundário (linha da bad rate)
    ax2 = ax1.twinx()
    ax2.plot(safra_stats[safra_col], safra_stats["badrate"], color="red",
             marker="o", linestyle="-", linewidth=2, label="Bad Rate")
    ax2.set_ylabel("Bad Rate (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Definir limites do eixo secundário, se fornecidos
    if bad_rate_min is not None and bad_rate_max is not None:
        ax2.set_ylim(bad_rate_min, bad_rate_max)

    # Título e layout ajustado
    plt.title("Total de IDs por Safra e Bad Rate")
    fig.tight_layout()
    plt.show()

    df[safra_col] = df[safra_col].astype('int64')

    return safra_stats


def dividir_base_safra(df: pd.DataFrame, safra_corte: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide um dataframe em base de treino e teste OOT com base em uma safra de corte.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo a coluna 'safra'.
    safra_corte (int): Valor da safra que define o corte entre treino e teste OOT.

    Retorna:
    tuple[pd.DataFrame, pd.DataFrame]: 
        - treino (pd.DataFrame): Base de treino contendo safras anteriores ao safra_corte.
        - teste_oot (pd.DataFrame): Base de teste OOT contendo safras a partir do safra_corte.
    """

    # Garantindo que a coluna 'safra' seja do tipo inteiro
    df['safra'] = df['safra'].astype(int)

    # Separando a base de treino (safras menores que a safra de corte)
    treino: pd.DataFrame = df[df['safra'] < safra_corte]

    # Separando a base de teste OOT (safras maiores ou iguais à safra de corte)
    teste_oot: pd.DataFrame = df[df['safra'] >= safra_corte]

    # Criando uma tabela de volumetria para melhor visualização da divisão
    volumetria: pd.DataFrame = pd.DataFrame({
        'Conjunto': ['Treino', 'Teste OOT'],
        'Registros': [treino.shape[0], teste_oot.shape[0]]
    })

    # Exibindo os resultados
    print("\n### Volumetria da Base ###")
    print(volumetria)

    print("\n### Safras na Base de Treino ###")
    print(treino['safra'].value_counts().sort_index())

    print("\n### Safras na Base de Teste OOT ###")
    print(teste_oot['safra'].value_counts().sort_index())

    return treino, teste_oot


def remover_missings(df: pd.DataFrame, perc_miss: int = 20) -> pd.DataFrame:
    """
    Remove colunas que possuem um percentual de valores ausentes (missings) maior ou igual ao valor definido em perc_miss.

    Parâmetros:
    - df (pd.DataFrame): DataFrame de entrada.
    - perc_miss (int, opcional): Percentual máximo de valores ausentes permitido em uma coluna. 
      Colunas com valores ausentes acima desse percentual serão removidas. Padrão é 20.

    Retorna:
    - pd.DataFrame: DataFrame sem as colunas que ultrapassam o limite de valores ausentes.
    """
    qt_rows = df.shape[0]

    # Calcula a porcentagem de valores ausentes por coluna
    pct_missing = df.isnull().sum() / qt_rows * 100

    # Filtra as colunas que devem ser removidas
    colunas_removidas = pct_missing[pct_missing >= perc_miss].index.tolist()

    # Exibe a lista de colunas removidas
    if colunas_removidas:
        print(
            f"Colunas removidas({len(colunas_removidas)}): {colunas_removidas}")
    else:
        print("Nenhuma coluna removida.")

    # Retorna o DataFrame filtrado
    return df.drop(columns=colunas_removidas)


def escolher_estrategia_imputacao(df: pd.DataFrame) -> dict:
    """
    Função que determina a estratégia de imputação de valores ausentes para cada coluna de um DataFrame,
    com base no tipo da variável, presença de outliers e porcentagem de valores ausentes.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados a serem analisados.

    Retorna:
    dict: Um dicionário onde as chaves são os nomes das colunas e os valores são as estratégias de imputação.
    """
    estrategias = {}

    for coluna in df.columns:
        # Porcentagem de valores ausentes
        missing_pct = df[coluna].isna().mean()
        dtype = df[coluna].dtype  # Tipo de dado da coluna

        if dtype == 'object':  # Variável categórica
            estrategia = 'Moda'

        else:  # Variável numérica
            valores = df[coluna].dropna()

            # Identificando outliers usando IQR
            Q1, Q3 = np.percentile(valores, [25, 75])
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            tem_outliers = ((valores < limite_inferior) |
                            (valores > limite_superior)).any()

            # Definição da estratégia baseada em missing_pct e outliers
            if tem_outliers and 0.05 <= missing_pct <= 0.20:
                estrategia = 'median'
            elif not tem_outliers and missing_pct < 0.05:
                estrategia = 'mean'
            else:
                estrategia = 'median'  # Estratégia segura para outros casos

        estrategias[coluna] = estrategia

    return estrategias


def aplicar_imputacao_treino(df: pd.DataFrame, regra_imputacao: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, float], Dict[str, float]]:
    """
    Aplica imputação de valores ausentes em um DataFrame com base em uma regra especificada.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados a serem processados.
    - regra_imputacao (Dict[str, str]): Dicionário onde as chaves são os nomes das colunas do DataFrame 
      e os valores são as regras de imputação ('median' para mediana ou 'mean' para média).

    Retorna:
    - Tuple contendo:
        1. O DataFrame com os valores ausentes imputados.
        2. O dicionário de regras de imputação utilizado.
        3. O dicionário com os valores de mediana calculados por coluna.
        4. O dicionário com os valores de média calculados por coluna.

    Obs: Se a coluna informada na regra de imputação não existir no DataFrame, uma mensagem será exibida.

    """

    # Calcula os valores de mediana e média de cada coluna
    dict_mediana: Dict[str, float] = df.median().to_dict()
    dict_media: Dict[str, float] = df.mean().to_dict()

    # Itera sobre as colunas do DataFrame e aplica a imputação conforme a regra especificada
    for col in df.columns:
        if col in regra_imputacao:
            if regra_imputacao[col] == 'median':
                df[col] = df[col].fillna(dict_mediana[col])
            elif regra_imputacao[col] == 'mean':
                df[col] = df[col].fillna(dict_media[col])
        else:
            print(
                f"A regra de imputação para a coluna '{col}' não foi especificada.")

    return df, regra_imputacao, dict_mediana, dict_media


def aplicar_imputacao_teste(df: pd.DataFrame,
                            regra_imputacao: Dict[str, str],
                            dict_mediana: Dict[str, float],
                            dict_media: Dict[str, float]) -> pd.DataFrame:
    """
    Aplica imputação de valores ausentes em um DataFrame com base em regras especificadas.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados com valores ausentes a serem imputados.

    regra_imputacao : Dict[str, str]
        Dicionário onde as chaves são os nomes das colunas e os valores indicam a regra de imputação:
        - 'median' para imputação com a mediana.
        - 'mean' para imputação com a média.

    dict_mediana : Dict[str, float]
        Dicionário contendo as medianas das colunas a serem imputadas.

    dict_media : Dict[str, float]
        Dicionário contendo as médias das colunas a serem imputadas.

    Retorno:
    --------
    pd.DataFrame
        DataFrame com os valores ausentes imputados conforme as regras definidas.
    """

    # Itera sobre as colunas do DataFrame e aplica a imputação conforme a regra especificada
    for col in df.columns:
        if col in regra_imputacao:
            if regra_imputacao[col] == 'median':
                df[col] = df[col].fillna(dict_mediana.get(col, df[col]))
            elif regra_imputacao[col] == 'mean':
                df[col] = df[col].fillna(dict_media.get(col, df[col]))
        else:
            print(
                f"A regra de imputação para a coluna '{col}' não foi especificada.")

    return df


def selecao_variaveis(
    data: pd.DataFrame,
    target: str,
    methods: List[Literal['classic', 'univariate', 'sequential']],
    selection_rule: Literal['intersection', 'union', 'voting'] = 'intersection'
) -> List[str]:
    """
    Realiza a seleção de variáveis no PyCaret usando diferentes métodos e regras de combinação.

    Parâmetros:
    - data (pd.DataFrame): Dataset contendo as variáveis preditoras e a variável alvo.
    - target (str): Nome da variável alvo.
    - methods (List[str]): Lista com os métodos de seleção a serem aplicados. Opções:
        - 'classic' (RFE - Recursive Feature Elimination)
        - 'univariate' (Testes estatísticos ANOVA/qui-quadrado)
        - 'sequential' (Sequential Feature Selection - SFS)
    - selection_rule (str): Método de combinação das variáveis selecionadas. Opções:
        - 'intersection': Mantém apenas as variáveis escolhidas por todos os métodos.
        - 'union': Mantém todas as variáveis selecionadas por pelo menos um método.
        - 'voting': Mantém variáveis selecionadas por pelo menos 2 dos métodos escolhidos.

    Retorno:
    - List[str]: Lista final de variáveis selecionadas.
    """

    """
    Observações relevantes sobre o uso:
    Bases Pequenas/Médias (até 1000 variáveis) → classic ou sequential
    Se precisar de um modelo bem ajustado → sequential.
    Se quiser um método robusto baseado no impacto real das variáveis → classic.

    Bases Grandes (acima de 5000 variáveis) → univariate
    Se a base é muito grande, o univariate é mais rápido e ajuda a filtrar variáveis antes de rodar modelos mais pesados.
    Depois, pode usar classic ou sequential só nas melhores variáveis.
    
    """

    # Verifica se os métodos fornecidos são válidos
    valid_methods = {'classic', 'univariate', 'sequential'}
    if not set(methods).issubset(valid_methods):
        raise ValueError(
            f"Os métodos devem estar entre {valid_methods}, mas recebeu {methods}")

    selected_features_sets = []

    for method in methods:
        exp = ClassificationExperiment()  # Inicializa o experimento
        exp.setup(data, target=target, feature_selection=True,
                  feature_selection_method=method, verbose=False)

        # Pegamos as features selecionadas via get_config
        selected_features = exp.get_config("X_train").columns.to_list()
        selected_features_sets.append(set(selected_features))

    # Combinação das seleções
    if selection_rule == 'intersection':
        variaveis_selecionadas = list(
            set.intersection(*selected_features_sets))
    elif selection_rule == 'union':
        variaveis_selecionadas = list(set.union(*selected_features_sets))
    elif selection_rule == 'voting':
        feature_counts = Counter(
            [feat for features in selected_features_sets for feat in features])
        variaveis_selecionadas = [feat for feat,
                                  count in feature_counts.items() if count >= 2]
    else:
        raise ValueError(
            "selection_rule deve ser 'intersection', 'union' ou 'voting'")

    return variaveis_selecionadas


def resumo_estatistico(df: pd.DataFrame) -> None:
    """
    Exibe um resumo estatístico das variáveis numéricas e categóricas do DataFrame.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    """
    df_numeric = df.select_dtypes(include=[np.number])
    df_categoric = df.select_dtypes(include=["O"])

    if not df_numeric.empty:
        print("📌 Resumo Estatístico das Variáveis Numéricas:")
        display(df_numeric.describe())

    if not df_categoric.empty:
        print("\n📌 Resumo Estatístico das Variáveis Categóricas:")
        display(df_categoric.describe())


def grafico_percentual_valores_ausentes(df: pd.DataFrame) -> None:
    """
    Plota um gráfico de barras com o percentual de valores ausentes por variável.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    """
    percentual_missing = (df.isnull().sum() / len(df)) * 100
    percentual_missing = percentual_missing[percentual_missing > 0].sort_values(
        ascending=False)

    if percentual_missing.empty:
        print("✅ Nenhuma variável possui valores ausentes.")
        return

    plt.figure(figsize=(10, 5))
    sns.barplot(x=percentual_missing.index,
                y=percentual_missing.values, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Percentual de valores ausentes (%)")
    plt.xlabel("Variáveis")
    plt.title("Percentual de Valores Ausentes por Variável")

    # Exibir os valores acima das barras
    for index, value in enumerate(percentual_missing):
        plt.text(index, value, f"{value:.0f}%",
                 ha="center", va="bottom", fontsize=8)

    plt.show()


def matriz_correlacao(df: pd.DataFrame) -> None:
    """
    Plota uma matriz de correlação de Pearson para variáveis numéricas.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    """
    df_numeric = df.select_dtypes(include=[np.number])

    if df_numeric.empty:
        print("⚠️ Nenhuma variável numérica para calcular correlação.")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_numeric.corr(), annot=True,
                cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlação de Pearson")
    plt.show()


def histograma_variaveis_numericas(df: pd.DataFrame) -> None:
    """
    Plota histogramas para todas as variáveis numéricas do DataFrame.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    """
    df_numeric = df.select_dtypes(include=[np.number])

    if df_numeric.empty:
        print("⚠️ Nenhuma variável numérica encontrada no DataFrame.")
        return

    df_numeric.hist(figsize=(8, 8), bins=20,
                    color="skyblue", edgecolor="black")
    plt.suptitle("Distribuição das Variáveis Numéricas", fontsize=14)
    plt.show()


def grafico_variaveis_categoricas(df: pd.DataFrame) -> None:
    """
    Plota gráficos de barras para as variáveis categóricas do DataFrame.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    """
    df_categoric = df.select_dtypes(include=["O"])

    if df_categoric.empty:
        print("⚠️ Nenhuma variável categórica encontrada no DataFrame.")
        return

    for col in df_categoric.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(
            x=df[col], order=df[col].value_counts().index, palette="Set2")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Distribuição da Variável Categórica: {col}")
        plt.ylabel("Contagem")
        plt.show()


################################# MODELAGEM - TREINAMENTO E ESCORAGEM ####################################################################


def pipeline_modelagem(train: pd.DataFrame, test: pd.DataFrame, id_col: str, safra_col: str, target_col: str, lista_vars_numericas_categorizar: List[str]) -> tuple[
    pd.DataFrame, pd.DataFrame, Any, Any, Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Função para treinar modelos de classificação (LightGBM e Regressão Logística) com tunning de hiperparâmetros usando PyCaret.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados
    id_col (str): Nome da coluna identificadora (será mantida nas bases escoradas)
    safra_col (str): Nome da coluna da safra (será mantida nas bases escoradas)
    target_col (str): Nome da variável alvo (target)

    Retorna:
    - pipeline_miss: objeto para fit_transform, imputer de missings
    - DataFrame de desenvolvimento (train)
    - DataFrame de validação (test)
    - Objeto do modelo LightGBM ajustado
    - Objeto do modelo Regressão Logística ajustado
    - DataFrame de treino escorado para LightGBM
    - DataFrame de teste escorado para LightGBM
    - DataFrame de treino escorado para Regressão Logística
    - DataFrame de teste escorado para Regressão Logística
    """

    # Configuração do PyCaret

    clf_setup = setup(
        # Remove ID e Safra na modelagem
        data=train.drop(columns=[id_col, safra_col]),
        target=target_col,

        # SMOTE (mesmo sendo 30% de 1 e 70% de 0 vamos testar)
        # Balanceamento de classes (caso haja desbalanceamento)
        # fix_imbalance=True,
        # fix_imbalance_method='SMOTE',
        numeric_imputation="median",
        categorical_imputation="most_frequent",

        # 🔸 Discretizando (binning) variáveis numéricas para reduzir oscilações extremas
        bin_numeric_features=lista_vars_numericas_categorizar,  # Agrupar em faixas

        # vamos avaliar o modelo por meio da escoragem oos e oot, então não precisamos de um novo conjunto de testes.
        train_size=0.9,
        data_split_shuffle=True,  # Embaralhar antes de dividir
        data_split_stratify=True,  # Estratificação para manter a proporção das classes

        session_id=42,
        verbose=False,
        fold_strategy='stratifiedkfold',
        fold=10,

        remove_multicollinearity=True,
        # Remove features com correlação acima de 90% (pois vamos tentar gerar novas features de segunda ordem)
        multicollinearity_threshold=0.9,

        # Engenharia de atributos
        # polynomial_features=True,  # Criar features polinomiais
        # polynomial_degree=2,  # Grau do polinômio

        normalize=True,
        normalize_method='zscore',  # (valor - media)/desvio_padrao
        feature_selection=False,  # já fizemos esse passo anteriormente


    )

    # Modelos a serem comparados
    models_to_compare = ['lightgbm', 'lr']

    # Comparação de modelos
    best_model_initial = compare_models(
        include=models_to_compare, sort='AUC', turbo=False)

    # Hiperparâmetros otimizados para LightGBM
    tuned_lightgbm = tune_model(
        create_model('lightgbm'),  # lightgbm
        custom_grid={

            # ajusta a influência das classes automaticamente. (testando)
            'class_weight': ['balanced'],

            # Número máximo de folhas em cada árvore (define a complexidade dos splits)
            # Valores maiores permitem capturar interações mais complexas, mas aumentam o risco de overfitting.
            'num_leaves': [3, 5, 10, 20],

            # Taxa de aprendizado (step size que controla o ajuste do modelo a cada iteração)
            # Valores menores tornam o treinamento mais estável, mas exigem mais iterações.
            'learning_rate': [0.005, 0.01, 0.03],

            # Número total de árvores no modelo
            # Um número maior pode melhorar a performance, mas pode levar a overfitting se for muito alto.
            'n_estimators': [50, 100, 200],

            # Profundidade máxima das árvores (limita a complexidade do modelo)
            # Evita que o modelo fique muito profundo e overfitado aos dados de treino.
            'max_depth': [3, 5],

            # Fracção aleatória das amostras usadas para construir cada árvore (controle de bagging)
            # Valores menores aumentam a diversidade das árvores e reduzem overfitting.
            'subsample': [0.6, 0.75, 0.9],

            # Fracção das features usadas para construir cada árvore (controle de feature bagging)
            # Reduz a dependência de features específicas, melhorando a generalização.
            'colsample_bytree': [0.6, 0.75, 0.9],

            # Regularização L1 (Lasso), penaliza coeficientes grandes e força alguns a zero
            # Ajuda a reduzir o overfitting tornando o modelo mais simples.
            'reg_alpha': [0.1, 0.5, 1, 2],

            # Regularização L2 (Ridge), penaliza coeficientes grandes, mas sem zerá-los
            # Suaviza os pesos do modelo e ajuda na generalização.
            'reg_lambda': [0.1, 0.5, 1, 2]

        },

        optimize='AUC'
    )

    # Hiperparâmetros otimizados para Regressão Logística
    tuned_lr = tune_model(
        create_model('lr'),
        custom_grid={
            # Parâmetro de regularização inversa (quanto menor, maior a regularização)
            # Valores menores impõem mais penalização nos coeficientes, ajudando a evitar overfitting.
            'C': [0.01, 0.1, 1, 10],

            # Número máximo de iterações para a convergência do algoritmo
            # Se o modelo não convergir, aumentar esse valor pode ajudar.
            'max_iter': [100, 200, 500],

            # Algoritmo utilizado para otimizar a regressão logística
            # 'liblinear' é indicado para pequenos datasets e modelos simples
            # 'lbfgs' funciona bem para grandes conjuntos de dados e suporta regularização L2.
            'solver': ['liblinear', 'lbfgs']
        },
        optimize='AUC'
    )

    # Escolha do melhor modelo
    final_best_model = compare_models(
        include=[tuned_lightgbm, tuned_lr],
        sort='AUC'
    )

    # Obtendo métricas de cada modelo
    results_lgbm = pull()
    metrics_lgbm = results_lgbm.loc[results_lgbm['Model']
                                    == 'Light Gradient Boosting Machine']

    results_lr = pull()
    metrics_lr = results_lr.loc[results_lr['Model'] == 'Logistic Regression']

    auc_lgbm = metrics_lgbm['AUC'].values[0] if 'AUC' in results_lgbm.columns else "N/A"
    auc_lr = metrics_lr['AUC'].values[0] if 'AUC' in results_lr.columns else "N/A"

    # Impressão do modelo vencedor e justificativa
    print("\n🏆 **Modelo Vencedor:**", final_best_model)
    if final_best_model == tuned_lightgbm:
        print(f"✅ LightGBM escolhido com AUC: {auc_lgbm}")
    else:
        print(f"✅ Regressão Logística escolhida com AUC: {auc_lr}")

    # 🔹 ESCORAGEM DOS MODELOS

    # Aplicando LightGBM no conjunto de treino e teste
    train_lightgbm_scored = predict_model(
        tuned_lightgbm, data=train, probability_threshold=0.5, raw_score=True)
    test_lightgbm_scored = predict_model(
        tuned_lightgbm, data=test, probability_threshold=0.5, raw_score=True)

    # Aplicando Regressão Logística no conjunto de treino e teste
    train_lr_scored = predict_model(
        tuned_lr, data=train, probability_threshold=0.5, raw_score=True)
    test_lr_scored = predict_model(
        tuned_lr, data=test, probability_threshold=0.5, raw_score=True)

    print("base escorada pycaret")
    print(train_lightgbm_scored)
    print(train_lightgbm_scored.columns)

    prob_col_1 = 'prediction_score_1'

    # Criar a coluna Score_0 como 1 - probabilidade da classe 1
    train_lightgbm_escorado = train[[id_col, safra_col, target_col]].copy()
    train_lightgbm_escorado["score_1"] = train_lightgbm_scored[prob_col_1]
    train_lightgbm_escorado["score_0"] = 1 - train_lightgbm_escorado["score_1"]

    test_lightgbm_escorado = test[[id_col, safra_col, target_col]].copy()
    test_lightgbm_escorado["score_1"] = test_lightgbm_scored[prob_col_1]
    test_lightgbm_escorado["score_0"] = 1 - test_lightgbm_escorado["score_1"]

    train_regressao_escorado = train[[id_col, safra_col, target_col]].copy()
    train_regressao_escorado["score_1"] = train_lr_scored[prob_col_1]
    train_regressao_escorado["score_0"] = 1 - \
        train_regressao_escorado["score_1"]

    test_regressao_escorado = test[[id_col, safra_col, target_col]].copy()
    test_regressao_escorado["score_1"] = test_lr_scored[prob_col_1]
    test_regressao_escorado["score_0"] = 1 - test_regressao_escorado["score_1"]

    # Retornando os resultados
    return train, test, tuned_lightgbm, tuned_lr, train_lightgbm_escorado, test_lightgbm_escorado, train_regressao_escorado, test_regressao_escorado


############################################ FUNCOES DE METRICAS E AVALIACAO DE MODELOS ############################################


def plot_comparacao_roc(
    train_lightgbm: pd.DataFrame,
    test_lightgbm: pd.DataFrame,
    train_regressao: pd.DataFrame,
    test_regressao: pd.DataFrame,
    test_oot_lightgbm: pd.DataFrame = None,
    test_oot_regressao: pd.DataFrame = None
) -> None:
    """
    Gera dois gráficos de curva ROC lado a lado para comparar os modelos LightGBM e Regressão Logística.

    O primeiro gráfico contém as curvas ROC do modelo LightGBM para treino, teste e opcionalmente OOT.
    O segundo gráfico contém as curvas ROC do modelo de Regressão Logística para treino, teste e opcionalmente OOT.

    Parâmetros:
    - train_lightgbm (pd.DataFrame): DataFrame com os dados de treino para o modelo LightGBM.
    - test_lightgbm (pd.DataFrame): DataFrame com os dados de teste para o modelo LightGBM.
    - train_regressao (pd.DataFrame): DataFrame com os dados de treino para o modelo de Regressão Logística.
    - test_regressao (pd.DataFrame): DataFrame com os dados de teste para o modelo de Regressão Logística.
    - test_oot_lightgbm (pd.DataFrame, opcional): DataFrame com os dados OOT para o modelo LightGBM.
    - test_oot_regressao (pd.DataFrame, opcional): DataFrame com os dados OOT para o modelo de Regressão Logística.

    Retorno:
    - None. A função exibe os gráficos.
    """

    # Criar figura com dois subgráficos
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Lista de dados para iteração
    modelos = [
        ("LightGBM", train_lightgbm, test_lightgbm,
         test_oot_lightgbm, axes[0]),
        ("Regressão Logística", train_regressao,
         test_regressao, test_oot_regressao, axes[1])
    ]

    for nome_modelo, train_df, test_df, test_oot_df, ax in modelos:
        # Garantir que estamos usando score_1 (probabilidade da classe positiva)
        y_train, y_test = train_df["y"], test_df["y"]
        scores_train, scores_test = train_df["score_1"], test_df["score_1"]

        # Calcular curva ROC para treino
        fpr_train, tpr_train, _ = roc_curve(y_train, scores_train)
        auc_train = auc(fpr_train, tpr_train)

        # Calcular curva ROC para teste
        fpr_test, tpr_test, _ = roc_curve(y_test, scores_test)
        auc_test = auc(fpr_test, tpr_test)

        # Plotar curvas para treino e teste
        ax.plot(fpr_train, tpr_train,
                label=f'Treino (AUC = {auc_train:.2f})', color='blue')
        ax.plot(fpr_test, tpr_test,
                label=f'Teste (AUC = {auc_test:.2f})', color='red')

        # Se houver dados OOT, calcular e plotar
        if test_oot_df is not None:
            y_oot = test_oot_df["y"]
            scores_oot = test_oot_df["score_1"]
            fpr_oot, tpr_oot, _ = roc_curve(y_oot, scores_oot)
            auc_oot = auc(fpr_oot, tpr_oot)
            ax.plot(fpr_oot, tpr_oot,
                    label=f'OOT (AUC = {auc_oot:.2f})', color='green')

        # Linha diagonal de referência
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')

        # Configurações do gráfico
        ax.set_title(f'Curva ROC - {nome_modelo}')
        ax.set_xlabel('Taxa de Falsos Positivos (FPR)')
        ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
        ax.legend()
        ax.grid(True)

    # Exibir gráficos
    plt.tight_layout()
    plt.show()


def plot_comparacao_prc(
    train_lightgbm: pd.DataFrame,
    test_lightgbm: pd.DataFrame,
    train_regressao: pd.DataFrame,
    test_regressao: pd.DataFrame,
    test_oot_lightgbm: pd.DataFrame = None,
    test_oot_regressao: pd.DataFrame = None
) -> None:
    """
    Gera dois gráficos de Curva de Precisão-Revocação (PRC) lado a lado para comparar os modelos LightGBM e Regressão Logística.

    O primeiro gráfico contém as curvas PRC do modelo LightGBM para treino, teste e opcionalmente OOT.
    O segundo gráfico contém as curvas PRC do modelo de Regressão Logística para treino, teste e opcionalmente OOT.

    Parâmetros:
    - train_lightgbm (pd.DataFrame): DataFrame com os dados de treino para o modelo LightGBM.
    - test_lightgbm (pd.DataFrame): DataFrame com os dados de teste para o modelo LightGBM.
    - train_regressao (pd.DataFrame): DataFrame com os dados de treino para o modelo de Regressão Logística.
    - test_regressao (pd.DataFrame): DataFrame com os dados de teste para o modelo de Regressão Logística.
    - test_oot_lightgbm (pd.DataFrame, opcional): DataFrame com os dados OOT para o modelo LightGBM.
    - test_oot_regressao (pd.DataFrame, opcional): DataFrame com os dados OOT para o modelo de Regressão Logística.

    Retorno:
    - None. A função exibe os gráficos.
    """

    # Criar uma figura com dois gráficos lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Lista com os modelos para iterar
    modelos = [
        ("LightGBM", train_lightgbm, test_lightgbm,
         test_oot_lightgbm, axes[0]),
        ("Regressão Logística", train_regressao,
         test_regressao, test_oot_regressao, axes[1])
    ]

    for nome_modelo, train_df, test_df, test_oot_df, ax in modelos:
        # Definir variáveis de resposta e pontuações preditivas
        y_train, y_test = train_df["y"], test_df["y"]
        scores_train, scores_test = train_df["score_1"], test_df["score_1"]

        # Calcular Curva PRC para treino
        precision_train, recall_train, _ = precision_recall_curve(
            y_train, scores_train)
        auc_train = auc(recall_train, precision_train)

        # Calcular Curva PRC para teste
        precision_test, recall_test, _ = precision_recall_curve(
            y_test, scores_test)
        auc_test = auc(recall_test, precision_test)

        # Plotar curva PRC para treino e teste
        ax.plot(recall_train, precision_train,
                label=f'Treino (AUC = {auc_train:.2f})', color='blue')
        ax.plot(recall_test, precision_test,
                label=f'Teste (AUC = {auc_test:.2f})', color='red')

        # Se houver dados OOT, calcular e plotar
        if test_oot_df is not None:
            y_oot = test_oot_df["y"]
            scores_oot = test_oot_df["score_1"]
            precision_oot, recall_oot, _ = precision_recall_curve(
                y_oot, scores_oot)
            auc_oot = auc(recall_oot, precision_oot)
            ax.plot(recall_oot, precision_oot,
                    label=f'OOT (AUC = {auc_oot:.2f})', color='green')

        # Configurações do gráfico
        ax.set_title(f'Curva PRC - {nome_modelo}')
        ax.set_xlabel('Revocação')
        ax.set_ylabel('Precisão')
        ax.legend()
        ax.grid(True)

    # Ajustar layout e exibir gráficos
    plt.tight_layout()
    plt.show()


def plot_comparacao_ks(
    train_lightgbm: pd.DataFrame,
    test_lightgbm: pd.DataFrame,
    train_regressao: pd.DataFrame,
    test_regressao: pd.DataFrame,
    test_oot_lightgbm: pd.DataFrame = None,
    test_oot_regressao: pd.DataFrame = None
) -> None:
    """
    Gera dois gráficos da Curva KS (Kolmogorov-Smirnov) lado a lado para comparar os modelos LightGBM e Regressão Logística.

    O eixo X representa a probabilidade de mau (score do modelo, variando de 0 a 1).
    O eixo Y representa a população acumulada para cada classe (mau e bom).

    O primeiro gráfico contém as curvas KS do modelo LightGBM para treino, teste e opcionalmente OOT.
    O segundo gráfico contém as curvas KS do modelo de Regressão Logística para treino, teste e opcionalmente OOT.

    Parâmetros:
    - train_lightgbm (pd.DataFrame): DataFrame com os dados de treino para o modelo LightGBM.
    - test_lightgbm (pd.DataFrame): DataFrame com os dados de teste para o modelo LightGBM.
    - train_regressao (pd.DataFrame): DataFrame com os dados de treino para o modelo de Regressão Logística.
    - test_regressao (pd.DataFrame): DataFrame com os dados de teste para o modelo de Regressão Logística.
    - test_oot_lightgbm (pd.DataFrame, opcional): DataFrame com os dados OOT para o modelo LightGBM.
    - test_oot_regressao (pd.DataFrame, opcional): DataFrame com os dados OOT para o modelo de Regressão Logística.

    Retorno:
    - None. A função exibe os gráficos.
    """

    def calcular_ks(y_true, scores):
        """Calcula a curva KS para 'mau' (y=1) e 'bom' (y=0),
        mostrando a população acumulada sobre a probabilidade de mau."""

        df = pd.DataFrame({"y": y_true, "score": scores})
        df = df.sort_values("score", ascending=True)

        total_mau = (df["y"] == 1).sum()
        total_bom = (df["y"] == 0).sum()

        df["cumulativo_mau"] = (df["y"] == 1).cumsum() / total_mau
        df["cumulativo_bom"] = (df["y"] == 0).cumsum() / total_bom

        df["ks"] = np.abs(df["cumulativo_mau"] - df["cumulativo_bom"])
        ks_max = df["ks"].max()
        ks_max_score = df.loc[df["ks"].idxmax(), "score"]
        probabilidade_mau_ks = df.loc[df["ks"].idxmax(), "cumulativo_mau"]

        return df["score"], df["cumulativo_mau"], df["cumulativo_bom"], ks_max, ks_max_score, probabilidade_mau_ks

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    modelos = [
        ("LightGBM", train_lightgbm, test_lightgbm,
         test_oot_lightgbm, axes[0]),
        ("Regressão Logística", train_regressao,
         test_regressao, test_oot_regressao, axes[1])
    ]

    for nome_modelo, train_df, test_df, test_oot_df, ax in modelos:
        y_train, y_test = train_df["y"], test_df["y"]
        scores_train, scores_test = train_df["score_1"], test_df["score_1"]

        prob_train, cum_mau_train, cum_bom_train, ks_train, ks_train_score, prob_mau_train = calcular_ks(
            y_train, scores_train)
        prob_test, cum_mau_test, cum_bom_test, ks_test, ks_test_score, prob_mau_test = calcular_ks(
            y_test, scores_test)

        ax.plot(prob_train, cum_mau_train, label=f'Treino - Mau', color='blue')
        ax.plot(prob_train, cum_bom_train, label=f'Treino - Bom',
                linestyle='--', color='blue')
        ax.scatter(ks_train_score, prob_mau_train, color='blue', marker='o',
                   label=f'KS Treino = {ks_train:.2%} (P={ks_train_score:.2f})')

        ax.plot(prob_test, cum_mau_test, label=f'Teste - Mau', color='red')
        ax.plot(prob_test, cum_bom_test, label=f'Teste - Bom',
                linestyle='--', color='red')
        ax.scatter(ks_test_score, prob_mau_test, color='red', marker='o',
                   label=f'KS Teste = {ks_test:.2%} (P={ks_test_score:.2f})')

        if test_oot_df is not None:
            y_oot = test_oot_df["y"]
            scores_oot = test_oot_df["score_1"]
            prob_oot, cum_mau_oot, cum_bom_oot, ks_oot, ks_oot_score, prob_mau_oot = calcular_ks(
                y_oot, scores_oot)
            ax.plot(prob_oot, cum_mau_oot, label=f'OOT - Mau', color='green')
            ax.plot(prob_oot, cum_bom_oot, label=f'OOT - Bom',
                    linestyle='--', color='green')
            ax.scatter(ks_oot_score, prob_mau_oot, color='green', marker='o',
                       label=f'KS OOT = {ks_oot:.2%} (P={ks_oot_score:.2f})')

        ax.set_title(f'Curva KS - {nome_modelo}')
        ax.set_xlabel('Probabilidade de Mau')
        ax.set_ylabel('Percentual Acumulado da População')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_comparacao_decil(
    train_lightgbm: pd.DataFrame,
    test_lightgbm: pd.DataFrame,
    train_regressao: pd.DataFrame,
    test_regressao: pd.DataFrame,
    test_oot_lightgbm: pd.DataFrame = None,
    test_oot_regressao: pd.DataFrame = None,
    num_divisoes: int = 10
) -> None:
    """
    Gera gráficos de barras comparativos da distribuição por decis (ou outra divisão escolhida)
    para os modelos LightGBM e Regressão Logística. Inclui uma linha horizontal indicando a taxa média de maus.

    Parâmetros:
    - train_lightgbm (pd.DataFrame): DataFrame de treino para o modelo LightGBM.
    - test_lightgbm (pd.DataFrame): DataFrame de teste para o modelo LightGBM.
    - train_regressao (pd.DataFrame): DataFrame de treino para o modelo de Regressão Logística.
    - test_regressao (pd.DataFrame): DataFrame de teste para o modelo de Regressão Logística.
    - test_oot_lightgbm (pd.DataFrame, opcional): DataFrame OOT para LightGBM.
    - test_oot_regressao (pd.DataFrame, opcional): DataFrame OOT para Regressão Logística.
    - num_divisoes (int): Número de divisões para os grupos (exemplo: 10 para decis, 5 para quintis etc.).

    Retorno:
    - None. A função exibe os gráficos de barras.
    """

    def calcular_decil(y_true, scores, num_divisoes):
        """Divide os dados em grupos e calcula a taxa de eventos (mau) por grupo."""
        df = pd.DataFrame({"y": y_true, "score": scores})

        # Criar os grupos (decis, quintis, etc.)
        df["grupo"] = pd.qcut(df["score"], q=num_divisoes,
                              labels=False, duplicates="drop")

        # Calcular a taxa de maus por grupo
        decil_summary = df.groupby("grupo")["y"].mean()

        return decil_summary

    # Criar figura com dois subgráficos
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Lista de dados para iteração
    modelos = [
        ("LightGBM", train_lightgbm, test_lightgbm,
         test_oot_lightgbm, axes[0]),
        ("Regressão Logística", train_regressao,
         test_regressao, test_oot_regressao, axes[1])
    ]

    for nome_modelo, train_df, test_df, oot_df, ax in modelos:
        # Garantir que estamos usando 'score_1' como probabilidade da classe positiva (mau)
        y_train, y_test = train_df["y"], test_df["y"]
        scores_train, scores_test = train_df["score_1"], test_df["score_1"]

        # Calcular distribuição por decis
        decil_train = calcular_decil(y_train, scores_train, num_divisoes)
        decil_test = calcular_decil(y_test, scores_test, num_divisoes)

        # Taxa média de maus
        taxa_mau_train = round(y_train.mean(), 4)
        taxa_mau_test = round(y_test.mean(), 4)

        # Se houver OOT, calcular também
        if oot_df is not None:
            y_oot = oot_df["y"]
            scores_oot = oot_df["score_1"]
            decil_oot = calcular_decil(y_oot, scores_oot, num_divisoes)
            taxa_mau_oot = round(y_oot.mean(), 4)
        else:
            decil_oot = None
            taxa_mau_oot = None

        # Criar gráfico de barras
        indices = np.arange(len(decil_train))
        largura = 0.3  # Largura das barras

        ax.bar(indices - largura, decil_train,
               largura, label="Treino", color="blue")
        ax.bar(indices, decil_test, largura, label="Teste", color="red")

        if decil_oot is not None:
            ax.bar(indices + largura, decil_oot,
                   largura, label="OOT", color="green")

        # Adicionar linha horizontal com a taxa média de maus, com valores na legenda
        ax.axhline(taxa_mau_train, color="blue", linestyle="--", linewidth=1,
                   label=f"Média Treino: {taxa_mau_train:.2%}")
        ax.axhline(taxa_mau_test, color="red", linestyle="--", linewidth=1,
                   label=f"Média Teste: {taxa_mau_test:.2%}")

        if taxa_mau_oot is not None:
            ax.axhline(taxa_mau_oot, color="green", linestyle="--", linewidth=1,
                       label=f"Média OOT: {taxa_mau_oot:.2%}")

        # Configurações do gráfico
        ax.set_title(f'Distribuição por Grupo - {nome_modelo}')
        ax.set_xlabel(f'Grupo ({num_divisoes} divisões)')
        ax.set_ylabel('Taxa de Mau (%)')
        ax.set_xticks(indices)
        ax.set_xticklabels(indices + 1)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Exibir gráficos
    plt.tight_layout()
    plt.show()


def gerar_tabela_avaliacao(
    train_lightgbm: pd.DataFrame,
    test_lightgbm: pd.DataFrame,
    train_regressao: pd.DataFrame,
    test_regressao: pd.DataFrame,
    test_oot_lightgbm: pd.DataFrame = None,
    test_oot_regressao: pd.DataFrame = None,
    num_divisoes: int = 10
) -> pd.DataFrame:
    """
    Gera uma tabela única contendo estatísticas sobre quantis de score para diferentes conjuntos de dados.

    Parâmetros:
    -----------
    train_lightgbm : pd.DataFrame
        Conjunto de treino do modelo LightGBM.
    test_lightgbm : pd.DataFrame
        Conjunto de teste do modelo LightGBM.
    train_regressao : pd.DataFrame
        Conjunto de treino do modelo de regressão.
    test_regressao : pd.DataFrame
        Conjunto de teste do modelo de regressão.
    test_oot_lightgbm : pd.DataFrame, opcional
        Conjunto de teste OOT (Out-of-Time) para o modelo LightGBM.
    test_oot_regressao : pd.DataFrame, opcional
        Conjunto de teste OOT (Out-of-Time) para o modelo de regressão.
    num_divisoes : int, padrão=10
        Número de quantis a serem calculados.

    Retorno:
    --------
    pd.DataFrame
        DataFrame contendo estatísticas agregadas por quantil.
    """

    # Lista de DataFrames para processamento
    dataframes = {
        "Train LightGBM": train_lightgbm,
        "Test LightGBM": test_lightgbm,
        "Train Regressão": train_regressao,
        "Test Regressão": test_regressao,
    }

    # Adiciona os OOT caso existam
    if test_oot_lightgbm is not None:
        dataframes["OOT LightGBM"] = test_oot_lightgbm
    if test_oot_regressao is not None:
        dataframes["OOT Regressão"] = test_oot_regressao

    # Lista para armazenar os resultados
    resultados = []

    for nome, df in dataframes.items():
        # Ordena pelo score_1 para garantir a correta distribuição dos quantis
        df = df.copy()
        df["quantil"] = pd.qcut(
            df["score_1"], num_divisoes, labels=False, duplicates='drop')

        # Ajusta para que os quantis comecem em 1 em vez de 0
        df["quantil"] += 1

        # Calcula métricas por quantil
        resumo = df.groupby("quantil").agg(
            total_casos=("id", "count"),
            total_mau=("y", "sum")
        ).reset_index()

        # Adiciona colunas derivadas
        resumo["total_bom"] = resumo["total_casos"] - resumo["total_mau"]
        resumo["maus_acumulados"] = resumo["total_mau"].cumsum()
        resumo["percentual_eventos"] = (
            resumo["total_mau"] / resumo["total_mau"].sum()) * 100
        resumo["cumulative_percentual_eventos"] = resumo["maus_acumulados"] / \
            resumo["total_mau"].sum() * 100

        # Cálculo do Gain
        resumo["gain"] = resumo["cumulative_percentual_eventos"]

        # Cálculo do Cumulative Lift
        resumo["cumulative_lift"] = resumo["gain"] / \
            ((resumo["quantil"] / num_divisoes) * 100)

        # Adiciona o nome do dataframe ao resultado
        resumo.insert(0, "nome_dataframe", nome)

        # Renomeia a coluna do quantil para melhor interpretação
        resumo.rename(columns={"quantil": "quantil", "total_casos": "total_casos", "total_mau": "total_mau",
                               "maus_acumulados": "maus_acumulados", "percentual_eventos": "% maus acumulados",
                               "gain": "Gain", "cumulative_lift": "Cumulative Lift"}, inplace=True)

        # Adiciona ao conjunto de resultados
        resultados.append(resumo)

    # Concatena todos os resultados em um único DataFrame
    resultado_final = pd.concat(resultados, ignore_index=True)

    resultado_final = resultado_final[['nome_dataframe', 'quantil', 'total_casos', 'total_mau', 'total_bom',
                                       'maus_acumulados', '% maus acumulados']]

    return resultado_final


def gerar_tabela_avaliacao(
    train_lightgbm: pd.DataFrame,
    test_lightgbm: pd.DataFrame,
    train_regressao: pd.DataFrame,
    test_regressao: pd.DataFrame,
    test_oot_lightgbm: pd.DataFrame = None,
    test_oot_regressao: pd.DataFrame = None,
    num_divisoes: int = 10
) -> pd.DataFrame:
    """
    Gera uma tabela única contendo estatísticas sobre quantis de score para diferentes conjuntos de dados.

    Parâmetros:
    -----------
    train_lightgbm : pd.DataFrame
        Conjunto de treino do modelo LightGBM.
    test_lightgbm : pd.DataFrame
        Conjunto de teste do modelo LightGBM.
    train_regressao : pd.DataFrame
        Conjunto de treino do modelo de regressão.
    test_regressao : pd.DataFrame
        Conjunto de teste do modelo de regressão.
    test_oot_lightgbm : pd.DataFrame, opcional
        Conjunto de teste OOT (Out-of-Time) para o modelo LightGBM.
    test_oot_regressao : pd.DataFrame, opcional
        Conjunto de teste OOT (Out-of-Time) para o modelo de regressão.
    num_divisoes : int, padrão=10
        Número de quantis a serem calculados.

    Retorno:
    --------
    pd.DataFrame
        DataFrame contendo estatísticas agregadas por quantil.
    """

    # Lista de DataFrames para processamento
    dataframes = {
        "Train LightGBM": train_lightgbm,
        "Test LightGBM": test_lightgbm,
        "Train Regressão": train_regressao,
        "Test Regressão": test_regressao,
    }

    # Adiciona os OOT caso existam
    if test_oot_lightgbm is not None:
        dataframes["OOT LightGBM"] = test_oot_lightgbm
    if test_oot_regressao is not None:
        dataframes["OOT Regressão"] = test_oot_regressao

    # Lista para armazenar os resultados
    resultados = []

    for nome, df in dataframes.items():
        # Ordena pelo score_1 para garantir a correta distribuição dos quantis
        df = df.copy()
        df["quantil"] = pd.qcut(
            df["score_1"], num_divisoes, labels=False, duplicates='drop')

        # Ajusta para que os quantis comecem em 1 em vez de 0
        df["quantil"] += 1

        # Calcula métricas por quantil
        resumo = df.groupby("quantil").agg(
            score_0_min=("score_0", "min"),
            score_0_max=("score_0", "max"),
            total_casos=("id", "count"),
            total_mau=("y", "sum")
        ).reset_index()

        # Adiciona colunas derivadas
        resumo["total_bom"] = resumo["total_casos"] - resumo["total_mau"]
        resumo["maus_acumulados"] = resumo["total_mau"].cumsum()
        resumo["% maus acumulados"] = (
            resumo["maus_acumulados"] / resumo["total_mau"].sum()) * 100

        # Cálculo do KS por faixa
        resumo["bons_acumulados"] = resumo["total_bom"].cumsum()
        resumo["% bons acumulados"] = (
            resumo["bons_acumulados"] / resumo["total_bom"].sum()) * 100
        resumo["KS"] = abs(resumo["% maus acumulados"] -
                           resumo["% bons acumulados"])

        # Adiciona o nome do dataframe ao resultado
        resumo.insert(0, "nome_dataframe", nome)

        # Renomeia a coluna do quantil para melhor interpretação
        resumo.rename(columns={"quantil": "quantil", "score_0_min": "score_0 min",
                      "score_0_max": "score_0 max"}, inplace=True)

        # Seleciona apenas as colunas desejadas
        resumo = resumo[['nome_dataframe', 'quantil', 'score_0 min', 'score_0 max', 'total_casos', 'total_mau', 'total_bom',
                         'maus_acumulados', '% maus acumulados', 'KS']]

        # Adiciona ao conjunto de resultados
        resultados.append(resumo)

    # Concatena todos os resultados em um único DataFrame
    resultado_final = pd.concat(resultados, ignore_index=True)

    return resultado_final


def calcular_ks_por_safra(base_escorada: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o valor máximo da estatística KS (Kolmogorov-Smirnov) para cada safra em um DataFrame.

    Parâmetros:
    base_escorada (pd.DataFrame): DataFrame contendo as colunas ['id', 'safra', 'y', 'score_1', 'score_0'].

    Retorna:
    pd.DataFrame: DataFrame com as colunas ['safra', 'contagem_de_linhas', 'ks_max', 'ponto_ks'].
    """

    def calcular_ks(df: pd.DataFrame) -> Tuple[float, float]:
        """
        Calcula o KS máximo de um DataFrame contendo colunas:
        ['id', 'safra', 'y', 'score_1', 'score_0'].

        Retorna o KS máximo (em percentual) e o ponto onde ele ocorre.
        """
        df = df.sort_values(
            by='score_1', ascending=False)  # Ordena pelo score_1 em ordem decrescente

        total_eventos = df['y'].sum()
        total_nao_eventos = (df['y'] == 0).sum()

        # Evita divisão por zero
        if total_eventos == 0 or total_nao_eventos == 0:
            return 0.0, np.nan

        df['acumulado_eventos'] = df['y'].cumsum() / total_eventos
        df['acumulado_nao_eventos'] = (
            (df['y'] == 0).cumsum()) / total_nao_eventos

        df['diferença'] = abs(df['acumulado_eventos'] -
                              df['acumulado_nao_eventos'])

        ks_max = df['diferença'].max() * 100  # Convertendo KS para percentual

        # Garantindo que ponto_ks seja um único valor
        ponto_ks = df.loc[df['diferença'] == df['diferença'].max(), 'score_1']
        ponto_ks = ponto_ks.iloc[0] if not np.isscalar(ponto_ks) else ponto_ks

        return ks_max, ponto_ks

    resultados = []

    for safra, grupo in base_escorada.groupby('safra', observed=True):
        ks_max, ponto_ks = calcular_ks(grupo)
        resultados.append([safra, len(grupo), ks_max, ponto_ks])

    tabela_resultados = pd.DataFrame(
        resultados, columns=['safra', 'contagem_de_linhas', 'ks_max', 'ponto_ks'])

    # Garantir que a safra seja ordenada corretamente como categoria
    tabela_resultados['safra'] = pd.Categorical(
        tabela_resultados['safra'], ordered=True)
    tabela_resultados = tabela_resultados.sort_values(by='safra')

    return tabela_resultados


def plotar_ks_safra(tabela_resultados: pd.DataFrame) -> None:
    """
    Gera um gráfico de barras mostrando a volumetria por safra e,
    no eixo secundário, o valor do KS por safra (em percentual).

    Parâmetros:
    tabela_resultados (pd.DataFrame): DataFrame com as colunas ['safra', 'contagem_de_linhas', 'ks_max', 'ponto_ks'].
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Gráfico de barras para volumetria
    ax1.bar(tabela_resultados['safra'].astype(str), tabela_resultados['contagem_de_linhas'],
            color='skyblue', label='Volumetria por Safra')
    ax1.set_xlabel('Safra')
    ax1.set_ylabel('Contagem de Linhas', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Criar um segundo eixo para o KS
    ax2 = ax1.twinx()
    ax2.plot(tabela_resultados['safra'].astype(str), tabela_resultados['ks_max'],
             color='red', marker='o', label='KS Máximo')
    ax2.set_ylabel('KS Máximo (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 100)  # Garantindo que o eixo do KS vá de 0 a 100

    # Título e legenda
    plt.title('Volumetria por Safra e KS Máximo (%)')
    fig.tight_layout()
    plt.show()


def calcular_psi(safra_referencia: pd.Series, safra_atual: pd.Series, bins: int = 10) -> Tuple[float, pd.DataFrame]:
    """
    Calcula o Population Stability Index (PSI) para uma variável contínua.

    Parâmetros:
    - safra_referencia (pd.Series): Série de valores da safra de referência.
    - safra_atual (pd.Series): Série de valores da safra atual.
    - bins (int): Número de faixas para discretizar os dados (padrão=10).

    Retorna:
    - Tuple[float, pd.DataFrame]: PSI total e um DataFrame com os detalhes por bin.
    """

    # Criar bins baseados na safra de referência
    bins_edges = np.linspace(safra_referencia.min(),
                             safra_referencia.max(), bins + 1)

    # Contar os valores dentro de cada bin
    ref_counts, _ = np.histogram(safra_referencia, bins=bins_edges)
    atual_counts, _ = np.histogram(safra_atual, bins=bins_edges)

    # Converter para proporções
    ref_props = ref_counts / ref_counts.sum()
    atual_props = atual_counts / atual_counts.sum()

    # Evitar divisão por zero (substituir 0 por um valor mínimo)
    ref_props = np.where(ref_props == 0, 0.0001, ref_props)
    atual_props = np.where(atual_props == 0, 0.0001, atual_props)

    # Calcular PSI para cada bin
    psi_values = (ref_props - atual_props) * np.log(ref_props / atual_props)

    # PSI total
    psi_total = psi_values.sum()

    # Criar DataFrame com os resultados
    psi_df = pd.DataFrame({
        'Bin': [f'{round(bins_edges[i], 2)} - {round(bins_edges[i+1], 2)}' for i in range(bins)],
        'Ref_Proporção': ref_props,
        'Atual_Proporção': atual_props,
        'PSI_Bin': psi_values
    })

    return psi_total, psi_df


def monitorar_variaveis_continuas(
    safra_referencia: pd.DataFrame, safra_atual: pd.DataFrame, colunas_numericas: List[str],
    psi_threshold: float = 0.1, ks_threshold: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """
    Monitora a estabilidade das variáveis contínuas entre duas safras usando PSI e KS Test.

    Parâmetros:
    - safra_referencia (pd.DataFrame): DataFrame com os dados da safra de referência.
    - safra_atual (pd.DataFrame): DataFrame com os dados da safra atual.
    - colunas_numericas (List[str]): Lista de colunas numéricas a serem monitoradas.
    - psi_threshold (float): Limiar para considerar PSI significativo (padrão=0.1).
    - ks_threshold (float): Limiar para considerar KS Test significativo (padrão=0.05).

    Retorna:
    - Dict[str, Dict[str, float]]: Dicionário com variáveis que tiveram mudanças significativas.
    """

    alertas = {'psi': {}, 'ks': {}}

    for col in colunas_numericas:
        # Calcular PSI
        psi_total, _ = calcular_psi(safra_referencia[col], safra_atual[col])

        # Aplicar KS Test
        stat, p_value = ks_2samp(
            safra_referencia[col].dropna(), safra_atual[col].dropna())

        # Verificar se o PSI indica mudança significativa
        if psi_total >= psi_threshold:
            alertas['psi'][col] = {
                'PSI': psi_total,
                'Alerta': 'Mudança Moderada' if 0.1 <= psi_total < 0.25 else 'Mudança Significativa'
            }

        # Verificar se o KS Test indica mudança significativa
        if p_value < ks_threshold:
            alertas['ks'][col] = {
                'KS_Stat': stat,
                'p_value': p_value,
                'Alerta': 'Mudança Significativa'
            }

    psi = pd.DataFrame.from_dict(alertas['psi'], orient='index')

    ks = pd.DataFrame.from_dict(alertas['ks'], orient='index')

    return psi, ks


def obter_importancia_variaveis(modelo, nome_modelo=""):
    """
    Obtém a importância das variáveis de um modelo PyCaret.

    Parâmetros:
    modelo: Modelo treinado pelo PyCaret.
    nome_modelo: Nome do modelo (opcional, apenas para identificação).

    Retorna:
    DataFrame com as colunas 'nome_variavel' e 'importancia', ordenado do maior para o menor.
    """
    if hasattr(modelo, 'feature_importances_'):  # LightGBM e outros modelos baseados em árvores
        importancia = modelo.feature_importances_
        variaveis = modelo.feature_name_ if hasattr(
            modelo, 'feature_name_') else range(len(importancia))

    elif hasattr(modelo, 'coef_'):  # Modelos lineares como regressão logística
        importancia = modelo.coef_.ravel()  # Mantendo os valores originais dos betas
        variaveis = modelo.feature_names_in_ if hasattr(
            modelo, 'feature_names_in_') else range(len(importancia))

        # Criar DataFrame e ordenar pelo valor absoluto dos coeficientes, mas mantendo os sinais originais
        df_importancia = pd.DataFrame(
            {'nome_variavel': variaveis, 'importancia': importancia})
        df_importancia['importancia_abs'] = df_importancia['importancia'].abs()
        df_importancia = df_importancia.sort_values(by="importancia_abs", ascending=False).drop(
            columns=['importancia_abs']).reset_index(drop=True)

        return df_importancia

    else:
        raise ValueError(
            f"O modelo {nome_modelo} não possui um método de importância de variáveis.")

    df_importancia = pd.DataFrame(
        {'nome_variavel': variaveis, 'importancia': importancia})
    df_importancia = df_importancia.sort_values(
        by="importancia", ascending=False).reset_index(drop=True)

    return df_importancia


def calcular_metricas_multiplas(bases_escoradas: Dict[str, pd.DataFrame], limiar: float = 0.5) -> pd.DataFrame:
    """
    Calcula métricas de avaliação para um dicionário de DataFrames contendo bases escoradas.

    Parâmetros:
    -----------
    bases_escoradas : dict[str, pd.DataFrame]
        Dicionário onde:
        - As chaves são os nomes das bases.
        - Os valores são DataFrames com as colunas:
            - 'id': Identificador único.
            - 'safra': Período de referência.
            - 'y': Variável alvo (0 ou 1).
            - 'score_1': Probabilidade prevista da classe positiva.
            - 'score_0': Probabilidade prevista da classe negativa.

    limiar : float, opcional (default=0.5)
        Valor de corte para classificar as previsões. Valores acima do limiar são considerados positivos.

    Retorna:
    --------
    pd.DataFrame:
        DataFrame contendo as métricas para cada base no dicionário:
        - Nome da Base
        - Acurácia
        - Precisão
        - Recall
        - F1-score
        - AUC (Área sob a curva ROC)
        - KS MAX (Kolmogorov-Smirnov)
        - GINI
        - Verdadeiros Positivos (TP)
        - Falsos Positivos (FP)
        - Verdadeiros Negativos (TN)
        - Falsos Negativos (FN)
    """

    # Lista para armazenar os resultados
    resultados = []

    # Percorre cada DataFrame no dicionário
    for nome_base, base_escorada in bases_escoradas.items():
        # Verifica se o elemento é realmente um DataFrame
        if not isinstance(base_escorada, pd.DataFrame):
            raise TypeError(
                f"O valor associado a '{nome_base}' não é um DataFrame. Recebido: {type(base_escorada)}")

        # Verifica se as colunas necessárias estão presentes
        colunas_necessarias = {'id', 'safra', 'y', 'score_1', 'score_0'}
        if not colunas_necessarias.issubset(base_escorada.columns):
            raise ValueError(
                f"O DataFrame '{nome_base}' deve conter as colunas {colunas_necessarias}")

        # Obtendo os valores reais (y) e as previsões baseadas no limiar
        y_true = base_escorada['y']
        y_pred = (base_escorada['score_1'] >= limiar).astype(int)
        # Probabilidades da classe positiva
        y_scores = base_escorada['score_1']

        # Calculando métricas básicas
        acuracia = accuracy_score(y_true, y_pred)
        precisao = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Matriz de confusão
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Cálculo do AUC (Área sob a curva ROC)
        auc = roc_auc_score(y_true, y_scores)

        # KS MAX (Kolmogorov-Smirnov)
        ks_stat = ks_2samp(y_scores[y_true == 1],
                           y_scores[y_true == 0]).statistic

        # GINI = 2 * AUC - 1
        gini = 2 * auc - 1

        # Adiciona os resultados na lista
        resultados.append({
            "Nome da Base": nome_base,
            "Acurácia": round(acuracia, 4),
            "Precisão": round(precisao, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4),
            "AUC": round(auc, 4),
            "KS MAX": round(ks_stat, 4),
            "GINI": round(gini, 4),
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        })

    # Converte a lista de resultados para um DataFrame e retorna
    return pd.DataFrame(resultados)


def train_woe_binning(
    df: pd.DataFrame,
    variables: List[str],
    target: str,
    time_variable: str,
    n_clusters: int = 4,
    plot_vars: List[str] = None
) -> Tuple[pd.DataFrame, Dict]:

    df_result = df.copy()
    df_result[time_variable] = df_result[time_variable].astype(
        str)  # Garantindo que safra seja categórica
    binning_rules = {}

    for variable in variables:
        # 1️⃣ Treinando o binning de WOE
        binning = OptimalBinning(name=variable, dtype="numerical", solver="cp")
        binning.fit(df[variable], df[target])

        # Aplicando o binning para obter os bins
        df_result[f"{variable}_bin"] = binning.transform(
            df[variable], metric="bins")

        # 2️⃣ Criando DataFrame para análise de estabilidade
        stability_df = df_result.groupby(
            [time_variable, f"{variable}_bin"]).size().unstack().fillna(0)
        stability_df = stability_df.div(stability_df.sum(axis=1), axis=0)

        # Ordenando `safra` para manter a sequência correta no gráfico
        stability_df = stability_df.sort_index()

        # 🚨 Ajustando o número de clusters para não exceder a quantidade de bins
        num_bins = len(stability_df.columns)
        # Garante que n_clusters não seja maior que o número de bins
        adjusted_clusters = min(n_clusters, num_bins)

        if adjusted_clusters < 2:
            print(
                f"⚠️ Aviso: {variable} tem apenas {num_bins} bins. Não será clusterizada.")
            bin_map = {bin_label: "Bin_1" for bin_label in stability_df.columns}
        else:
            # 3️⃣ Clusterizando bins com comportamento semelhante
            kmeans = KMeans(n_clusters=adjusted_clusters,
                            random_state=42, n_init=10)
            bin_clusters = kmeans.fit_predict(stability_df.T)
            bin_map = {bin_label: f"Bin_{cluster+1}" for bin_label,
                       cluster in zip(stability_df.columns, bin_clusters)}

        df_result[f"{variable}_bin_group"] = df_result[f"{variable}_bin"].map(
            bin_map)

        # 4️⃣ Criando as regras de transformação
        bin_edges = binning.splits
        bin_labels = sorted(
            df_result[f"{variable}_bin"].dropna().unique())  # Removendo NaNs
        bin_cluster_map = {bin_: bin_map[bin_] for bin_ in bin_labels}

        binning_rules[variable] = {
            "edges": bin_edges,
            "labels": bin_labels,
            "bin_to_group": bin_cluster_map
        }

        # 5️⃣ Plotando estabilidade antes e depois do agrupamento
        if plot_vars is None or variable in plot_vars:
            stability_grouped_df = df_result.groupby(
                [time_variable, f"{variable}_bin_group"]).size().unstack().fillna(0)
            stability_grouped_df = stability_grouped_df.div(
                stability_grouped_df.sum(axis=1), axis=0)

            # Ordenando a safra para melhor visualização
            stability_grouped_df = stability_grouped_df.sort_index()

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            stability_df.plot(ax=axes[0], marker='o')
            axes[0].set_title(
                f"Distribuição de {variable} ao Longo do Tempo (Antes do Agrupamento)")
            axes[0].set_ylabel("Proporção")
            axes[0].set_xlabel(time_variable)

            stability_grouped_df.plot(ax=axes[1], marker='o', cmap="tab10")
            axes[1].set_title(
                f"Distribuição de {variable} ao Longo do Tempo (Depois do Agrupamento)")
            axes[1].set_ylabel("Proporção")
            axes[1].set_xlabel(time_variable)

            plt.tight_layout()
            plt.show()

    return df_result, binning_rules
