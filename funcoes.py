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


def selecionar_variaveis_lightgbm_var_aleatoria(
    df: pd.DataFrame,
    target: str,
    id_column: str,
    ignore_features: List[str],
    percentual_corte: float = 0.05,
    random_var: bool = False,
    session_id: int = 123
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Seleciona as variáveis mais importantes utilizando um modelo LightGBM.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame de entrada contendo as variáveis independentes e a variável alvo.

    target : str
        Nome da variável alvo (coluna de interesse para predição).

    id_column : str
        Nome da coluna identificadora única de cada amostra.

    ignore_features : List[str]
        Lista de colunas a serem ignoradas na modelagem.

    percentual_corte : float, opcional (default=0.05)
        Percentual do valor da maior importância para definir o limiar de corte.
        útil quando random_var=False, utilizamos quando queremos selecioanr variáveis apenas por importância relativa

    random_var: bool = True,
        Insere uma variável aleatória e descarta variáveis que possuem importância menor ou igual a ela.
        Se uma variável aleatória for mais importante que uma variável explicativa, não podemos confiar na variável.

    session_id : int, opcional (default=123)
        ID de sessão para garantir reprodutibilidade na configuração do PyCaret.

    Retorna:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - DataFrame original com apenas as variáveis selecionadas.
        - DataFrame contendo as variáveis mantidas e suas importâncias.
        - DataFrame contendo as variáveis descartadas e suas importâncias.
    """

    # inserindo uma variável aleatória entre [0,1)
    if random_var:
        df['random_var'] = np.random.rand(df.shape[0])

    # Configurando o PyCaret para classificação
    s = setup(df,
              index=id_column,
              ignore_features=ignore_features,
              target=target,
              session_id=session_id,
              verbose=False)

    # Criando e treinando o modelo LightGBM
    light_gbm = create_model('lightgbm',
                             fold=5,
                             # Número máximo de folhas em cada árvore. Controla a complexidade do modelo.
                             num_leaves=31,
                             # Profundidade máxima de cada árvore. Um valor menor ajuda a evitar overfitting.
                             max_depth=6,
                             # Número total de árvores no modelo. Valores maiores aumentam a precisão, mas também o tempo de treino.
                             n_estimators=500,
                             # Taxa de aprendizado. Valores menores tornam o modelo mais estável, mas exigem mais estimadores.
                             learning_rate=0.05,
                             # Fração dos dados usada para treinar cada árvore. Evita overfitting e melhora a generalização.
                             subsample=0.8,
                             # Proporção de features usadas em cada árvore. Valores menores reduzem overfitting.
                             colsample_bytree=0.8,
                             # Número mínimo de amostras necessárias para dividir um nó. Evita divisões muito pequenas e overfitting.
                             min_child_samples=20,
                             # Regularização L2 (Ridge). Ajuda a controlar overfitting aumentando a penalização de pesos elevados.
                             reg_lambda=1,
                             # Regularização L1 (Lasso). Zera coeficientes de features menos importantes para melhorar a interpretabilidade.
                             reg_alpha=0.1,
                             verbose=False)

    # Criando um DataFrame com a importância das features
    feature_importance = pd.DataFrame({
        'Nome_Variavel': df.drop(columns=[target, id_column] + ignore_features).columns,
        'Importancia_Variavel': light_gbm.feature_importances_
    })

    # Ordenando da maior para a menor importância
    feature_importance = feature_importance.sort_values(
        by='Importancia_Variavel', ascending=False)

    # Encontrando a maior importância e definindo o limiar de corte
    max_importancia = feature_importance['Importancia_Variavel'].max()
    limiar_importancia = max_importancia * percentual_corte

    # valor limite da variável randômica
    if random_var:
        valor_limite_var_random = feature_importance.loc[feature_importance["Nome_Variavel"]
                                                         == "random_var", "Importancia_Variavel"].values[0]

    # Separando variáveis mantidas e descartadas com base no limiar

    if random_var:
        variaveis_mantidas = feature_importance[feature_importance['Importancia_Variavel']
                                                > valor_limite_var_random]
        variaveis_descartadas = feature_importance[feature_importance['Importancia_Variavel']
                                                   <= valor_limite_var_random]

    else:
        variaveis_mantidas = feature_importance[feature_importance['Importancia_Variavel']
                                                >= limiar_importancia]
        variaveis_descartadas = feature_importance[feature_importance['Importancia_Variavel']
                                                   < limiar_importancia]

    # Criando um novo DataFrame apenas com as variáveis mantidas
    colunas_mantidas = [id_column, target] + ignore_features + \
        variaveis_mantidas['Nome_Variavel'].tolist()
    df_selecionado = df[colunas_mantidas]

    # Exibir variáveis descartadas
    if random_var:
        print("\n❌ Importância da variável aleatória:\n", valor_limite_var_random)

    print("\n❌ Variáveis Descartadas:\n", variaveis_descartadas)

    return df_selecionado, variaveis_mantidas, variaveis_descartadas


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


def pipeline_modelagem(train: pd.DataFrame, test: pd.DataFrame, id_col: str, safra_col: str, target_col: str) -> tuple[
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
        train_size=0.8,
        session_id=42,
        verbose=False,
        fold_strategy='stratifiedkfold',
        fold=10,
        normalize=True,
        normalize_method='zscore',
        feature_selection=False,
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
            # Reduzido para evitar complexidade excessiva
            'num_leaves': [2, 5, 10],
            # Taxas de aprendizado menores
            'learning_rate': [0.005, 0.01, 0.05],
            # Evita excesso de árvores
            'n_estimators': [100, 200, 300],
            'max_depth': [3],  # Limitando a profundidade
            'subsample': [0.6, 0.8],  # Introduzindo mais aleatoriedade
            # Evita dependência de features específicas
            'colsample_bytree': [0.6, 0.8],
            'reg_alpha': [0.1, 0.5, 1],  # Regularização L1 mais forte
            'reg_lambda': [0.1, 0.5, 1]  # Regularização L2 mais forte
        },
        optimize='AUC'
    )

    # Hiperparâmetros otimizados para Regressão Logística
    tuned_lr = tune_model(
        create_model('lr'),
        custom_grid={
            'C': [0.01, 0.1, 1, 10],
            'max_iter': [100, 200, 500],
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

    a = train_lightgbm_scored.copy()

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
    return a, train, test, tuned_lightgbm, tuned_lr, train_lightgbm_escorado, test_lightgbm_escorado, train_regressao_escorado, test_regressao_escorado


############################################ FUNCOES DE METRICAS E AVALIACAO DE MODELOS ############################################


def plot_comparacao_roc(
    train_lightgbm: pd.DataFrame,
    test_lightgbm: pd.DataFrame,
    train_regressao: pd.DataFrame,
    test_regressao: pd.DataFrame
) -> None:
    """
    Gera dois gráficos de curva ROC lado a lado para comparar os modelos LightGBM e Regressão Logística.

    O primeiro gráfico contém as curvas ROC do modelo LightGBM para treino e teste.
    O segundo gráfico contém as curvas ROC do modelo de Regressão Logística para treino e teste.

    Parâmetros:
    - train_lightgbm (pd.DataFrame): DataFrame com os dados de treino para o modelo LightGBM.
    - test_lightgbm (pd.DataFrame): DataFrame com os dados de teste para o modelo LightGBM.
    - train_regressao (pd.DataFrame): DataFrame com os dados de treino para o modelo de Regressão Logística.
    - test_regressao (pd.DataFrame): DataFrame com os dados de teste para o modelo de Regressão Logística.

    Retorno:
    - None. A função exibe os gráficos.
    """

    # Criar figura com dois subgráficos
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Lista de dados para iteração
    modelos = [
        ("LightGBM", train_lightgbm, test_lightgbm, axes[0]),
        ("Regressão Logística", train_regressao, test_regressao, axes[1])
    ]

    for nome_modelo, train_df, test_df, ax in modelos:
        # GARANTIR que estamos usando score_1 (probabilidade da classe positiva)
        y_train, y_test = train_df["y"], test_df["y"]
        scores_train, scores_test = train_df["score_1"], test_df["score_1"]

        # Calcular curva ROC para treino
        fpr_train, tpr_train, _ = roc_curve(y_train, scores_train)
        auc_train = auc(fpr_train, tpr_train)

        # Calcular curva ROC para teste
        fpr_test, tpr_test, _ = roc_curve(y_test, scores_test)
        auc_test = auc(fpr_test, tpr_test)

        # Plotar curvas
        ax.plot(fpr_train, tpr_train,
                label=f'Treino (AUC = {auc_train:.2f})', color='blue')
        ax.plot(fpr_test, tpr_test,
                label=f'Teste (AUC = {auc_test:.2f})', color='red')

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
        # Usar 'score_1' como probabilidade da classe positiva
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

        # Plotar curvas para treino e teste
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

    # Exibir gráficos
    plt.tight_layout()
    plt.show()
