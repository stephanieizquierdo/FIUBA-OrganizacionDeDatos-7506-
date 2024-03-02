import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import requests

################ CONFIGURACIONES INICIALES ################

def download_raw_data():
    #descargamos datasets
    with requests.get(
            "https://docs.google.com/spreadsheets/d/1-DWTP8uwVS-dZY402-dm0F9ICw_6PNqDGLmH0u8Eqa0/export?format=csv"
    ) as r, open("impuestos_train.cvs", "wb") as f:
        for chunk in r.iter_content():
            f.write(chunk)
    with requests.get(
            "https://docs.google.com/spreadsheets/d/1ObsojtXfzvwicsFieGINPx500oGbUoaVTERTc69pzxE/export?format=csv"
    ) as r, open("impuestos_holdout.cvs", "wb") as f:
        for chunk in r.iter_content():
            f.write(chunk)

            
def prepare_holdout_dataset(holdout_df: pd.DataFrame):
    """ Prepara el set de holdout"""
    holdout_df.drop(columns='representatividad_poblacional', inplace=True)
    return holdout_df

################ AUXILIARES ################

def stratify_years_studied(number_of_years):
    if number_of_years <= 1:
        return 'jardin'
    elif number_of_years <= 9:
        return 'primaria'
    elif number_of_years <= 14:
        return 'secundaria'
    return 'universitaria'


def encode_to_numeric(df, columns):
    for column in columns:
        le = preprocessing.LabelEncoder()
        le.fit(list(set(list(df[column].unique()))))
        df[column] = le.transform(df[column])
    return df


def age_to_decile(age):
    ranges = [0, 22, 26, 30, 33, 37, 41, 45, 50, 58]
    for i in range(len(ranges) -1):
        if ranges[i] <= age <= ranges[i+1]:
            return i+1
    return 10


def stock_profit_to_quartile(profit):
    bins_profit = [0, 3411.0, 7298.0, 14084]
    bins_loss = [-1977.0, -1887.0, -1672.0, 0]

    if profit == 0:
        return 0
    if profit > 0:
        for i in range(len(bins_profit) - 1):
            if bins_profit[i] <= profit <= bins_profit[i + 1]:
                return i + 1
        return 4
    if profit < 0:
        for i in range(len(bins_loss) - 1):
            if profit < bins_loss[0]:
                return -4
            if bins_loss[i] <= profit <= bins_loss[i + 1]:
                return - 4 + (i + 1)


def working_years_higher_than_27(working_years):
    return working_years > 27


################ PREPROCESSING ################
def encoding_sorted(df, columns):

    educacion_dict = {"jardin": 0, "primaria": 1, "secundaria": 2, "universitaria": 3}
    if "educacion" in columns:
        df["educacion"] = df['anios_estudiados'].apply(stratify_years_studied).apply(lambda x: educacion_dict[x])

    if "edad" in columns:
        df["edad"] = df["edad"].apply(lambda x: age_to_decile(x))

    if "bolsa" in columns:
        df["cuartil_bolsa"] = df["ganancia_perdida_declarada_bolsa_argentina"].apply(lambda x: stock_profit_to_quartile(x))


    return df


def encode_one_hot(df, columns):
    for column in columns:
        one_hot = pd.get_dummies(df[column])
        # Drop column B as it is now encoded
        df = df.drop(column, axis=1)
        # Join the encoded df
        df = df.join(one_hot)
    return (df)


def add_features(df, feature_names):
    if "working_years_higher_than_27" in feature_names:
        df["working_years_higher_than_27"] = df["horas_trabajo_registradas"].apply(working_years_higher_than_27)

    return df

def prepare_existing_data(df1, df2):
    # llenamos nans de trabajo, barrio y categoria de trabajo
    #reemplazamos valor "otro" de religion por "otra_religion" ya que hay otra columna que tiene un valor con el mismo nombre
    df1['trabajo'].fillna('No responde_trabajo', inplace=True)
    df1['barrio'].fillna('No responde_barrio', inplace=True)
    df1['categoria_de_trabajo'].fillna('No responde_categoria_de_trabajo', inplace=True)
    df1["religion"] = df1["religion"].apply(lambda religion: "otra_religion" if religion == "otro" else religion)
    df1.drop(columns=['educacion_alcanzada', 'barrio'], inplace=True)
    df2['trabajo'].fillna('No responde_trabajo', inplace=True)
    df2['barrio'].fillna('No responde_barrio', inplace=True)
    df2['categoria_de_trabajo'].fillna('No responde_categoria_de_trabajo', inplace=True)
    df2["religion"] = df2["religion"].apply(lambda religion: "otra_religion" if religion == "otro" else religion)
    df2.drop(columns=['representatividad_poblacional', 'educacion_alcanzada', 'barrio'], inplace=True)
    return (df1, df2)


def normalize_columns(data: pd.DataFrame):
    names = data.columns
    normalized_data = preprocessing.normalize(data, axis=1)
    normalized_data = pd.DataFrame(normalized_data, columns=names)

    return normalized_data
