'''______________________________________________________________________________________________________________________________________'''
# Необходимые модули:
'''______________________________________________________________________________________________________________________________________'''
import os
import pickle
import asyncio
import IPython
import datetime
import holidays
import warnings
import statsmodels
'''______________________________________________________________________________________________________________________________________'''
# Необходимые модули под псевдонимами:
'''______________________________________________________________________________________________________________________________________'''
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
'''______________________________________________________________________________________________________________________________________'''
# Необходимые сущности (функции, классы, переменные):
'''______________________________________________________________________________________________________________________________________'''
from scipy import stats
from scipy.signal import periodogram
from datetime import date
from dateutil.rrule import rrule, HOURLY
from xgboost import XGBRegressor, plot_importance
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import MinMaxScaler,  StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
'''______________________________________________________________________________________________________________________________________'''
# Отключение предупреждений:
'''______________________________________________________________________________________________________________________________________'''
pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=FutureWarning)
'''______________________________________________________________________________________________________________________________________'''
# Необходимые переменные:
'''______________________________________________________________________________________________________________________________________'''
# Ключевые признаки сырых данных:
main_cols = {"Datetime", "Name", "IBR_ActualConsumption"}

# Словарь временных интервалов:
time_steps = {
              "hour": 1,
              "day": 24,
              "week": 24 * 7,
              "month": int(24 * 30.4),
              "year": int(24 * 365.25)
             }

# Список уникальных субъектов:
uniq_subjs = [
              "Алтайский край", "Амурская область", "Архангельская область", "Астраханская область",
              "Белгородская область", "Брянская область", "Владимирская область", "Волгоградская область",
              "Вологодская область", "Воронежская область", "Еврейская автономная область", "Забайкальский край",
              "Западный энергорайон Якутии", "Ивановская область", "Иркутская область", "Кабардино-Балкарская Республика",
              "Калининградская область", "Калужская область", "Карачаево-Черкесская Республика", "Кемеровская область - Кузбасс",
              "Кировская область", "Костромская область", "Краснодарский край", "Красноярский край",
              "Курганская область", "Курская область", "Ленинградская область", "Липецкая область",
              "Московская область", "Мурманская область", "Нижегородская область", "Новгородская область",
              "Новосибирская область", "ОЭР Хабаровского края", "Омская область", "Оренбургская область",
              "Орловская область", "Пензенская область", "Пермский край", "Приморский край",
              "Псковская область", "Республика Башкортостан", "Республика Бурятия", "Республика Дагестан",
              "Республика Ингушетия", "Республика Калмыкия", "Республика Карелия", "Республика Коми",
              "Республика Крым", "Республика Марий Эл", "Республика Мордовия", "Республика Северная Осетия-Алания",
              "Республика Татарстан (Татарстан)", "Республика Тыва", "Республика Хакасия", "Ростовская область",
              "Рязанская область", "Самарская область", "Саратовская область", "Свердловская область",
              "Смоленская область", "Ставропольский край", "Тамбовская область", "Тверская область",
              "Томская область", "Тульская область", "Тюменская область", "Удмуртская Республика",
              "Ульяновская область", "Центральный энергорайон Якутии", "Челябинская область", "Чеченская Республика",
              "Чувашская Республика - Чувашия", "Южно-Якутский энергорайон", "Ярославская область"
             ]

# Признаки для обучения и прогнозирования:
data_features = [
                 "time_dummie", "year", "month", "day_of_month",
                 "hour", "day_of_week", "day_of_year", "week_of_year",
                 "quarter", "holiday", "lag_hour", "lag_day",
                 "lag_week", "lag_month", "lag_year"
                ] + uniq_subjs.copy()

# Цвета отображения графиков:
pers_palette = [
                "#ff595e", "#ffca3a", "#8ac926", "#1982c4",
                "#6a4c93"
               ]

# Список дат государственных праздников России:
russ_holidays = [str(i)[5: ] for i in set(holidays.RUS(years=2020).keys())]

# Адрес каталога с файлами данных (относительно базового каталога):
data_url = "../parsed_data/"

feature_data_url = "../../prepare_data/feature_data.gzip"

# Адрес подготовленных данных (относительно базового каталога):
prep_data_url = "../prepare_data/data.gzip"
'''______________________________________________________________________________________________________________________________________'''
# Функции для работы с моделью:
'''______________________________________________________________________________________________________________________________________'''
def save_model(model_name: str, model) -> None:
    """
    
    """

    with open(f"{model_name}.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

def open_model(model_path: str):
    """
    ...
    """

    with open(f"{model_path}.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    return model
'''______________________________________________________________________________________________________________________________________'''
# Функции для работы с прогнозами:
'''______________________________________________________________________________________________________________________________________'''
# def check_res(
#               y_pred: np.array,
#               y_true: np.array,
#               metric_name=None
#              ) -> None:
#     """
#     Вывод значений всех метрик для задачи прогнозирования временных рядов.
    
#     :param y_pred: np.array
#         A predicted values of the target variable.

#     :param y_true: np.array
#         A real values of the target variable.

#     :return: None.
#     """

#     print(f"""Значения метрик:
#                               MSE - {mean_squared_error(y_true, y_pred)}
#                               MAE - {mean_absolute_error(y_true, y_pred)}
#                               MAPE - {mean_absolute_percentage_error(y_true, y_pred)}"""
#          )
'''______________________________________________________________________________________________________________________________________'''
# Функции для визуализации данных:
'''______________________________________________________________________________________________________________________________________'''
# def plot_periodogram(time_series, detrend="linear", ax=None):
#     fs = pd.Timedelta("1Y") / (24 * pd.Timedelta("1H"))
#     freqencies, spectrum = periodogram(
#                                        time_series,
#                                        fs=fs,
#                                        detrend=detrend,
#                                        window="boxcar",
#                                        scaling='spectrum',
#                                       )
    
#     if ax is None:
#         _, ax = plt.subplots()
    
#     ax.step(freqencies, spectrum, color="blue")
#     ax.set_xscale("log")
#     ax.set_xticks([1, 2, 4, 12])
#     ax.set_xticklabels(
#         [
#          "Year (1)",
#          "Semiannual (2)",
#          "Quarterly (4)",
#          "Monthly (12)",
#         ],
#         rotation=90,
#     )
    
#     plt.figure(figsize=(20, 8))
#     ax.ticklabel_format(
#                         axis="y",
#                         style="sci",
#                         scilimits=(0, 0)
#                        )
    
#     ax.set_ylabel("Variance")
#     ax.set_title("Periodogram")
    
#     return ax
'''______________________________________________________________________________________________________________________________________'''
# Функции для работы с признаками данных:
'''______________________________________________________________________________________________________________________________________'''
# Функции создания признаков:
def get_year(curr_dt: str) -> int:
    """
    Возвращает год из строки даты со временем.

    :param cur_datetime: str
        A datetime string.

    :return: int
        A year.
    """

    return int(curr_dt.replace(' ', '-').split('-')[0])

def get_month(curr_dt: str) -> int:
    """
    Select a month from datetime string.

    :param cur_datetime: str
        A datetime string.

    :return: int
        A month.
    """

    return int(curr_dt.replace(' ', '-').split('-')[1])

def get_day_month(curr_dt: str) -> int:
    """
    Select a day of month from datetime string.

    :param cur_datetime: str
        A datetime string.

    :return: int
        A day of month.
    """

    return int(curr_dt.replace(' ', '-').split('-')[2])

def get_hour(curr_dt: str) -> int:
    """
    Select an hour from datetime string.

    :param cur_datetime: str
        A datetime string.

    :return: int
        An hour.
    """

    return int(curr_dt.replace(' ', '-').split('-')[3][:2])

def get_day_week(curr_dt: str) -> int:
    """
    Returns a day of week from datetime string.

    :param cur_datetime: str
        A datetime string.

    :return: int
        A day of week.
    """

    return datetime.datetime(
                             year=get_year(curr_dt),
                             month=get_month(curr_dt),
                             day=get_day_month(curr_dt)
                            ).weekday() + 1


def get_day_year(curr_dt: str) -> int:
    """
    Returns a day of year from datetime string.

    :param cur_datetime: str
        A datetime string.

    :return: int
        A day of year.
    """

    return datetime.datetime(
                             year=get_year(curr_dt),
                             month=get_month(curr_dt),
                             day=get_day_month(curr_dt)
                            ).timetuple().tm_yday


def get_week_year(curr_dt: str) -> int:
    """
    Returns a week of year from datetime string.

    :param cur_datetime: str
        A datetime string.

    :return: int
        A week of year.
    """

    return datetime.datetime(
                             year=get_year(curr_dt),
                             month=get_month(curr_dt),
                             day=get_day_month(curr_dt)
                            ).isocalendar()[1]


def get_quarter(curr_month: int) -> int:
    """
    Возвращает квартал.

    :param month: month
        A month of sample.

    :return: int
        An quarter.
    """

    if curr_month in {1, 2, 12}:
        return 4
    elif curr_month in {3, 4, 5}:
        return 1
    elif curr_month in {6, 7, 8}:
        return 2
    else:
        return 3


def is_holiday(curr_dt: str) -> int:
    """
    Returns flag of that date is holiday.

    :param cur_datetime: str
        A cur_datetime string.

    :return: int
        A holiday flag.
    """

    return int(curr_dt.split()[0][5:] in russ_holidays)

def create_lag_feature(df: pd.DataFrame, step_count: int) -> pd.Series:
    """
    ...
    """
    
    temp_df = pd.DataFrame()

    for subj in df["subject_name"].unique():
        subj_df = df[df["subject_name"] == subj].copy()
        subj_df["actual_consumption"] = subj_df["actual_consumption"].shift(step_count)
        temp_df = pd.concat([temp_df, subj_df], axis=0) 
        #         temp_df = temp_df.append(subj_df)
        
    temp_df.sort_values(by=["datetime", "subject_name"], inplace=True)
    
    return temp_df["actual_consumption"]


def create_time_dummie_feature(df: pd.DataFrame) -> pd.Series:
    """
    
    """
    
    temp_df = pd.DataFrame()

    for subj in df["subject_name"].unique():
        subj_df = df[df["subject_name"] == subj].copy()
        subj_df["time_dummie"] = range(1, subj_df.shape[0] + 1)
        temp_df = pd.concat([temp_df, subj_df], axis=0) 
#         temp_df.append(subj_df)
        
    temp_df.sort_values(by=["datetime", "subject_name", "time_dummie"], inplace=True)
    
    return temp_df["time_dummie"]

def create_time_step_features(df):
    df["time_dummie"] = create_time_dummie_feature(df)
    df["year"] = df["datetime"].apply(get_year)
    df["month"] = df["datetime"].apply(get_month)
    df["day_of_month"] = df["datetime"].apply(get_day_month)
    df["hour"] = df["datetime"].apply(get_hour)
    df["day_of_week"] = df["datetime"].apply(get_day_week)
    df["day_of_year"] = df["datetime"].apply(get_day_year)
    df["week_of_year"] = df["datetime"].apply(get_week_year)
    df["quarter"] = df["datetime"].apply(get_quarter)
    df["holiday"] = df["datetime"].apply(is_holiday)

    return df
'''______________________________________________________________________________________________________________________________________'''
# Функции для преобразования данных:
'''______________________________________________________________________________________________________________________________________'''
def create_forecast_data(
                         step_count: int,
                         step_val: str,
                         main_df_path: str
                        ) -> pd.DataFrame | None:
    """
    Возвращает dataframe за требуемый период.
    Требуется указать: количество количество временных интервалов (дальность прогнозирования),
    размер временного интервала,
    адрес dataframe- с последними данными по обучению модели (нужно для получение последний актуальной даты модели)
    
    ...
    """

    if step_val.lower() in ['d', "day"]:
        step_count *= 24
    elif step_val.lower() in ['w', "week"]:
        step_count *= 24 * 7
    elif step_val.lower() in ['m', "month"]:
        step_count *= 24 * 30
    elif step_val.lower() in ['y', "year"]:
        step_count *= 24 * 30 * 12
    else:
        return None

    forecast_data = pd.DataFrame()
    main_df = pd.read_parquet(main_df_path)
    last_main_df_dt = datetime.datetime(
                                        int(main_df.iloc[-1]["year"]),
                                        int(main_df.iloc[-1]["month"]),
                                        int(main_df.iloc[-1]["day_of_month"]),
                                        int(main_df.iloc[-1]["hour"])
                                       )

    for dt in rrule(
                    HOURLY,
                    dtstart=last_main_df_dt,
                    count=step_count
                   ):
        curr_df = pd.DataFrame()
        subjs = pd.Series(uniq_subjs)
        hour_consumpt = np.array([dt] * len(subjs))
        actual_consumpt = np.full((len(subjs), 1), np.nan)

        curr_df["subject_name"] = subjs
        curr_df["datetime"] = hour_consumpt
        curr_df["actual_consumption"] = actual_consumpt

        forecast_data = pd.concat([forecast_data, curr_df], axis=0)

    forecast_data.reset_index(drop=True, inplace=True)
    forecast_data.sort_values(by=["datetime", "subject_name"])

    return forecast_data


def make_subj_pred(
                   step_count: int,
                   step_val: str,
                   main_df_path: str,
                   model_path: str,
                   subj_name: str = None
                  ):
    """
    Прогнозирует энергопотребление для конкретного субъекта на n временных интервалов вперёд.

    :param step_count: int
        Количество временных шагов (дальность) для прогнозирования.
    :param date: datetime
        A date of the request.
    :param case: int [1]
        A parsing case number.

    :return: list
        A list of dictionaries with data.
    """
    
    model = open_model(model_path)
    forecast_data = create_forecast_data(step_count, step_val, main_df_path)
    feature_forecast_data = prep_forecast_data(forecast_data, main_df_path)
    predictions = model.predict(feature_forecast_data[data_features])
    feature_forecast_data.reset_index(drop=True, inplace=True)
    data = pd.concat([feature_forecast_data, pd.Series(predictions, name="actual_consumption")], axis=1)
    
    if subj_name is None:
        return data
    else:
        return data[data[subj_name] == 1]


def find_index_with_one(row):
    for i, val in enumerate(row):
        if val == 1:
            return i
    return None


def decoding(encoding_columns):
    zxc = encoding_columns.apply(find_index_with_one, axis=1)
    tmp = uniq_subjs
    
    return [tmp[i] for i in zxc]


def prep_forecast_data(
                       df: pd.DataFrame,
                       main_df_path: str
                      ) -> pd.DataFrame:
    """
    Подготавливает данные для прогнозирования.
    Подготовка подразумевает под собой: создание фич, сортировка
    Требуются данные в формате dataframe со следующими колонками: "subject_name", "datetime", "actual_consumption".

    ...
    """

    main_df = pd.read_parquet(main_df_path)

    forecast_data = df.copy()
    forecast_data.sort_values(by=["subject_name", "datetime"], inplace=True)
    forecast_data.drop_duplicates(inplace=True)

    count_forecast_data_rows = forecast_data.shape[0]

    forecast_data["datetime"] = forecast_data["datetime"].astype(str)

    forecast_data = create_time_step_features(forecast_data)  ###
    forecast_data = pd.concat([forecast_data, pd.get_dummies(forecast_data["subject_name"], columns=uniq_subjs)], axis=1)
    forecast_data_columns = forecast_data.columns
    
    main_df["subject_name"] = decoding(main_df[uniq_subjs])
    
    main_df["datetime"] = main_df.index
    #
    forecast_data = pd.concat(
                              [main_df[forecast_data_columns], forecast_data],
                              axis=0,
                              ignore_index=True
                             )

    forecast_data["lag_hour"] = create_lag_feature(forecast_data, time_steps["hour"])
    forecast_data["lag_day"] = create_lag_feature(forecast_data, time_steps["day"])
    forecast_data["lag_week"] = create_lag_feature(forecast_data, time_steps["week"])
    forecast_data["lag_month"] = create_lag_feature(forecast_data, time_steps["month"])
    forecast_data["lag_year"] = create_lag_feature(forecast_data, time_steps["year"])

    forecast_data = forecast_data[forecast_data["day_of_week"] < 6]
     
    forecast_data.sort_values(by=["datetime", "subject_name", "time_dummie"], inplace=True)
    forecast_data.drop(columns=["datetime", "subject_name", "actual_consumption"], inplace=True)
    forecast_data.dropna(inplace=True)
    
    return forecast_data.iloc[-count_forecast_data_rows:]


# def prepare_df(df_name: str, data_format: str) -> pd.DataFrame:
#     # Загружаем всё:
#     base_df = pd.read_parquet("../../prepare_data/feature_data.gzip")
#     test_df = pd.read_csv(f"../../parsed_data/{df_name}.{data_format}")
    
#     test_df.rename(columns={
#                    "Name": "subject_name",
#                    "Datetime": "datetime",
#                    "IBR_ActualConsumption": "actual_consumption"
#                   }, inplace=True)
    
#     test_df.sort_values(by=["subject_name", "datetime"], inplace=True)
    
#     test_df["actual_consumption"] = test_df["actual_consumption"].apply(to_int)
#     test_df.drop_duplicates(inplace=True)
#     test_df.dropna(inplace=True)

#     test_df = test_df[["subject_name", "datetime", "actual_consumption"]].copy()
    
#     n = test_df.shape[0]
    
#     # Тут всё:
    
#     test_df = create_time_step_features(test_df)
#     zxc = pd.get_dummies(test_df["subject_name"], columns=Subj_req)
#     test_df = pd.concat([test_df, zxc], axis=1)
# #     test_df.drop(columns=["datetime"], inplace=True)
#     rows = test_df.columns
    
#     base_df["subject_name"] = decoding(base_df[uniq_subj])
#     base_df["datetime"] = base_df.index
#     new_df = pd.concat([base_df[rows], test_df], axis=0, ignore_index=True)

#     new_df["lag_hour"] = create_lag_feature(new_df, hour)
#     new_df["lag_day"] = create_lag_feature(new_df, day)
#     new_df["lag_week"] = create_lag_feature(new_df, week)
#     new_df["lag_month"] = create_lag_feature(new_df, month)
#     new_df["lag_year"] = create_lag_feature(new_df, year)

#     new_df = new_df[new_df["day_of_week"] < 6]
     
#     new_df.sort_values(by=["datetime", "subject_name"], inplace=True)
#     new_df.dropna(inplace=True)
#     new_df.drop(columns=["datetime", "subject_name"], inplace=True)
    
#     return new_df.iloc[-n:]


def get_real_values(df, dummies):
    """
    Восстанавливает оригинальные значения из закодированных дамми-переменных.
    
    Параметры:
    df (pd.DataFrame) - исходный DataFrame
    dummies (list) - список названий дамми-переменных
    
    Возвращает:
    pd.DataFrame - DataFrame с восстановленными исходными значениями
    """

    # Создаем копию DataFrame, чтобы не модифицировать исходный
    df_real = df.copy()
    
    # Находим уникальные значения для каждой дамми-переменной
    unique_values = {dummy: df[dummy].unique() for dummy in dummies}
    
    # Для каждой дамми-переменной
    for dummy, values in unique_values.items():
        # Находим индексы, где данная дамми-переменная = 1
        for value in values:
            indexes = df[dummy] == value
            
            # Восстанавливаем исходное значение
            df_real.loc[indexes, dummy] = dummy[1]
        
        # Удаляем дамми-переменную
        df_real = df_real.drop(dummy, axis=1)
    
    return df_real


def prep_target (raw_target: str) -> int | float:
    """
    Преобразует целевую переменную в числовой формат.
    
    :param raw_target: str
        Строка со значением целевой переменной в формате  (или символ '-' при пропуске).

    :return: float.
    """

    return np.nan if '-' in raw_target else int(raw_target.replace(' ', '').rstrip("МВт*ч"))
'''______________________________________________________________________________________________________________________________________'''
    