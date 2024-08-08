# Необходимых модулей для работы:
import os
import IPython
import holidays
import datetime
import warnings
import statsmodels
import IPython.display

# Необходимые для работы модули под псевдонимами:
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

# Необходимые сущности (функции, классы, переменные):
from scipy import stats
from datetime import date
from dateutil.rrule import rrule, HOURLY
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor, plot_importance
from sklearn.linear_model import LinearRegression
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV, train_test_split
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Отключение предупреждений:
pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=FutureWarning)

# Размеры временных периодов:
hour = 1
day = hour * 24
week = day * 7 
month = day * 30
year = day * 365

Subj_req = ['Алтайский край', 'Амурская область', 'Архангельская область',
       'Астраханская область', 'Белгородская область', 'Брянская область',
       'Владимирская область', 'Волгоградская область', 'Вологодская область',
       'Воронежская область', 'Еврейская автономная область',
       'Забайкальский край', 'Западный энергорайон Якутии',
       'Ивановская область', 'Иркутская область',
       'Кабардино-Балкарская Республика', 'Калининградская область',
       'Калужская область', 'Карачаево-Черкесская Республика',
       'Кемеровская область - Кузбасс', 'Кировская область',
       'Костромская область', 'Краснодарский край', 'Красноярский край',
       'Курганская область', 'Курская область', 'Ленинградская область',
       'Липецкая область', 'Московская область', 'Мурманская область',
       'Нижегородская область', 'Новгородская область',
       'Новосибирская область', 'ОЭР Хабаровского края', 'Омская область',
       'Оренбургская область', 'Орловская область', 'Пензенская область',
       'Пермский край', 'Приморский край', 'Псковская область',
       'Республика Башкортостан', 'Республика Бурятия', 'Республика Дагестан',
       'Республика Ингушетия', 'Республика Калмыкия', 'Республика Карелия',
       'Республика Коми', 'Республика Крым', 'Республика Марий Эл',
       'Республика Мордовия', 'Республика Северная Осетия-Алания',
       'Республика Татарстан (Татарстан)', 'Республика Тыва',
       'Республика Хакасия', 'Ростовская область', 'Рязанская область',
       'Самарская область', 'Саратовская область', 'Свердловская область',
       'Смоленская область', 'Ставропольский край', 'Тамбовская область',
       'Тверская область', 'Томская область', 'Тульская область',
       'Тюменская область', 'Удмуртская Республика', 'Ульяновская область',
       'Центральный энергорайон Якутии', 'Челябинская область',
       'Чеченская Республика', 'Чувашская Республика - Чувашия',
       'Южно-Якутский энергорайон', 'Ярославская область'],

# Список дат государственных праздников России:
rus_holidays = [str(i)[5:] for i in set(holidays.RUS(years=2020).keys())]

# Цвета отображения графиков:
custom_palette = ["#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93"]

# Адрес каталога с файлами данных (относительно базового каталога):
data_url = "../parsed_data/"

feature_data_url = "../../prepare_data/feature_data.gzip"

# Адрес подготовленных данных (относительно базового каталога):
prep_data_url = "../prepare_data/data.gzip"

# Признак для исследования (имеющиеся):
main_columns = {"Datetime", "Name", "IBR_ActualConsumption"}


# Функции создания признаков:
def get_year(cur_datetime: str) -> int:
    """
    Select a year from datetime string.

    :param cur_datetime: str
        A datetime string.

    :return: int
        A year.
    """

    return int(cur_datetime.replace(' ', '-').split('-')[0])


def get_month(cur_datetime: str) -> int:
    """
    Select a month from datetime string.

    :param cur_datetime: str
        A datetime string.

    :return: int
        A month.
    """

    return int(cur_datetime.replace(' ', '-').split('-')[1])


def create_lag_feature(df: pd.DataFrame, step_count: int) -> pd.Series:
    """
    
    """
    
    temp_df = pd.DataFrame()

    for subj in df["subject_name"].unique():
        subj_df = df[df["subject_name"] == subj].copy()
        subj_df["actual_consumption"] = subj_df["actual_consumption"].shift(step_count)
        temp_df = temp_df.append(subj_df)
        
    temp_df.sort_values(by=["datetime", "subject_name"], inplace=True)
    
    return temp_df["actual_consumption"]

def create_time_dummie_feature(df: pd.DataFrame) -> pd.Series:
    """
    
    """
    
    temp_df = pd.DataFrame()

    for subj in df["subject_name"].unique():
        subj_df = df[df["subject_name"] == subj].copy()
        subj_df["time_dummie"] = range(1, subj_df.shape[0] + 1)
        temp_df = temp_df.append(subj_df)
        
    temp_df.sort_values(by=["datetime", "subject_name", "time_dummie"], inplace=True)
    
    return temp_df["time_dummie"]


def get_day_month(cur_datetime: str) -> int:
    """
    Select a day of month from datetime string.

    :param cur_datetime: str
        A datetime string.

    :return: int
        A day of month.
    """

    return int(cur_datetime.replace(' ', '-').split('-')[2])


def get_hour(cur_datetime: str) -> int:
    """
    Select an hour from datetime string.

    :param cur_datetime: str
        A datetime string.

    :return: int
        An hour.
    """

    return int(cur_datetime.replace(' ', '-').split('-')[3][:2])


def get_day_week(cur_datetime: str) -> int:
    """
    Returns a day of week from datetime string.

    :param cur_datetime: str
        A datetime string.

    :return: int
        A day of week.
    """

    return datetime.datetime(
                             year=get_year(cur_datetime),
                             month=get_month(cur_datetime),
                             day=get_day_month(cur_datetime)
                            ).weekday() + 1


def get_day_year(cur_datetime: str) -> int:
    """
    Returns a day of year from datetime string.

    :param cur_datetime: str
        A datetime string.

    :return: int
        A day of year.
    """

    return datetime.datetime(
                             year=get_year(cur_datetime),
                             month=get_month(cur_datetime),
                             day=get_day_month(cur_datetime)
                            ).timetuple().tm_yday


def get_week_year(cur_datetime: str) -> int:
    """
    Returns a week of year from datetime string.

    :param cur_datetime: str
        A datetime string.

    :return: int
        A week of year.
    """

    return datetime.datetime(
                             year=get_year(cur_datetime),
                             month=get_month(cur_datetime),
                             day=get_day_month(cur_datetime)
                            ).isocalendar()[1]


def get_quarter(month: int) -> int:
    """
    Returns an quarter from month number.

    :param month: month
        A month of sample.

    :return: int
        An quarter.
    """

    if month in {1, 2, 12}:
        return 4
    elif month in {3, 4, 5}:
        return 1
    elif month in {6, 7, 8}:
        return 2
    else:
        return 3


def is_holiday(cur_datetime: str) -> int:
    """
    Returns flag of that date is holiday.

    :param cur_datetime: str
        A cur_datetime string.

    :return: int
        A holiday flag.
    """

    return int(cur_datetime.split()[0][5:] in rus_holidays)


# Функции для преобразования данных:
def to_int(sample: str) -> float:
    """
    Convert dataframe field to float.
    
    :param sample: str
        A string row with number (or symbol '-').

    :return: float.
    """

    return np.nan if '-' in sample else int(sample.replace(' ', '').rstrip("МВт*ч"))


# Функция оценки прогноза модели:
def check_res(y_pred: np.array, y_true: np.array) -> None:
    """
    Prints all necessary metrics for time series.
    
    :param y_pred: np.array
        A predicted values of the target variable.

    :param y_true: np.array
        A real values of the target variable.

    :return: None.
    """

    print(f"""Значения метрик:
                              MSE - {mean_squared_error(y_true, y_pred)}
                              MAE - {mean_absolute_error(y_true, y_pred)}
                              MAPE - {mean_absolute_percentage_error(y_true, y_pred)}"""
         )
    