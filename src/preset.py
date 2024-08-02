# Необходимых модулей для работы:
import holidays
import datetime
import warnings

# Необходимые для работы модули под псевдонимами:
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# Необходимые сущности (функции, классы, переменные):
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV, train_test_split 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# Отключение предупреждений:
pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=FutureWarning)


# Список дат государственных праздников России:
rus_holidays = [str(i)[5:] for i in set(holidays.RUS(years=2020).keys())]

# Адрес каталога с файлами данных (относительно базового каталога):
data_url = "../parsed_data/"

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
    
    
    
    """
    
    print("Mean squared error on test set: ", mean_squared_error(y_true, y_pred))
    print("Mean absolute error on test set: ", mean_absolute_error(y_true, y_pred))
    print("Mean percentage absolute error on test set: ", mean_absolute_percentage_error(y_true, y_pred))
    