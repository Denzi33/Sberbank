"""
------------------------------------------------------------------------------------------------------------------------
The module contains utils functions.
------------------------------------------------------------------------------------------------------------------------
ASYNC Functions:
------------------------------------------------------------------------------------------------------------------------
    clean_data:
        Description: Deletes unnecessary row columns.

        Parameters:
            data_rows - A list of dictionaries to clean,
            del_columns - A list of columns to delete,
            inplace - A flag of filling in-place.

        Parameters type:
            list | None,
            list | None,
            bool [False].

        Returns: A list of cleared data rows.

        Return type: list | None.

    make_request:
        Description: Makes requests to the specified address.

        Parameters:
            url - A request link,
            verify - A site security flag.

        Parameters type:
            str,
            bool [False].

        Returns: Response as a data dictionary.

        Return type: dict | None.
------------------------------------------------------------------------------------------------------------------------
Functions:
------------------------------------------------------------------------------------------------------------------------
    csv_to_gzip:
        Description: Сonvert .csv files to .gzip files.

        Parameters:
            start_year - The first year for conversion,
            end_year - The last year for conversion,
            rel_path - A relative file path.

        Parameters type:
            int,
            int,
            str.

        Return type: None.
    
    read_json:
        Description: Returns json file data as dictionary.

        Parameters: json_file_name - A json file name.

        Parameters type: str.

        Returns: A data dictionary.

        Return type: dict | None.
------------------------------------------------------------------------------------------------------------------------
"""

# Necessary modules:
import json

# Necessary modules with alias:
import pandas as pd

# Necessary functions:
from itertools import product

# Necessary functions and variables with alias:
from aiohttp import ClientSession as ClSess, TCPConnector as Conn


async def clean_data(
                     data_rows: list | None,
                     del_columns: list | None,
                     inplace: bool = False
) -> list | None:
    """
    Deletes unnecessary row columns.

    :param data_rows: list | None
        A list of dictionaries to clean.
    :param del_columns: list | None
        A list of columns to delete.
    :param inplace: bool [False]
        A flag of filling in-place.

    :return: list | None
        A list of cleared data rows.
    """

    # Inplace functionality case:
    prep_data: list | None = data_rows if inplace is True else data_rows.copy()

    # Delete useless columns:
    [row.pop(column, default=0) for row, column in product(prep_data, del_columns)]

    return prep_data


async def make_request(url: str, verify: bool = False) -> dict | None:
    """
    Makes requests to the specified address.

    :param url: str
        A request link,
    :param verify: bool [False]
        A site security flag.

    :return: dict | None
        Response as a data dictionary.
    """

    # Create session for async requests:
    async with ClSess(connector=Conn(ssl=verify)) as session:
        async with session.get(url) as resp:
            return await resp.json() if resp.status == 200 else None


def csv_to_gzip(
                start_year: int,
                end_year: int,
                rel_path: str
) -> None:
    """
    Convert .csv files to .gzip files.

    :param start_year: int
        The first year for conversion.
    :param end_year: int
        The last year for conversion.
    :param rel_path: str
        A relative file path.

    :return: None
    """

    [pd.read_csv(rel_path + f"{year}.csv").to_parquet(rel_path + f"{year}.gzip") for
     year in range(start_year, end_year)]


def read_json(json_file_name: str) -> dict | None:
    """
    Returns .json file data as dictionary.

    :param json_file_name: str
        A .json file name.

    :return: dict | None
        A data dictionary.
    """

    try:
        # Take json data:
        return json.load(open(json_file_name))
    except Exception as err:
        print(err)
