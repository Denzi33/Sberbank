"""
------------------------------------------------------------------------------------------------------------------------
A launching module of parsing.
------------------------------------------------------------------------------------------------------------------------
ASYNC Functions:
------------------------------------------------------------------------------------------------------------------------
    main:
        Description: Saves a list of data dictionaries to a ".csv" file.

        Parameters:
            param_file_name - A name of parameters file (.JSON),
            res_file_name - A name of result file (.CSV),
            case - A parsing case number.

        Parameters type:
            str,
            str,
            int [1].

        Return type: None.
------------------------------------------------------------------------------------------------------------------------
"""

# Necessary modules:
import time
import asyncio

# Necessary modules with alias:
import pandas as pd
import utils_functions

# Necessary functions and variables:
from datetime import datetime, timedelta
from parsing import parsing
from urllib3.exceptions import InsecureRequestWarning

# Necessary functions and variables with alias:
from urllib3 import disable_warnings as dis_warn
from utils_functions import read_json


async def main(
               param_file_name: str,
               res_file_name: str,
               case: int = 1
) -> None:
    """
    Saves a list of data dictionaries to a ".csv" file.

    :param param_file_name: str
        A name of parameters files (.JSON),
    :param res_file_name: str
        A name of result file (.CSV),
    :param case: int [1]
        A parsing case number.

    :return: None.
    """

    # Save start time:
    # start_time = time.time()

    # Define time interval of request:
#     first_date = datetime(2011, 1, 1)
    tmp = read_json("../../resources/parameters.json")
    first_date = datetime.strptime(tmp["last_datetime"], "%Y-%m-%d %H:%M:%S"),
    last_date = datetime.now() - timedelta(hours=2)

    # Save data to pandas dataframe:
    pd.DataFrame(
                 await parsing(
                               f"{param_file_name}.json",
                               first_date[0],
                               last_date,
                               case
                              )
                ).to_parquet(f"../../data/{res_file_name}.gzip")

    # print(f"Parsing time (in seconds): {int(time.time() - start_time)}")


if __name__ == "__main__":
    # Turn off request warnings:
    dis_warn(InsecureRequestWarning)
    asyncio.run(main("../../resources/parameters", "new_data", 2))
