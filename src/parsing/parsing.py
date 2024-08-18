"""
------------------------------------------------------------------------------------------------------------------------
The module is designed to parsing data from web-site (API).
------------------------------------------------------------------------------------------------------------------------
ASYNC Functions:
------------------------------------------------------------------------------------------------------------------------
    get_data:
        Description: Returns data from a resource with set parameters.

        Parameters:
            param - A parameters for requests,
            date - A date of the request,
            case - A parsing case number.

        Parameters type:
            dict,
            datetime,
            int [1].

        Returns: A list of dictionaries with data.

        Return type: list.

    parsing:
        Description: The method parses the data.

        Parameters:
            param_file_name - A name of parameters file (.JSON),
            start_date - The left time limit of the request,
            end_date - The right time limit of the request,
            case - A parsing case number.

        Parameters type:
            str,
            datetime,
            datetime,
            int [1].

        Returns: A parsed data.

        Return type: list.
------------------------------------------------------------------------------------------------------------------------
"""

# Necessary functions and variables:
from datetime import datetime
from utils_functions import make_request, read_json  #, clean_data
from dateutil.rrule import rrule, HOURLY


async def get_data(
                   param: dict,
                   date: datetime,
                   case: int = 1
) -> list:
    """
    Returns data from a resource with set parameters.

    :param param: dict
        Parameters for requests.
    :param date: datetime
        A date of the request.
    :param case: int [1]
        A parsing case number.

    :return: list
        A list of dictionaries with data.
    """

    # Final answer:
    resp: list = []

    # Prepare request URL:
    website_url: str = f"{param.get('website_url')}Date={date.strftime('%Y-%m-%d')}&Hour={date.strftime('%H')}"

    # Case with districts:
    if case == 1:
        try:
            resp = (await make_request(website_url)).get("SubAreas")

            # The data will be processed by pandas:
            # clear_resp: list = await clean_data(resp, parameters.get("del_columns"), inplace=False)

            # Add date column:
            [row.update({"Datetime": date}) for row in resp]

            return resp

        except Exception as err:
            print(err)

            return []

    # Case with regions:
    elif case == 2:
        try:
            for region_id in param.get("regions_ids"):
                cur_website_url = f"{website_url}&PowerSystemId={region_id}"

                # Add regions data:
                resp.extend((await make_request(cur_website_url)).get("SubAreas"))

                # The data will be processed by pandas:
                # clear_resp: list = await clean_data(resp, parameters.get("del_columns"), inplace=False)

                # Add date column:
                [row.update({"Datetime": date}) for row in resp]

            return resp

        except Exception as err:
            print(err)

            return []

    # Case with districts and regions (too slow):
    elif case == 3:
        ...


async def parsing(
                  param_file_name: str,
                  start_date: datetime,
                  end_date: datetime,
                  case: int = 1
) -> list:
    """
    The method parses the data.

    :param param_file_name: str
        A name of parameters file (.JSON),
    :param start_date: datetime
        The left time limit of the request,
    :param end_date: datetime
        The right time limit of the request,
    :param case: int [1]
        A parsing case number.

    :return: list
        A parsed data.
    """

    params: dict = read_json(param_file_name)

    # All parsing data:
    result: list = []

    curr_date: datetime = datetime.now()

    # Check the request date:
    if end_date > curr_date:
        raise Exception("Error! Incorrect last date.")

    # Parse data from the website:
    [result.extend(await get_data(params, date, case)) for date
     in rrule(HOURLY, dtstart=start_date, until=end_date)]

    return result
