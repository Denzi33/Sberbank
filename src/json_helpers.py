"""
------------------------------------------------------------------------------------------------------------------------
The module is designed to work with json files.
------------------------------------------------------------------------------------------------------------------------
Functions:
------------------------------------------------------------------------------------------------------------------------
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


def read_json(json_file_name: str) -> dict | None:
    """
    Returns json file data as dictionary.

    :param json_file_name: str
        A json file name.

    :return: dict | None
        A data dictionary.
    """

    try:
        # Take json data:
        return json.load(open(json_file_name))
    except Exception as err:
        print(err)
