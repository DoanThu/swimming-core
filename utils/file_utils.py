import json
import os
from typing import List
import yaml

# def write_json(data: List[str], filename:str):
#     """ Save list of json object to filename. Create folders to filename if they do not exist.

#     Args:
#         data (List[str]): List of json object
#         filename (str): filename including .json
#     """
#     if not os.path.exists(os.path.dirname(filename)):
#         os.makedirs(os.path.dirname(filename))
#     with open(filename, 'w') as f:
#         json.dump(data, f)
            
def read_yaml(filename: str) -> dict:
    """ Read yaml file and return a dict

    Args:
        filename (str): path to the yaml file 

    Returns:
        dict: dict of config file
    """
    with open(filename) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise Exception(exc)

def write_to_csv(data, filename):
    """
    If the folders not exist create ones and append data to csv file.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'a') as file:
        file.write(data + '\n')