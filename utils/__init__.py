import shutil

import yaml

from typing import List
from collections import OrderedDict
from pyspark.sql import DataFrame
from data_trials.exception import DataTrialException
from data_trials.logger import logger
import os, sys
import json
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def write_yaml_file(file_path: str, data: dict = None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as yaml_file:
            if data is not None:
                yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
                yaml.dump(data,yaml_file,default_flow_style=False)
    except Exception as e:
        raise DataTrialException(e, sys)


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise DataTrialException(e, sys) from e


def get_score(dataframe: DataFrame, metric_name, label_col, prediction_col) -> float:
    try:
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol=prediction_col,
            metricName=metric_name)
        score = evaluator.evaluate(dataframe)
        print(f"{metric_name} score: {score}")
        logger.info(f"{metric_name} score: {score}")
        return score
    except Exception as e:
        raise DataTrialException(e, sys)


def create_directories(directories_list: List[str], new_directory=False):
    try:

        for dir_path in directories_list:
            if dir_path.startswith("s3"):
                continue
            if os.path.exists(dir_path) and new_directory:
                shutil.rmtree(dir_path)
                logger.info(f"Directory removed: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Directory created: {dir_path}")
    except Exception as e:
        raise DataTrialException(e, sys)

def cleanJsonfile(file_path:str):
#FOR API RESPONSE ALWAYS EXTRACT ONLY COLUMNS OF INTEREST ELSE IT BECOMES DIFFICULT TO BREAK THE RESPONSE
        try:
            #ile_path = "D://MLProjectNeuron//searchArt.json"
            with open(file_path, 'r', encoding='utf-8') as f:
                 news_list = json.load(f)
                             
           
            abstract_data = []
            
            for key in news_list.keys():
                #print(str(key))
                if("response" in str(key)):
                    response_values = list(news_list.values())[1]
                    for responseKey in enumerate(response_values):
                        if("0" in str(responseKey)):
                              abstract_data = list(response_values.values())[0]                             
                              break             
                              
            ##ALWAYS USE JSON DUMPS TO ADD STRING TO FILE ELSE IT GIVES INCORRECT JSON OBJECT
            #new_file_path = "D://MLProjectNeuron//modified.json"               
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(abstract_data, f)
           
            
        
        except Exception as e:
            raise DataTrialException(e, sys)
        finally: f.close()