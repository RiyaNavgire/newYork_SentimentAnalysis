import os
import re
import sys
import time
import uuid
import pdb
from collections import namedtuple
from typing import List

import json
import pandas as pd
import requests

from data_trials.config.pipeline.training import DataConfig
from data_trials.constant.environment.variable_key import NEWS_API_KEY
from data_trials.config.spark_manager import spark_session
from data_trials.entity.artifact_entity import DataIngestionArtifact
from data_trials.entity.config_entity import DataIngestionConfig
from data_trials.entity.metadata_entity import DataIngestionMetadata
from data_trials.exception import DataTrialException
from data_trials.logger import logger
from datetime import datetime
from data_trials.constant import TIMESTAMP


DownloadUrl = namedtuple("DownloadUrl", ["url", "file_path", "n_retry"])


class DataIngestion:
    # Used to download data in chunks.
    def __init__(self, data_ingestion_config: DataIngestionConfig, n_retry: int = 5, ):
        """
        data_ingestion_config: Data Ingestion config
        n_retry: Number of retry filed should be tried to download in case of failure encountered
        n_month_interval: n month data will be downloded
        """
        try:
            logger.info(f"{'>>' * 20}Starting data ingestion.{'<<' * 20}")
            self.data_ingestion_config = data_ingestion_config
            self.failed_download_urls: List[DownloadUrl] = []
            self.n_retry = n_retry

        except Exception as e:
            raise DataTrialException(e, sys)

    def download_files(self,):
        """
        n_month_interval_url: if not provided then information default value will be set
        =======================================================================================
        returns: List of DownloadUrl = namedtuple("DownloadUrl", ["url", "file_path", "n_retry"])
        """
        try:
            logger.info("Started downloading files")
            
            logger.debug("Generating data download url")
            datayear = self.data_ingestion_config.year
            datamonth = self.data_ingestion_config.month
            apikey = self.data_ingestion_config.apikey
            datasource_url: str = self.data_ingestion_config.datasource_url
            logger.debug(f"Url: {datasource_url}")
            logger.debug(f"Year: {datayear}")
            logger.debug(f"Month: {datamonth}")
            file_name = f"{self.data_ingestion_config.file_name}_{datayear}_{datamonth}.json"
            file_path = os.path.join(self.data_ingestion_config.download_dir, file_name)
            url = datasource_url.replace("<year>", datayear).replace("<month>", datamonth).replace("<key>",apikey)
            download_url = DownloadUrl(url=url, file_path=file_path, n_retry=self.n_retry)
            #logger.debug(f"Url: {url}")
            self.download_data(download_url = download_url)
            #url = datasource_url
                       
            
            logger.info(f"File download completed")
        except Exception as e:
            raise DataTrialException(e, sys)

    def download_data(self, download_url: DownloadUrl):
      
        try:
            logger.info(f"Starting download operation: {download_url}")
            download_dir = os.path.dirname(download_url.file_path)

            # creating download directory
            os.makedirs(download_dir, exist_ok=True)

            # downloading data
            data = requests.get(download_url.url)
            
            result = []
          
            try:
                logger.info(f"Started writing downloaded data into json file: {download_url.file_path}")
                # saving downloaded data into hard disk                
                    
                with open(download_url.file_path, "w") as file_obj:
                   storage = data.json()
                   responselength = len(storage['response']['docs'])                                        
                   for i in range(0, responselength): #iterate over all responses objects
                        #    print(storage['response']['docs'][i]['lead_paragraph'])   
                            dictionary_para = {}                     
                            item = storage['response']['docs'][i]['lead_paragraph']
                            dictionary_para['lead_paragraph'] = item  
                            result.append(dictionary_para)
                   json.dump(result, file_obj)
                
                logger.info(f"Downloaded data has been written into file: {download_url.file_path}")
            except Exception as e:
                logger.info("Failed to download hence retry again.")
                
                # removing file failed file exist
                if os.path.exists(download_url.file_path):
                    os.remove(download_url.file_path)
                self.retry_download_data(data, download_url=download_url)
                
        except Exception as e:
            logger.info(e)
            raise DataTrialException(e, sys)
        #finally : file_obj.close()
        
    def convert_files_to_parquet(self, ) -> str:
        """
        downloaded files will be converted and merged into single parquet file
        json_data_dir: downloaded json file directory
        data_dir: converted and combined file will be generated in data_dir
        output_file_name: output file name 
        =======================================================================================
        returns output_file_path
        """
        try:
            json_data_dir = self.data_ingestion_config.download_dir          
            data_dir = self.data_ingestion_config.feature_store_dir
            output_file_name = self.data_ingestion_config.file_name
            os.makedirs(data_dir, exist_ok=True)
            file_path = os.path.join(data_dir, f"{output_file_name}")            
            logger.info(f"Parquet file will be created at: {file_path}")
            if not os.path.exists(json_data_dir):
                return file_path
            for file_name in os.listdir(json_data_dir):
                json_file_path = os.path.join(json_data_dir, file_name)
                logger.debug(f"Converting {json_file_path} into parquet format at {file_path}") 
               #spark_session.conf.set("spark.sql.caseSensitive", "true")
                #df = spark_session.read.option("multiline","true").json(json_file_path)
                df = spark_session.read.json(json_file_path)
            if df.count() > 0:
                 df.write.mode('append').parquet(file_path)                   
            df.show(10,False)
            return file_path
        except Exception as e:
            raise DataTrialException(e, sys)
        
            

    def retry_download_data(self, data, download_url: DownloadUrl):
        """
        This function help to avoid failure as it help to download failed file again
        
        data:failed response
        download_url: DownloadUrl
        """
        try:
            # if retry still possible try else return the response
            if download_url.n_retry == 0:
                self.failed_download_urls.append(download_url)
                logger.info(f"Unable to download file {download_url.url}")
                return

            # to handle throatling requestion and can be slove if we wait for some second.
            content = data.content.decode("utf-8")
            wait_second = re.findall(r'\d+', content)

            if len(wait_second) > 0:
                time.sleep(int(wait_second[0]) + 2)

            # Writing response to understand why request was failed
            failed_file_path = os.path.join(self.data_ingestion_config.failed_dir,
                                            os.path.basename(download_url.file_path))
            os.makedirs(self.data_ingestion_config.failed_dir, exist_ok=True)
            with open(failed_file_path, "wb") as file_obj:
                file_obj.write(data.content)

            # calling download function again to retry
            download_url = DownloadUrl(download_url.url, file_path=download_url.file_path,
                                       n_retry=download_url.n_retry - 1)
            self.download_data(download_url=download_url)
        except Exception as e:
            raise DataTrialException(e, sys)

    def write_metadata(self, file_path: str) -> None:
        """
        This function help us to update metadata information 
        so that we can avoid redundant download and merging.

        """
        try:
            logger.info(f"Writing metadata info into metadata file.")
            metadata_info = DataIngestionMetadata(metadata_file_path=self.data_ingestion_config.metadata_file_path)

            metadata_info.write_metadata_info(month=self.data_ingestion_config.month,
                                              year=self.data_ingestion_config.year,
                                              data_file_path=file_path)                                             
            logger.info(f"Metadata has been written.")
        except Exception as e:
            raise DataTrialException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info(f"Started downloading json file")
            self.download_files()

            if os.path.exists(self.data_ingestion_config.download_dir):
                logger.info(f"Converting and combining downloaded json into parquet file")
                file_path = self.convert_files_to_parquet()
                self.write_metadata(file_path=file_path)

            feature_store_file_path = os.path.join(self.data_ingestion_config.feature_store_dir, self.data_ingestion_config.file_name)
            artifact = DataIngestionArtifact(
                feature_store_file_path=feature_store_file_path,
                download_dir=self.data_ingestion_config.download_dir,
                metadata_file_path=self.data_ingestion_config.metadata_file_path,
            )

            logger.info(f"Data ingestion artifact: {artifact}")
            return artifact
        except Exception as e:
            raise DataTrialException(e, sys)


def main():
    try:
        config = DataConfig()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config, n_day_interval=6)
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        raise DataTrialException(e, sys)


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        logger.exception(e)
