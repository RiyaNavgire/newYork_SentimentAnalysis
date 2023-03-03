import requests
import os
import pdb
import argparse
from data_trials.exception import DataTrialException
from data_trials.pipeline import TrainingPipeline
from data_trials.logger import logger
from data_trials.config.pipeline.training import DataConfig
import sys

#def start_training(start=False):
 #   try:
  #      if not start:
   #         return None
    #    print("Training Running")
     #   TrainingPipeline(DataConfig()).start()
        
        
    #except Exception as e:
     #   raise DataTrialException(e, sys)

def start_training():
    try:
        print("Training Running")
        TrainingPipeline(DataConfig()).start()
        
        
    except Exception as e:
        raise DataTrialException(e, sys)
    
def main(training_status, prediction_status):
    try:

        start_training(start=training_status)
        

    except Exception as e:
        raise DataTrialException(e, sys)

def main():
    try:

        start_training()

    except Exception as e:
        raise DataTrialException(e, sys)
    
if __name__ == "__main__":
    try:
       # parser = argparse.ArgumentParser()
       # parser.add_argument("--t", default=0, type=int, help="If provided true training will be done else not")
       # parser.add_argument("--p", default=0, type=int, help="If provided prediction will be done else not")
        # args = parser.parse_args()

        #main(training_status=args.t, prediction_status=args.p)
        main()
    except Exception as e:
        print(e)
        pass
        logger.exception(DataTrialException(e, sys))


