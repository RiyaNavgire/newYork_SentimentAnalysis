import json
import os,sys
import requests
import pandas as pd
import yaml

if __name__ == "__main__":
    
    try:
            vocab_length = 8888
            maxlen = 171
            modeltrainer_config_file = "D:\\MLProjectNeuron\\dataTrials\\data_trials\\constant\\training_pipeline_config\\model_trainer.py"
            with open(modeltrainer_config_file, "r") as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                # if dictionary is found
                if "VOCAB_LEN" in line:
                    # replace the value of this key
                        value = line.split("=")[1]                    
                        lines[i] = line.replace(value, str(vocab_length)+"\n")                                                
                if "MAX_LEN" in line:
                    # replace the value of this key
                        value = line.split("=")[1]                    
                        lines[i] = line.replace(value, str(maxlen))
                        break
        
            with open(modeltrainer_config_file, "w") as f1:
                f1.writelines(lines)           
                    
    except Exception as e:
        raise Exception(e, sys)
          
