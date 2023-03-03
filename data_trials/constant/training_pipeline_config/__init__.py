import os

PIPELINE_NAME = "new_york_times_news"
PIPELINE_ARTIFACT_DIR = os.path.join(os.getcwd(), "new_york_times_news_artifact")

from data_trials.constant.training_pipeline_config.data_ingestion import *
from data_trials.constant.training_pipeline_config.data_validation import *
from data_trials.constant.training_pipeline_config.data_transformation import *
from data_trials.constant.training_pipeline_config.model_trainer import *
from data_trials.constant.training_pipeline_config.model_evaluation import *
from data_trials.constant.training_pipeline_config.model_pusher import *
