from data_trials.exception import DataTrialException
from data_trials.logger import logger
from data_trials.config.pipeline.training import DataConfig
from data_trials.component import DataIngestion, DataValidation, DataTransformation, ModelTrainer,\
    ModelEvaluation, ModelPusher
from data_trials.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, \
    DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
import sys


class TrainingPipeline:

    def __init__(self, data_config: DataConfig):
        self.finance_config: DataConfig = data_config

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = self.finance_config.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()            
            return data_ingestion_artifact

        except Exception as e:
            raise DataTrialException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = self.finance_config.get_data_validation_config()
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)

            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise DataTrialException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = self.finance_config.get_data_transformation_config()
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=data_transformation_config

                                                     )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise DataTrialException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.finance_config.get_model_trainer_config()
                                         )
            model_trainer_artifact = model_trainer.initiate_model_training()
            return model_trainer_artifact
        except Exception as e:
            raise DataTrialException(e, sys)

    def start_model_evaluation(self, data_validation_artifact, model_trainer_artifact) -> ModelEvaluationArtifact:
        try:
            model_eval_config = self.finance_config.get_model_evaluation_config()
            model_eval = ModelEvaluation(data_validation_artifact=data_validation_artifact,
                                         model_trainer_artifact=model_trainer_artifact,
                                         model_eval_config=model_eval_config
                                         )
            return model_eval.initiate_model_evaluation()
        except Exception as e:
            raise DataTrialException(e, sys)

    def start_model_pusher(self, model_trainer_artifact: ModelTrainerArtifact):
        try:
            model_pusher_config = self.finance_config.get_model_pusher_config()
            model_pusher = ModelPusher(model_trainer_artifact=model_trainer_artifact,
                                       model_pusher_config=model_pusher_config
                                       )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise DataTrialException(e, sys)


    def start(self):
        try:
            #data_ingestion_artifact = self.start_data_ingestion()
            #data_ingestion_artifact = DataIngestionArtifact(
             #   feature_store_file_path="D:\\MLProjectNeuron\\dataTrials\\new_york_times_news_artifact\\data_ingestion\\feature_store\\new_york_times_news",
              #  download_dir="D:\\MLProjectNeuron\dataTrials\\new_york_times_news_artifact\\data_ingestion\\20230224_114952\\downloaded_files",
              #  metadata_file_path="D:\\MLProjectNeuron\\dataTrials\\new_york_times_news_artifact\\data_ingestion\\meta_info.yaml",)
            #data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            #data_validation_artifact = DataValidationArtifact(accepted_file_path=
             #                                                'D:\\MLProjectNeuron\\dataTrials\\new_york_times_news_artifact\\data_validation\\20230301_105119\\accepted_data\\new_york_times', 
             #                                                 rejected_dir='D:\\MLProjectNeuron\\dataTrials\\new_york_times_news_artifact\\data_validation\\20230301_105119\\rejected_data')
            #data_transformation_artifact = self.start_data_transformation(
             #    data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = DataTransformationArtifact(transformed_train_file_path='D:\\MLProjectNeuron\\dataTrials\\new_york_times_news_artifact\\data_transformation\\20230303_113708\\train\\newyork_transformation.txt', 
                                                                     exported_pipeline_file_path='D:\\MLProjectNeuron\\dataTrials\\new_york_times_news_artifact\\data_transformation\\20230303_113708\\transformed_pipeline',
                                                                     transformed_test_file_path='D:\\MLProjectNeuron\\dataTrials\\new_york_times_news_artifact\\data_transformation\\20230303_113708\\test\\newyork_transformation.txt')
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            #model_eval_artifact = self.start_model_evaluation(data_validation_artifact=data_validation_artifact,
             #                                                 model_trainer_artifact=model_trainer_artifact )
            #if model_eval_artifact.model_accepted:
             #   self.start_model_pusher(model_trainer_artifact=model_trainer_artifact)
        except Exception as e:
            raise DataTrialException(e, sys)
