import sys

from cnnClassifier.aws.s3_operations import SimpleStorageService
from cnnClassifier.exception import cnnClassifierException
from cnnClassifier.logger import logging
from cnnClassifier.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from cnnClassifier.entity.config_entity import ModelPusherConfig
from cnnClassifier.aws.s3_estimator import ChestDiseaseEstimator


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of model evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        
        # ChestDiseaseEstimator ka use karenge s3 interaction ke liye
        self.chest_estimator = ChestDiseaseEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This function is used to upload the trained model to the S3 bucket
        
        Output      :   Returns ModelPusherArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            logging.info("Uploading trained model to S3 bucket...")

            # Model file ko S3 par upload karna
            self.chest_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            logging.info("Successfully uploaded trained model to S3 bucket.")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            
            return model_pusher_artifact
            
        except Exception as e:
            raise cnnClassifierException(e, sys) from e