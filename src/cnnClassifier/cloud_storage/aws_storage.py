import boto3
from cnnClassifier.configuration.aws_connection import S3Client
import os, sys
from cnnClassifier.logger import logging
from mypy_boto3_s3.service_resource import Bucket
from cnnClassifier.exception import cnnClassifierException
from botocore.exceptions import ClientError
import tensorflow as tf
from pathlib import Path


class SimpleStorageService:

    def __init__(self):
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        """Checks if a file or folder path exists in the S3 bucket."""
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects) > 0
        except Exception as e:
            raise cnnClassifierException(e, sys)

    def get_bucket(self, bucket_name: str) -> Bucket:
        """Returns the S3 bucket resource object."""
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            return bucket
        except Exception as e:
            raise cnnClassifierException(e, sys) from e

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> tf.keras.Model:
        """Downloads and loads a TensorFlow/Keras deep learning model from S3."""
        logging.info("Entered the load_model method of SimpleStorageService class")
        try:
            model_file = model_name if model_dir is None else f"{model_dir}/{model_name}"
            
            # Temporary local path par download karke model load karna safe hota hai
            local_temp_path = Path("temp_model.h5")
            self.s3_client.download_file(bucket_name, model_file, str(local_temp_path))
            
            model = tf.keras.models.load_model(local_temp_path)
            
            # Cleanup temporary file
            if os.path.exists(local_temp_path):
                os.remove(local_temp_path)
                
            logging.info("Exited the load_model method of SimpleStorageService class")
            return model
        except Exception as e:
            raise cnnClassifierException(e, sys) from e

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = False):
        """Uploads any file (like images, zip files, or model weights) to S3."""
        logging.info("Entered the upload_file method of SimpleStorageService class")
        try:
            logging.info(f"Uploading {from_filename} to s3://{bucket_name}/{to_filename}")
            self.s3_client.upload_file(from_filename, bucket_name, to_filename)
            logging.info(f"Successfully uploaded {from_filename}")

            if remove is True and os.path.exists(from_filename):
                os.remove(from_filename)
                logging.info(f"Removed local file: {from_filename}")

        except Exception as e:
            raise cnnClassifierException(e, sys) from e

    def download_file(self, s3_filename: str, local_filename: str, bucket_name: str):
        """Downloads any file or model weight from S3 to local machine."""
        logging.info("Entered the download_file method of SimpleStorageService class")
        try:
            self.s3_client.download_file(bucket_name, s3_filename, local_filename)
            logging.info(f"Successfully downloaded s3://{bucket_name}/{s3_filename} to {local_filename}")
        except Exception as e:
            raise cnnClassifierException(e, sys) from e