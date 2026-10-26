from cnnClassifier.aws.s3_operations import SimpleStorageService
from cnnClassifier.exception import cnnClassifierException
import sys
import tensorflow as tf
from pathlib import Path
from PIL import Image
import numpy as np


class ChestDiseaseEstimator:
    """
    This class is used to save and retrieve the Chest Disease VGG16/CNN model in an S3 bucket and perform image predictions.
    """

    def __init__(self, bucket_name: str, model_path: str):
        """
        :param bucket_name: Name of your S3 model bucket
        :param model_path: S3 path/key of your model (.h5 or .keras)
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model: tf.keras.Model = None

    def is_model_present(self, model_path: str) -> bool:
        """Checks if the model file exists in the S3 bucket."""
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except Exception as e:
            print(e)
            return False

    def load_model(self) -> tf.keras.Model:
        """
        Load the TensorFlow/Keras model from the S3 model_path.
        :return: tf.keras.Model
        """
        try:
            return self.s3.load_model(model_name=self.model_path, bucket_name=self.bucket_name)
        except Exception as e:
            raise cnnClassifierException(e, sys)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Save/Upload the local model to the S3 bucket model_path.
        :param from_file: Your local system model path (e.g., artifacts/training/model.h5)
        :param remove: If True, deletes the local file after uploading.
        """
        try:
            self.s3.upload_file(
                from_filename=from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )
        except Exception as e:
            raise cnnClassifierException(e, sys)

    def predict(self, image_path: str) -> np.ndarray:
        """
        Perform prediction on a chest X-ray image path.
        :param image_path: Path to the local image file to be predicted.
        :return: Model prediction probabilities array.
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()

            # Image ko preprocess karna (Model ke training size ke anusaar adjust karein, e.g., 224x224)
            image_size = (224, 224) 
            img = Image.open(image_path).resize(image_size)
            
            # Convert image to numpy array & expand dimensions for batch size
            test_image = np.array(img)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255.0  # Rescaling as done during training

            # Predict
            predictions = self.loaded_model.predict(test_image)
            return predictions

        except Exception as e:
            raise cnnClassifierException(e, sys)