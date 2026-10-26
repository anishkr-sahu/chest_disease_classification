from pathlib import Path


CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"


"""
model evaluation related constants
"""
AWS_BUCKET_NAME = "chest-disease-mlops-bucket-2026"
MODEL_PUSHER_S3_KEY = "model.h5"

