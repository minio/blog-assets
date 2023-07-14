"""
PyTorch Serve model handler using MinIO for model file fetching.

For more information about custom handlers and the Handler class: 
https://pytorch.org/serve/custom_service.html#custom-handler-with-class-level-entry-point
"""
import os
from minio import Minio
from minio.error import S3Error
import torch
from transformers import AutoTokenizer
import transformers
from ts.torch_handler.base_handler import BaseHandler

# In this example, we serve the Falcon-7B Large Language Model (https://huggingface.co/tiiuae/falcon-7b)
# However, you can use your model of choice. Just make sure to edit the implementations of
# initialize() and handle() according to your model!

# Make sure the following are populated with your MinIO details
# (Best practice is to use environment variables!)
MINIO_ENDPOINT = ''
MINIO_ACCESS_KEY = ''
MINIO_SECRET_KEY = ''
MODEL_BUCKET = 'models'
CURRENT_MODEL_NAME = "falcon-7b"

def get_minio_client():
    """
    Initializes and returns a Minio client object
    """
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
    )
    return client

class MinioModifiedHandler(BaseHandler):
    """
    Handler class that loads model files from MinIO.
    """
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.model = None
        self.tokenizer = None

    def load_model_files_from_bucket(self, context):
        """
        Fetch model files from MinIO if not present in Model Store. 
        """
        client = get_minio_client()
        properties = context.system_properties
        object_name_prefix_len = len(CURRENT_MODEL_NAME) + 1
        # model_dir is the temporary directory (TEMP) allocated in the Model Store for this model
        model_dir = properties.get("model_dir")
        try:
            # fetch all the model files and place them in TEMP
            # the following assumes a bucket organized like this:
            #    MODEL_BUCKET -> CURRENT_MODEL_NAME -> all the model files
            for item in client.list_objects(MODEL_BUCKET, prefix=CURRENT_MODEL_NAME, recursive=True):
                # We don't include the object name's prefix in the destination file path because we
                # don't want the enclosing folder to be added to TEMP.
                destination_file_path = model_dir + "/" + item.object_name[object_name_prefix_len:]
                # only fetch the model file if it is not already in TEMP
                if not os.path.exists(destination_file_path):
                    client.fget_object(MODEL_BUCKET, item.object_name, destination_file_path)
            return True
        except S3Error:
            return False

    def initialize(self, context):
        """
        Worker initialization method. 
        Loads up a copy of the trained model.

        See https://huggingface.co/tiiuae/falcon-7b for details about how 
        the Falcon-7B LLM is loaded with the use of the Transformers library
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        success = self.load_model_files_from_bucket(context)
        if not success:
            print("Something went wrong while attempting to fetch model files.")
            return
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_dir,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model = pipeline
        self.tokenizer = tokenizer
        self.initialized = True

    def handle(self, data, context):
        """
        Entrypoint for inference call to TorchServe.
        Note: This example assumes the API request body looks like:
        {
            "input": "<input for inference>"
        }
        Note: Check the 'data' argument to see how your request body looks.
        """
        input_text = data[0].get("body").get("input")
        sequences = self.model(
            input_text,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return [sequences]
