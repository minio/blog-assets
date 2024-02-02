"""
This is a simple Flask text extraction server that functions as a webhook service endpoint 
for PUT events in a MinIO bucket. Apache Tika is used to extract the text from the new objects.
"""
from flask import Flask, request, abort, make_response
import io
import logging
from tika import parser
from minio import Minio

# Make sure the following are populated with your MinIO details
# (Best practice is to use environment variables!)
MINIO_ENDPOINT = ''
MINIO_ACCESS_KEY = ''
MINIO_SECRET_KEY = ''


# This depends on how you are deploying Tika (and this server):
TIKA_SERVER_URL = 'http://localhost:9998/tika'

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/', methods=['POST'])
async def text_extraction_webhook():
    """
    This endpoint will be called when a new object is placed in the bucket
    """
    if request.method == 'POST':
        # Get the request event from the 'POST' call
        event = request.json
        bucket = event['Records'][0]['s3']['bucket']['name']
        obj_name = event['Records'][0]['s3']['object']['key']

        obj_response = client.get_object(bucket, obj_name)
        obj_bytes = obj_response.read()
        file_like = io.BytesIO(obj_bytes)
        parsed_file = parser.from_buffer(file_like.read(), serverEndpoint=TIKA_SERVER_URL)
        text = parsed_file["content"]
        metadata = parsed_file["metadata"]
        logger.info(text)
        result = {
            "text": text, 
            "metadata": metadata
            }
        resp = make_response(result, 200)
        return resp
    else:
        abort(400)

if __name__ == '__main__':
    app.run()
