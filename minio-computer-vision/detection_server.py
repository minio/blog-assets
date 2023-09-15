"""
This is a simple Flask inference server implementation that serves as a webhook for
the event of a new image being added to a MinIO bucket. Object detection using YOLO will
be performed on that image and the resulting predictions will be returned.
"""
from flask import Flask, request, abort, make_response
from ultralytics import YOLO
import tempfile
from minio import Minio

# Make sure the following are populated with your MinIO details
# (Best practice is to use environment variables!)
MINIO_ENDPOINT = ''
MINIO_ACCESS_KEY = ''
MINIO_SECRET_KEY = ''

model = YOLO('/PATH/TO/best.pt')  # load a custom model (path to trained weights)

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
)

app = Flask(__name__)

@app.route('/', methods=['POST'])
async def inference_bucket_webhook():
    """
    This endpoint will be called when a new object is placed in your inference bucket
    """
    if request.method == 'POST':
        # Get the request event from the 'POST' call
        event = request.json
        bucket = event['Records'][0]['s3']['bucket']['name']
        obj_name = event['Records'][0]['s3']['object']['key']
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_name = temp_dir+'/'+obj_name
            client.fget_object(bucket, obj_name, temp_file_name)
            # See https://docs.ultralytics.com/modes/predict/ for more information about YOLO inference options
            results = model.predict(source=temp_file_name, conf=0.5, stream=False)
            # A list of bounding boxes (if any) is returned.
            # Each bounding box is in the format [x1, y1, x2, y2, probability, class].
            result = {"results": results[0].boxes.data.tolist()}
            print(result)
            resp = make_response(result, 200)
            return resp
    else:
        abort(400)

if __name__ == '__main__':
    app.run()
