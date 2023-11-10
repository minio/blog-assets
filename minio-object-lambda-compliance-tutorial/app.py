from flask import Flask, request, abort, make_response
import requests
import json

app = Flask(__name__)

@app.route('/', methods=['POST'])
def get_webhook():
    if request.method == 'POST':
        # obtain the request event from the 'POST' call
        event = request.json

        object_context = event["getObjectContext"]

        # Get the presigned URL to fetch the requested
        # original object from MinIO
        s3_url = object_context["inputS3Url"]

        # Extract the route and request token from the input context
        request_route = object_context["outputRoute"]
        request_token = object_context["outputToken"]

        # Get the original S3 object using the presigned URL
        r = requests.get(s3_url)
        original_object = r.content.decode('utf-8')

        # Transform the JSON object by anonymizing the credit card number
        transformed_object = anonymize_credit_card(original_object)

        # Write object back to S3 Object Lambda
        # response sends the transformed data
        # back to MinIO and then to the user
        resp = make_response(transformed_object, 200)
        resp.headers['x-amz-request-route'] = request_route
        resp.headers['x-amz-request-token'] = request_token
        return resp

    else:
        abort(400)

def anonymize_credit_card(original_object):
    # Assume the original_object is a JSON string
    data = json.loads(original_object)

    # Check if the JSON is a list of transactions
    if isinstance(data, list):
        # Anonymize the credit card number in each transaction by keeping only the last four digits
        for transaction in data:
            if 'credit_card_number' in transaction:
                transaction['cc_last_four_digits'] = transaction.pop('credit_card_number')[-4:]

    # Convert the updated data back to JSON
    transformed_object = json.dumps(data)

    return transformed_object

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
