import hashlib
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

def sign(method, url, body, host, headers, service="s3tables", region="dummy", access_key=None, secret_key=None):
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    payload_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

    headers['x-amz-content-sha256'] = payload_hash
    headers['Host'] = host.replace("http://", "").replace("https://", "")

    request = AWSRequest(
        method=method,
        url=url,
        data=body,
        headers=headers,
    )
    SigV4Auth(session.get_credentials(), service, region).add_auth(request)
    return request
