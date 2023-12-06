#!/usr/local/bin python3

from flask import Flask, request, jsonify
import logging
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
logging.basicConfig(level=logging.INFO)

@app.route('/minio_event', methods=['POST'])
def log_bucket_event():
    """
    Logs events received from MinIO to the Python logger.
    """
    event_data = request.json
    logging.info(f"Event received: {event_data}")
    return jsonify({'message': 'Event logged successfully'})

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5000))
    debug_mode = os.getenv('FLASK_DEBUG', 'False') == 'True'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
