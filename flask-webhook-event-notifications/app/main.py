from flask import Flask, jsonify, request

# Flask app initialization
app = Flask(__name__)


@app.route('/minio-event', methods=['POST'])
def handle_minio_event():
    event_data = request.json
    # Using Flask's default logger to log the event data
    app.logger.info(f"Received MinIO event: {event_data}")
    return jsonify(event_data), 200


@app.route('/hello', methods=['GET'])
def hello():
    return "Hello MinIO!"


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
