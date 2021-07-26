from flask import Flask, request, jsonify, render_template
from flask.logging import create_logger
import logging
import pandas as pd
import json
import os
import joblib
import diagnostics

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)-15s %(message)s",
)
logger = logging.getLogger()

with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], config['ingested_data_file'])
prediction_model = os.path.join(config['prod_deployment_path'], config['model'])
test_records = os.path.join(config['prod_deployment_path'],
                            config['test_records'])

@app.route("/")
def home():
    html = "<h3>Customer Churn Prediction Home</h3>"
    return html.format(format)

@app.route("/predict", methods=['POST'])
def predict():
    """Performs an sklearn prediction
    input looks like:
    {"lastmonth_activity": {
            "0": 234},
    "lastyear_activity": {
            "0": 3},
    "number_of_employees": {
            "0": 10}}
    result looks like:
    { "prediction": [ 1 ] }
    """
    try:
        clf = joblib.load(prediction_model)
    except Exception as e:
        logger.error(e)
        return "Model not loaded"
    json_payload = request.json
    logger.info(f"JSON payload: {json_payload}")
    inference_payload = pd.DataFrame(json_payload)
    logger.info(f"inference payload DataFrame:{inference_payload}")
    prediction = list(clf.predict(inference_payload))
    prediction = [int(x) for x in prediction]
    return jsonify({'prediction': prediction})

@app.route("/score")
def score():
    with open(test_records, 'r') as file:
        f1 = file.read().split(' ')[-2]
        return jsonify({'f1_score': f1})

@app.route("/summary_stats")
def summary_stats():
    df = pd.read_csv(dataset_csv_path).describe()
    return render_template('simple.html', tables=[df.to_html()], titles=df.columns.values)

@app.route("/timing")
def timing():
    times = diagnostics.execution_time()
    return jsonify({'ingestion_time': times[0], 'training_time': times[1]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
