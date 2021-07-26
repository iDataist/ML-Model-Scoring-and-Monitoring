import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import json
import os
import joblib

with open('config.json','r') as f:
    config = json.load(f)

deployed_model_path = os.path.join(config['prod_deployment_path'], config['model'])
test_data_path = os.path.join(config['test_data_path'], config['test_data_file'])
output_path = os.path.join(config['classification_report'])

def score_model():
    model = joblib.load(deployed_model_path)
    test_data = pd.read_csv(test_data_path)
    X_test = test_data[['lastmonth_activity','lastyear_activity','number_of_employees']]
    y_test = test_data['exited']
    y_preds = model.predict(X_test)
    return y_test, y_preds


def classification_report_image(
    y_test, y_preds, output_path
):

    """
    produces classification report for testing results
    input:
            y_test:  test response values
            y_test_preds: test predictions
            output_path: path to store the figure
    output:
            None
    """

    plt.rc("figure", figsize=(7, 5))
    plt.text(
        0.01, 0.5, str("Test"), {"fontsize": 10}, fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.1,
        str(classification_report(y_test, y_preds)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.savefig(output_path)
    plt.close()


if __name__ == '__main__':
    y_test, y_preds = score_model()
    classification_report_image(y_test, y_preds, output_path)
