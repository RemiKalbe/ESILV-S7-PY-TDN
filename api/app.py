from models.logistic_regression import LRMUsing, LogisticRegressionModel
from models.knn import KNNModel
from models.decision_tree import DecisionTreeModel
from models.xg_boost import XGBoostModel
from models.gradient_boosting import GradientBoostingModel
from models.random_forest import RandomForestModel
import numpy as np
from uuid import uuid4
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import os
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__)
app.secret_key = 'super secret key'

UPLOAD_FOLDER = '/Users/remikalbe/Git/github.com/ESILV-S7-PY-TDN/api/uploads'
RESULT_FOLDER = '/Users/remikalbe/Git/github.com/ESILV-S7-PY-TDN/api/results'
ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        flash('No X_test provided')
        return redirect('/')
    app.logger.info(request.files)
    upload_files = request.files.getlist('files[]')
    app.logger.info(upload_files)
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    upload_id = str(uuid4())
    if not upload_files:
        flash('No selected file')
        return redirect('/')
    for file in upload_files:
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect('/')
        if file and allowed_file(file.filename):
            # create new folder for each upload
            os.makedirs(os.path.join(
                UPLOAD_FOLDER, upload_id), exist_ok=True)
            file.save(os.path.join(
                f'{UPLOAD_FOLDER}/{upload_id}', secure_filename(file.filename)))
            # Retirect to result page
    return redirect(f'/api/predict/result/{upload_id}')


@app.route('/api/predict/result/<upload_id>')
def result(upload_id):
    # Import X_train and Y_train from the ./models/data folder
    X_train = np.loadtxt('./models/data/X_train.csv', delimiter=',')
    y_train = np.loadtxt('./models/data/y_train.csv', delimiter=',')
    # Import X_test and Y_test
    X_test = np.loadtxt(
        f'{UPLOAD_FOLDER}/{upload_id}/X_test.csv', delimiter=',')
    y_test = np.loadtxt(
        f'{UPLOAD_FOLDER}/{upload_id}/y_test.csv', delimiter=',')

    # Initialize the models
    model_logreg_rfe = LogisticRegressionModel(
        X_train, y_train, X_test, y_test, LRMUsing.RFE)
    model_logreg_grid = LogisticRegressionModel(
        X_train, y_train, X_test, y_test, LRMUsing.GridSearchCV)
    model_knn = KNNModel(X_train, y_train, X_test, y_test)
    model_decision_tree = DecisionTreeModel(X_train, y_train, X_test, y_test)
    model_xg_boost = XGBoostModel(X_train, y_train, X_test, y_test)
    model_gradient_boosting = GradientBoostingModel(
        X_train, y_train, X_test, y_test)
    model_random_forest = RandomForestModel(X_train, y_train, X_test, y_test)

    # Get the results
    model_logreg_rfe_score = model_logreg_rfe.score()
    model_logreg_grid_score = model_logreg_grid.score()
    model_knn_score = model_knn.score()
    model_decision_tree_score = model_decision_tree.score()
    model_xg_boost_score = model_xg_boost.score()
    model_gradient_boosting_score = model_gradient_boosting.score()
    model_random_forest_score = model_random_forest.score()

    # Get the predictions
    model_logreg_rfe_predictions = model_logreg_rfe.predict()
    model_logreg_grid_predictions = model_logreg_grid.predict()
    model_knn_predictions = model_knn.predict()
    model_decision_tree_predictions = model_decision_tree.predict()
    model_xg_boost_predictions = model_xg_boost.predict()
    model_gradient_boosting_predictions = model_gradient_boosting.predict()
    model_random_forest_predictions = model_random_forest.predict()

    # Create the result folder
    os.makedirs(os.path.join(
        RESULT_FOLDER, upload_id), exist_ok=True)
    # Save each model's predictions
    logreg_grid_prediction_url = f'{RESULT_FOLDER}/{upload_id}/logreg_grid_predictions.csv'
    np.savetxt(logreg_grid_prediction_url,
               model_logreg_rfe_predictions, delimiter=',')
    logreg_rfe_prediction_url = f'{RESULT_FOLDER}/{upload_id}/logreg_rfe_predictions.csv'
    np.savetxt(logreg_rfe_prediction_url,
               model_logreg_grid_predictions, delimiter=',')
    knn_prediction_url = f'{RESULT_FOLDER}/{upload_id}/knn_predictions.csv'
    np.savetxt(knn_prediction_url,
               model_knn_predictions, delimiter=',')
    decision_tree_prediction_url = f'{RESULT_FOLDER}/{upload_id}/decision_tree_predictions.csv'
    np.savetxt(decision_tree_prediction_url,
               model_decision_tree_predictions, delimiter=',')
    xg_boost_prediction_url = f'{RESULT_FOLDER}/{upload_id}/xg_boost_predictions.csv'
    np.savetxt(xg_boost_prediction_url,
               model_xg_boost_predictions, delimiter=',')
    gradient_boosting_prediction_url = f'{RESULT_FOLDER}/{upload_id}/gradient_boosting_predictions.csv'
    np.savetxt(gradient_boosting_prediction_url,
               model_gradient_boosting_predictions, delimiter=',')
    random_forest_prediction_url = f'{RESULT_FOLDER}/{upload_id}/random_forest_predictions.csv'
    np.savetxt(random_forest_prediction_url,
               model_random_forest_predictions, delimiter=',')

    # Get the accuracy
    model_logreg_rfe_accuracy = model_logreg_rfe.accuracy()
    model_logreg_grid_accuracy = model_logreg_grid.accuracy()
    model_knn_accuracy = model_knn.accuracy()
    model_decision_tree_accuracy = model_decision_tree.accuracy()
    model_xg_boost_accuracy = model_xg_boost.accuracy()
    model_gradient_boosting_accuracy = model_gradient_boosting.accuracy()
    model_random_forest_accuracy = model_random_forest.accuracy()

    # Get the confusion matrix
    model_logreg_rfe_confusion_matrix = model_logreg_rfe.confusion_matrix_base64()
    model_logreg_grid_confusion_matrix = model_logreg_grid.confusion_matrix_base64()
    model_knn_confusion_matrix = model_knn.confusion_matrix_base64()
    model_decision_tree_confusion_matrix = model_decision_tree.confusion_matrix_base64()
    model_xg_boost_confusion_matrix = model_xg_boost.confusion_matrix_base64()
    model_gradient_boosting_confusion_matrix = model_gradient_boosting.confusion_matrix_base64()
    model_random_forest_confusion_matrix = model_random_forest.confusion_matrix_base64()

    # Knn specific
    knn_mae = model_knn.mae()
    knn_mse = model_knn.mse()
    knn_rmse = model_knn.rmse()

    return render_template('results.html',
                           logreg_rfe_confusion_matrix_base64=model_logreg_rfe_confusion_matrix.decode(
                               'utf8'),
                           logreg_grid_confusion_matrix_base64=model_logreg_grid_confusion_matrix.decode(
                               'utf8'),
                           knn_confusion_matrix_base64=model_knn_confusion_matrix.decode(
                               'utf8'),
                           dt_confusion_matrix_base64=model_decision_tree_confusion_matrix.decode(
                               'utf8'),
                           xgb_confusion_matrix_base64=model_xg_boost_confusion_matrix.decode(
                               'utf8'),
                           gb_confusion_matrix_base64=model_gradient_boosting_confusion_matrix.decode(
                               'utf8'),
                           rdmf_confusion_matrix_base64=model_random_forest_confusion_matrix.decode(
                               'utf8'),
                           logreg_grid_accuracy=model_logreg_grid_accuracy,
                           logreg_rfe_accuracy=model_logreg_rfe_accuracy,
                           knn_mae=knn_mae,
                           knn_mse=knn_mse,
                           knn_rmse=knn_rmse,
                           knn_score=model_knn_score,
                           dt_score=model_decision_tree_score,
                           xgb_score=model_xg_boost_score,
                           gb_score=model_gradient_boosting_score,
                           rdmf_score=model_random_forest_score,
                           logreg_grid_prediction_url=f'/api/uploads/{upload_id}/logreg_grid',
                           logreg_rfe_prediction_url=f'/api/uploads/{upload_id}/logreg_rfe',
                           knn_prediction_url=f'/api/uploads/{upload_id}/knn',
                           dt_prediction_url=f'/api/uploads/{upload_id}/decision_tree',
                           xgb_prediction_url=f'/api/uploads/{upload_id}/xg_boost',
                           gb_prediction_url=f'/api/uploads/{upload_id}/gradient_boosting',
                           rdmf_prediction_url=f'/api/uploads/{upload_id}/random_forest',
                           )


@app.route('/api/uploads/<upload_id>/<model>', methods=['GET'])
def download(upload_id, model):
    model = f'{model}_predictions.csv'
    path = os.path.join(RESULT_FOLDER, upload_id)
    return send_from_directory(path, model)


if __name__ == "__main__":
    app.run(port=8000, debug=True)
