from models.logistic_regression import LRMUsing, LogisticRegressionModel
import numpy as np
from uuid import uuid4
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
import os
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__)
app.secret_key = 'super secret key'

UPLOAD_FOLDER = '/Users/remikalbe/Git/github.com/ESILV-S7-PY-TDN/api/uploads'
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

    # Get the results
    model_logreg_rfe_score = model_logreg_rfe.score()
    model_logreg_grid_score = model_logreg_grid.score()

    # Get the predictions
    model_logreg_rfe_predictions = model_logreg_rfe.predict()
    model_logreg_grid_predictions = model_logreg_grid.predict()

    # Get the confusion matrix
    model_logreg_rfe_confusion_matrix = model_logreg_rfe.confusion_matrix_base64()
    model_logreg_grid_confusion_matrix = model_logreg_grid.confusion_matrix_base64()
    return render_template('results.html', logreg_rfe_confusion_matrix_base64=model_logreg_rfe_confusion_matrix.decode('utf8'))


if __name__ == "__main__":
    app.run(port=8000, debug=True)
