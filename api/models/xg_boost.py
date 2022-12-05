import io
import base64
import numpy as np
import seaborn as sns
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error
matplotlib.use('Agg')


plt = matplotlib.pyplot


class XGBoostModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # convert y_train and y_test to int
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)

        self.model = XGBClassifier()
        self.model = self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.last_prediction = self.model.predict(self.X_test)
        return self.last_prediction

    def score(self):
        return self.model.score(self.X_test, self.y_test)

    def accuracy(self):
        return accuracy_score(self.y_test.astype(int), self.last_prediction.astype(int))

    def confusion_matrix_base64(self):
        model_logreg_rfe_cm = confusion_matrix(
            self.y_test.astype(int), self.last_prediction.astype(int))

        f, ax = plt.subplots(figsize=(5, 5))

        sns.heatmap(model_logreg_rfe_cm, annot=True, linewidths=0.5,
                    linecolor="violet", fmt=".0f", ax=ax, cmap="BuPu")
        plt.xlabel("y_pred")
        plt.ylabel("y_true")

        plot_as_io_bytes = io.BytesIO()
        plt.savefig(plot_as_io_bytes, format='jpg')
        plot_as_io_bytes.seek(0)

        b64png = base64.b64encode(plot_as_io_bytes.read())

        # Clean up
        plot_as_io_bytes.close()
        plt.clf()

        return b64png
