import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

#mlflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)
args = parser.parse_args()

#evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from local
    data = pd.read_csv("dataset/winequality.csv", sep=';')
    # data.to_csv("dataset/red-wine-quality.csv", index=False)
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    print(train.head(5))
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop("quality", axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # ML Flow experiment track

    exp = mlflow.set_experiment(experiment_name = "WineQualityExperiment")

    with mlflow.start_run(experiment_id = exp.experiment_id):

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2",r2)

        # mlflow.sklearn.log_model(lr, "Model_LR")
        signature = infer_signature(train_x, lr.predict(train_x))
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="Model_LR",  # Retain for backward compatibility in folder structure.
            signature=signature,
            input_example=train_x.iloc[[0]],  # Use the first row of training data as an example.
        )
