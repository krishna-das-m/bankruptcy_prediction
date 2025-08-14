import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

test_size = 0.2
n_estimators = 100
max_depth = 5

# Different hyperparameter values
hyperparameter_values = [0.01, 0.1, 0.5, 1.0]

def performance(actual, pred):
    f1 = f1_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    return f1, accuracy, precision

if __name__ == "__main__":
    data = pd.read_csv(r'C:\Users\Admin\Projects\ML Projects\bankruptcy_prediction\data\data.csv')
    # Split the data into training and test sets
    train, test = train_test_split(data, test_size=test_size)

    # Create (X, y) data
    x_train = train.drop(["class"], axis=1)
    y_train = train["class"]
    
    x_test = test.drop(["class"], axis=1)
    y_test = test["class"]

    # create experiment
    my_exp = mlflow.set_experiment("baseline_random_forest")

    for lr in hyperparameter_values:
        # start mlflow run
        with mlflow.start_run(experiment_id=my_exp.experiment_id):

            # log parameters
            mlflow.log_param("learning_rate", lr)
            # Train model
            rf_model = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth)
            rf_model.fit(x_train, y_train)
        
            # Generate predictions
            predictions = rf_model.predict(x_test)

            # Determine performance metrics
            f1, accuracy, precision = performance(y_test, predictions)
            mlflow.log_params({"n_estimators":n_estimators,
                            "max_depth":max_depth})
            mlflow.log_metrics({"F1 score":f1,
                                "Accuracy": accuracy,
                                "Precision": precision})
            mlflow.sklearn.log_model(rf_model,"model")
        

    # Print model details
    print(f"Random Forest: {n_estimators=}, {max_depth=}")
    print(f"F1 score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")