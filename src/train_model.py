import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from utils import params
import time

import mlflow
import mlflow.sklearn

import optuna
from optuna.integration.mlflow import MLflowCallback

PARAM_SPACES = params.PARAM_SPACES # parameter space for different models
MODELS = params.MODEL_FACTORY # different models
SCORING_METRICS = params.SCORING_METRICS # various performance metrics

def suggest_params_from_config(trial, param_space):
    """
    Convert parameter space dictionary into Optuna trial suggestions.
    This is the magic function that makes everything flexible!
    """
    suggested_params = {}
    
    for param_name, config in param_space.items():
        if config['type'] == 'int':
            # Handle integer parameters
            if 'step' in config:
                value = trial.suggest_int(param_name, config['low'], config['high'], step=config['step'])
            else:
                value = trial.suggest_int(param_name, config['low'], config['high'])
                
        elif config['type'] == 'float':
            # Handle float parameters
            if config.get('log', False):
                value = trial.suggest_float(param_name, config['low'], config['high'], log=True)
            else:
                value = trial.suggest_float(param_name, config['low'], config['high'])
                
        elif config['type'] == 'categorical':
            # Handle categorical parameters
            value = trial.suggest_categorical(param_name, config['choices'])
            
        else:
            raise ValueError(f"Unknown parameter type: {config['type']}")
            
        suggested_params[param_name] = value
    
    return suggested_params

def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.
    """
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

exp_id = get_or_create_experiment('bankruptcy_prediction')

# Set the current active MLflow experiment
mlflow.set_experiment(experiment_id=exp_id)

    
def performance(actual, pred):
    f1 = f1_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)

    return f1, accuracy, precision, recall

def prepare_data(df):
    train, test = train_test_split(df, test_size=0.2)
    # create X, y data
    x_train = train.drop(["class"], axis=1)
    y_train = train['class']

    x_test = test.drop(["class"], axis=1)
    y_test = test['class']

    return x_train, y_train, x_test, y_test


def create_flexible_objective(X_train, y_train, model_name, scoring_metric='recall', cv_folds=5):
    """
    Create an objective function that works with any model and any metric.
    This solves your flexibility problem!
    """
    
    # Validate inputs
    if model_name not in PARAM_SPACES:
        raise ValueError(f"Model {model_name} not supported. Available: {list(PARAM_SPACES.keys())}")
    
    if scoring_metric not in SCORING_METRICS:
        raise ValueError(f"Scoring {scoring_metric} not supported. Available: {list(SCORING_METRICS.keys())}")
    
    param_space = PARAM_SPACES[model_name]
    model_info = MODELS[model_name]
    scoring = SCORING_METRICS[scoring_metric]
    
    def objective(trial):
        start_time = time.time()
        
        try:
            # 1. Get suggested parameters using our flexible function
            suggested_params = suggest_params_from_config(trial, param_space)
            
            # 2. Combine with default parameters
            all_params = {**model_info['default_params'], **suggested_params}
            
            # 3. Create model instance
            model = model_info['class'](**all_params)
            
            # 4. Perform cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
            mean_score = scores.mean()
            
            # 5. Log additional metrics for debugging
            trial_time = time.time() - start_time
            trial.set_user_attr("trial_duration", trial_time)
            trial.set_user_attr("cv_std", scores.std())
            
            # Log to MLflow (will be handled by callback, but we can add extra info)
            if mlflow.active_run():
                mlflow.log_metric("cv_std", scores.std())
                mlflow.log_metric("trial_duration", trial_time)
            
            return mean_score
            
        except Exception as e:
            # Enhanced debugging - log what went wrong
            print(f"Trial {trial.number} failed: {str(e)}")
            trial.set_user_attr("error", str(e))
            raise optuna.TrialPruned()
    
    return objective

### HYPERPARAMETER OPTIMIZATION ###
def run_optimization(X_train, y_train, model_name='RandomForest', scoring_metric='recall', 
                    n_trials=100, experiment_name=None):
    """
    Main optimization function - clean and focused.
    This solves your debugging and understanding problems!
    """
    
    # Setup MLflow experiment
    if experiment_name:
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_id=experiment_id)
    
    with mlflow.start_run(run_name=f'{model_name}_optimization'):
        
        # Log configuration for debugging
        mlflow.log_params({
            "model_name": model_name,
            "scoring_metric": scoring_metric,
            "n_trials": n_trials,
            "cv_folds": 5,
            "optimization_method": "Optuna_TPE"
        })
        
        # Create study with enhanced configuration
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=1
            )
        )
        
        # Setup MLflow callback
        mlflc = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name=scoring_metric,
            create_experiment=False,
            mlflow_kwargs={"nested": True}
        )
        
        # Create objective function
        objective = create_flexible_objective(X_train, y_train, model_name, scoring_metric)
        
        # Run optimization with progress tracking
        print(f"\n=== Starting {model_name} optimization for {scoring_metric} ===")
        print(f"Parameters to optimize: {list(PARAM_SPACES[model_name].keys())}")
        
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[mlflc],
            show_progress_bar=True
        )
        
        # Log results and create final model
        best_params = study.best_params
        best_score = study.best_value
        
        # Log optimization results
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric(f"best_{scoring_metric}", best_score)
        mlflow.log_metric("n_completed_trials", len(study.trials))
        
        # Create and log best model
        model_info = MODELS[model_name]
        all_params = {**model_info['default_params'], **best_params}
        best_model = model_info['class'](**all_params)
        best_model.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(best_model, "best_model")
        
        print(f"\nOptimization completed!")
        print(f"Best {scoring_metric}: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        return best_model, study

def evaluate_model(model, X_test, y_test, model_name):
    """
    Comprehensive model evaluation - separate and focused.
    """
    
    with mlflow.start_run(run_name=f'{model_name}_evaluation'):
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate all metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        # Log all metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)
        
        # Log model info
        mlflow.log_param("model_type", model_name)
        
        print(f"\n=== {model_name} Test Results ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name.capitalize()}: {value:.4f}")
        
        return metrics
    
def example_usage():
    """
    Show how to use the flexible framework for different scenarios
    """
    
    # Load your data (replace with actual data loading)
    data = pd.read_csv(r'C:\Users\Admin\Projects\ML Projects\bankruptcy_prediction\data\data.csv')
    X_train, y_train, X_test, y_test = prepare_data(data)  # your existing function
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("=== Flexible Optuna-MLflow Framework Examples ===\n")
    
    # Example 1: RandomForest optimizing for recall
    print("Example 1: RandomForest for recall")
    best_model_1, study_1 = run_optimization(
        X_train, y_train, 
        model_name='RandomForest', 
        scoring_metric='recall',
        n_trials=5,
        experiment_name='bankruptcy_prediction_recall'
    )
    
    # Example 2: XGBoost optimizing for F1-score  
    print("Example 2: XGBoost for F1-score")
    best_model_2, study_2 = run_optimization(
        X_train, y_train,
        model_name='XGBoost',
        scoring_metric='f1', 
        n_trials=5,
        experiment_name='bankruptcy_prediction_f1'
    )
    
    # Example 3: Compare models
    evaluate_model(best_model_1, X_test, y_test, 'RandomForest')
    evaluate_model(best_model_2, X_test, y_test, 'XGBoost')

if __name__ == "__main__":
    example_usage()

# def train_baseline_model(data):
#     X_train, y_train, x_test, y_test = prepare_data(data)

#     with mlflow.start_run(experiment_id=exp_id, run_name='baseline_random_forest'):
#         # log parameters
#         mlflow.log_param()

# if __name__ == "__main__":
#     data = pd.read_csv(r'C:\Users\Admin\Projects\ML Projects\bankruptcy_prediction\data\data.csv')
#     # Split the data into training and test sets
#     train, test = train_test_split(data, test_size=test_size)

#     # Create (X, y) data
#     x_train = train.drop(["class"], axis=1)
#     y_train = train["class"]
    
#     x_test = test.drop(["class"], axis=1)
#     y_test = test["class"]

#     # create experiment
#     my_exp = mlflow.set_experiment("baseline_random_forest")

#     for lr in hyperparameter_values:
#         # start mlflow run
#         with mlflow.start_run(experiment_id=my_exp.experiment_id):

#             # log parameters
#             mlflow.log_param("learning_rate", lr)
#             # Train model
#             rf_model = RandomForestClassifier(n_estimators=n_estimators,
#                                             max_depth=max_depth,)
#             rf_model.fit(x_train, y_train)
        
#             # Generate predictions
#             predictions = rf_model.predict(x_test)

#             # Determine performance metrics
#             f1, accuracy, precision = performance(y_test, predictions)
#             mlflow.log_params({"n_estimators":n_estimators,
#                             "max_depth":max_depth})
#             mlflow.log_metrics({"F1 score":f1,
#                                 "Accuracy": accuracy,
#                                 "Precision": precision})
#             mlflow.sklearn.log_model(rf_model,"model")
        

#     # Print model details
#     print(f"Random Forest: {n_estimators=}, {max_depth=}")
#     print(f"F1 score: {f1}")
#     print(f"Accuracy: {accuracy}")
#     print(f"Precision: {precision}")