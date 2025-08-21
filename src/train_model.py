import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
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

def prepare_data(df):
    X =  df.drop(['class'], axis=1)
    y = df['class']

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # split the data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    return X_train, y_train, X_test, y_test


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
            # 1. Get suggested parameters using flexible function
            suggested_params = suggest_params_from_config(trial, param_space)
            
            # 2. Combine with default parameters
            all_model_params = {**model_info['default_params'], **suggested_params}

            # 3. smote hyperparameters
            max_k =  min(10, sum(y_train==1)-1)
            k_neighbors = trial.suggest_int('k_neighbors', 1, max_k) if max_k >1 else 1
            sampling_strategy = trial.suggest_float('smote_sampling_strategy', 0.1, 1.0)

            smote = SMOTE(sampling_strategy=sampling_strategy,k_neighbors=k_neighbors, random_state=42)
    
            # 4. Create model instance
            model = model_info['class'](**all_model_params)

            # 5. create a pipeline with smote and the chosen model
            steps = [('smote', smote), ('model', model)]
            pipeline = Pipeline(steps=steps)
            # pipeline.fit(X_train, y_train) # cause data leakage
            # apply smote to training data
            # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            # 6. Perform cross-validation
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring=scoring)
            mean_score = scores.mean()
            
            # 7. Log additional metrics for debugging
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
        params_to_optimize = list(PARAM_SPACES[model_name].keys()) + ['k_neighbors', 'sampling_strategy']
        print(f"Parameters to optimize: {params_to_optimize}")
        
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
        # model_params = {**model_info['default_params'], **best_params}
        model_params = {k: v for k, v in best_params.items() 
                        if not k.startswith(('k_neighbors', 'smote_'))}
        # smote_params = {k:v for k,v in best_params.items() if k.startswith(('k_neighbors','smote_'))}
        best_model = model_info['class'](**model_params)
        # Create SMOTE with best parameters
        best_smote = SMOTE(
            sampling_strategy=best_params.get('smote_sampling_strategy', 0.5),
            k_neighbors=best_params.get('k_neighbors', 5),
            random_state=42
        )
        # best_model.fit(X_train, y_train)
        # Create final pipeline   
        steps = [('smote', best_smote), ('model', best_model)]
        best_model = Pipeline(steps=steps)
        best_model.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(best_model, "best_model")
        
        print(f"\nOptimization completed!")
        print(f"Best {scoring_metric}: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        return best_model, study

def evaluate_model(model, X_test, y_test, model_name, exp_name):
    """
    Comprehensive model evaluation - separate and focused.
    """
    mlflow.set_experiment(experiment_name=exp_name)
    
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
    
def usage():
    """
    Show how to use the framework for different scenarios
    """
    
    # Load your data (replace with actual data loading)
    data = pd.read_csv(r'C:\Users\Admin\Projects\ML Projects\bankruptcy_prediction\data\data.csv')
    X_train, y_train, X_test, y_test = prepare_data(data)  # your existing function
    
    print("=== Optuna-MLflow Framework ===\n")
    
    # Example 1: RandomForest optimizing for recall
    print("Example 1: RandomForest for recall")
    exp_name1 = 'bankruptcy_prediction_recall_smote'
    best_model_1, study_1 = run_optimization(
        X_train, y_train, 
        model_name='RandomForest', 
        scoring_metric='recall',
        n_trials=100,
        experiment_name=exp_name1
    )
    
    # Example 2: XGBoost optimizing for F1-score  
    print("Example 2: XGBoost for F1-score")
    exp_name2 = 'bankruptcy_prediction_f1_smote'
    best_model_2, study_2 = run_optimization(
        X_train, y_train,
        model_name='XGBoost',
        scoring_metric='f1', 
        n_trials=100,
        experiment_name=exp_name2
    )
    
    # Example 3: Compare models in their respective experiments
    evaluate_model(best_model_1, X_test, y_test, 'RandomForest', exp_name1)
    evaluate_model(best_model_2, X_test, y_test, 'XGBoost', exp_name2)

if __name__ == "__main__":
    usage()

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