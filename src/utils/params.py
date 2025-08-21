from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# ===== PARAMETER SPACE DEFINITIONS =====
# define all model parameters in dictionaries

PARAM_SPACES = {
    'RandomForest': {
        'n_estimators': {
            'type': 'int',
            'low': 50,
            'high': 500,
            'step': 50
        },
        'max_depth': {
            'type': 'int', 
            'low': 5,
            'high': 30
        },
        'min_samples_split': {
            'type': 'int',
            'low': 2,
            'high': 20
        },
        'min_samples_leaf': {
            'type': 'int',
            'low': 1,
            'high': 10
        },
        'max_features': {
            'type': 'categorical',
            'choices': ['sqrt', 'log2', None]
        },
        'bootstrap': {
            'type': 'categorical',
            'choices': [True, False]
        },
        'class_weight': {
            'type': 'categorical',
            'choices': ['balanced', 'balanced_subsample']
        }
    },
    
    'XGBoost': {
        'n_estimators': {
            'type': 'int',
            'low': 50,
            'high': 500,
            'step': 50
        },
        'max_depth': {
            'type': 'int',
            'low': 3,
            'high': 12
        },
        'learning_rate': {
            'type': 'float',
            'low': 0.01,
            'high': 0.3,
            'log': True
        },
        'subsample': {
            'type': 'float',
            'low': 0.6,
            'high': 1.0
        },
        'colsample_bytree': {
            'type': 'float',
            'low': 0.6,
            'high': 1.0
        },
        'reg_alpha': {
            'type': 'float',
            'low': 0.0,
            'high': 1.0
        },
        'reg_lambda': {
            'type': 'float',
            'low': 0.0,
            'high': 1.0
        }
    }
}

# ===== MODEL FACTORY =====
# Map model names to their classes and default parameters

MODEL_FACTORY = {
    'RandomForest': {
        'class': RandomForestClassifier,
        'default_params': {'random_state': 42, 'n_jobs': -1}
    },
    'XGBoost': {
        'class': xgb.XGBClassifier,
        'default_params': {'random_state': 42, 'eval_metric': 'logloss'}
    }
}

# ===== SCORING METRICS =====
# scoring options for cross-validation

SCORING_METRICS = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}