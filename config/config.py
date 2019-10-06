lgb = {
    'boosting_type': 'goss',
    'max_leaves': 300,
    'max_depth': -1,
    'learning_rate': 0.01,
    'n_estimators': 300,
    'objective': 'multiclass',
    'class_weight': 'balanced',
    'random_state': 42,
    'early_stopping_rounds': 2,
    'zero_as_missing': True
}

gb = {
    'max_depth': None,
    'learning_rate': 0.01,
    'n_estimators': 500,
    'random_state': 42,
}

