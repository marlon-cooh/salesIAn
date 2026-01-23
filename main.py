# Standard python libraries
from datetime import datetime
from tqdm import tqdm
import os

# sklearn libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
import mlflow
from mlflow.models import infer_signature
from prefect import flow, task

# Utils library
from utils.pipeline import remove_undesired_columns, split_by_level, generate_reports, df_to_model, terms_to_remove as terms
from utils.log_data import logger_setup
from utils.registry import best_experiment, promote_model, test_model_from_mlflow

# DEFINE LOGGER
logger = logger_setup('process_track', 'process_track.log')

## 1 -- Cleaning pipeline ---
@task(retries=3, retry_delay_seconds=2,
      name="Remove columns that are not part of academic subjects.",
      tags=["cleaning"]
)
def remove_undesired_columns_task(input_path) -> pd.DataFrame:
    df_without_rubbish_columns = remove_undesired_columns(path=input_path)
    return df_without_rubbish_columns

@task(retries=3, retry_delay_seconds=2,
      name="Split large dataset into several smaller dataframes according to level.",
      tags=["cleaning"]
)
def split_by_level_task(total_df) -> list:
    groups = split_by_level(df=total_df)
    return groups

@task(retries=3, retry_delay_seconds=2,
      name="Adjusts columns values that could introduce errors, as long as, dealing with missing values.",
      tags=["cleaning"]
)
def generate_reports_task(datasets:dict, destiny_path:str="./students/") -> None:
    # Updating datasets.
    datasets = {
        group:generate_reports(students) for group, students in datasets.items()
    }
    # Generating .csv for each group.    
    for group_name, dt in datasets.items():
        safe_group = str(group_name).replace(" ", "_")
        csv_name = os.path.join(destiny_path, f"report_{safe_group}.csv")
        dt.to_csv(csv_name, index=False)
        
@flow(name="Data Cleaning Pipeline", log_prints=True)
def data_cleaning_pipeline(input_path:str, destiny_path:str="./students/") -> None:
    """
        Main flow to clean student grade reports.
        Args:
            input_path (str): Path to the input Excel file containing student grade reports.
            destiny_path (str): Path to save the cleaned CSV files. Default is "./students/".
    """
    # Step 1: Remove undesired columns.
    cleaned_df = remove_undesired_columns_task(input_path=input_path)
    
    # Step 2: Split DataFrame by levels.
    levels_dict = split_by_level_task(total_df=cleaned_df)
    
    # Step 3: Generate cleaned reports and save as CSV.
    generate_reports_task(datasets=levels_dict, destiny_path=destiny_path)

## 2 -- Preprocessing and training ---
@task(retries=3, retry_delay_seconds=2,
      name="Prepare DataFrame for model training.",
      tags=["preprocessing"]
)
def prepare_for_model(dt:pd.DataFrame) -> pd.DataFrame:
    model_df = df_to_model(dt=dt)
    return model_df

@task(retries=3, retry_delay_seconds=2,
      name="Split data to train model.", 
      tags=["training"])
def split_data(dt:list, col:list = 'band'):
    # Implementation for refining train data goes here
    X = dt.drop(columns=[col])
    y = dt[col].to_numpy().ravel()
            
    # Train-test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

@task(retries=3, retry_delay_seconds=2,
      name="Run experiments in MLflow.", 
      tags=["training"])
def run_experiments_in_mlflow_task(train_info:tuple, exp_list:dict, register_model:str) -> None:
    """ 
        Runs multiple experiments in MLflow based on provided configurations.
        Args:
            train_info (tuple): Tuple containing training and testing data.
            exp_list (dict): Dictionary with experiment names as keys and configurations as values.
            register_model (str): Name to register the trained model under.
    """
    X_train, X_test, y_train, y_test = train_info
    
    # Finding best model.
    for exp_key, config in tqdm(exp_list.items(), desc="Training models"):
        
        experiment_name = config['experiment_name']
        
        # Hyperparameter tuning if param_grid is provided.
        param_grid = config['param_grid']
        if param_grid:
            estimator = GridSearchCV(
                estimator=config['model'](),
                param_grid=param_grid,
                scoring='accuracy',
                cv=5,
                n_jobs=1 # Use 1 job to avoid overloading the system.
            )
        else:
            estimator = config['model']
            
        # Training model.
        estimator.fit(X_train, y_train)
        
        # Predict on the test set.
        y_pred = estimator.predict(X_test)
        
        # Calculate metrics.
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        logger.info(f"Metrics for experiment {exp_key} were \n * F1-score: {f1} \n * Accuracy: {acc}")
        
        # Create MLflow experiment and log results.
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=experiment_name):
            
            # Log the hyperparameters if hyperparameter tuning was done.
            if isinstance(estimator, GridSearchCV):
                mlflow.log_params(estimator.best_params_)
                model_to_log = estimator.best_estimator_
                logger.info(f"Best parameters for {exp_key}: {estimator.best_params_}")
                logger.info(f"Best estimator for {exp_key}: {estimator.best_estimator_}")
            else:
                model_to_log = estimator
                mlflow.log_params(estimator.get_params())
                logger.info(f"Parameters for {exp_key}: {estimator.get_params()}")
                
            # Log metrics.
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            
            # Infer model signature.
            la_holanda_signature = infer_signature(X_train, model_to_log.predict(X_train))
            
            # Tag to remember the model type.
            mlflow.set_tag("Training info", f"{config['model']} for La Holanda dataset")
            
            # Log the model, which inherits the parameter and metric.
            model_info = mlflow.sklearn.log_model(
                sk_model=estimator,
                artifact_path="la_holanda_model",
                input_example=X_train[:20],
                signature=la_holanda_signature,
                registered_model_name=register_model, # Name to register the model under.
            )
            
            logger.info(f"Experiment {exp_key} logged in MLflow with run ID: {model_info.run_id}")
        
        logger.info("Model comparison experiments completed.")
        logger.info("MLflow server terminated.")

## 3 -- Testing --
@task(retries=3, retry_delay_seconds=2,
      name="Perform sanity check.",
      tags=["testing"]
)
def sanity_check_task(strategy:str, X_train:np.ndarray, y_train:np.ndarray) -> None:
    dummy = DummyClassifier(strategy=strategy)
    dummy.fit(X_train, y_train)
    predictions = dummy.predict(X_train)
    return f1_score(y_train, predictions, average='weighted')

@task(retries=3, retry_delay_seconds=2,
      name="Register best model respect to a selected metric (i.e. f1, roc-auc, etc.) from a list of experiments in Mlflow.",
      tags=["testing"]
)
def best_experiment_task(tracking_uri:str, metric:str='f1 score') -> tuple:
    client, best_run_id = best_experiment(tracking_uri=tracking_uri, metric=metric)
    return client, best_run_id

## 4 -- Production --
@task(retries=3, retry_delay_seconds=2,
      name="Promote best model to Production.",
      tags=["production"])
def promote_model_task(
    client:mlflow.client.MlflowClient,
    best_run_id:str,
    model_name:str,
    stage:str
):
    promotion_status = promote_model(client=client, best_run_id=best_run_id, model_name=model_name, stage=stage)
    logger.info(promotion_status)
    
@task(retries=3, retry_delay_seconds=2,
      name="Test promoted model to production stage.",
      tags=["production"])
def test_promoted_model_task(model_name:str, stage:str, X_test:np.ndarray, y_test:np.ndarray) -> dict:
    
    f1 = test_model_from_mlflow(model_name=model_name, stage=stage, X_test=X_test, y_test=y_test)
    return {"f1-score" : f1}

@flow
def deployment_pipeline(tracking_uri:str, model_name:str, metric:str='f1 score') -> None:
    
    client_db, best_run_id = best_experiment_task(tracking_uri=tracking_uri, metric=metric)
    logger.info(f"{client_db} was successfully connected. With best experiment tag as {best_run_id}.")
    
    # Client for promotion
    client_uri = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    
    # Stage settings.
    set_stage = "Production"
    
    response = promote_model_task(
        client=client_uri,
        best_run_id=best_run_id,
        model_name=model_name,
        stage=set_stage
    )
    logger.info(f"Promotion response: {response}")
    
if __name__ == "__main__":
    
    # Experiments
    EXPERIMENTS = {
    "logreg_simple": {
        "experiment_name": "LaHolanda_LogReg_Simple",
        "model": LogisticRegression(random_state=42, max_iter=2000),
        "param_grid": None,
    },
    "logreg_grid": {
        "experiment_name": "LaHolanda_LogReg_Grid",
        "model": LogisticRegression(random_state=42, max_iter=2000),
        "param_grid": {
            "C": [0.001, 0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        },
    },
    "rf_simple": {
        "experiment_name": "LaHolanda_RF_Simple",
        "model": RandomForestClassifier(random_state=42),
        "param_grid": None,
    },
    "rf_grid": {
        "experiment_name": "LaHolanda_RF_Grid",
        "model": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 200, 500],
            "max_depth": [10, 20, 50, 100],
            "criterion": ["gini", "entropy"],
        },
    },
    # later: "logreg_boosting", "naive_bayes", ...
    }
    
    datasets = data_cleaning_pipeline(input_path='./consolidados.csv')