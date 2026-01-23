import mlflow
from scipy.sparse import csr_matrix
from mlflow.tracking import MlflowClient
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from utils.log_data import logger_setup

# ---- 1. Configuration ----
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- 2. Functions ---
def test_model_from_mlflow(model_name:str, stage:str, X_test:csr_matrix, y_test) -> None:
    """this function tests a model from mlflow
    Args:
        model_name (str): name of the model
        stage (str): stage of the model
        X_test (scipy.sparse._csr.csr_matrix): test data
        Y_test (scipy.sparse._csr.csr_matrix): test target
    Returns:
        float: rmse of the model
    
    """
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    y_pred = model.predict(X_test)
    f1 = round(f1_score(y_test, y_pred, average='weighted'), 2)
    return {"f1_score": f1}

def best_experiment(tracking_uri:str, metric:str) -> str:
    """This function retrieves experiment_id from best experiment based on a selected tracking metric as f1 score or accuracy."""
    
    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments()
    
    # if amount of experiments is more than 1:
    scores = {}
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id]
        )
        scores.update({r.info.run_id:r.data.metrics[metric] for r in runs})
    best_run_id = max(scores, key=scores.get)
    
    return client, best_run_id

def promote_model(
    client:mlflow.tracking.client.MlflowClient,
    best_run_id:str,
    stage:str,
    version:int,
    model_name:str
):
    """
        this function promotes the best model to production.
    Args:
        * client (mlflow.tracking.client.MlflowClient): mlflow tracking client,
        * best_run_id: tag pointing run with best performance according to usage in `best_experiment()` function,
        * model_name (str): model name,
        * version (int): model version,
        * stage (str): Model stage, one of:
            - "None": Initial stage (default)
            - "Staging": Model in staging/testing phase
            - "Production": Model in production
            - "Archived": Model archived/deprecated
        
    """
    
    # Logger
    version_logger = logger_setup('version_log', 'version_log.log')
    
    # Registering best model.
    mlflow.register_model(model_uri=f"runs:/{best_run_id}/la_holanda_model", name=model_name)
    latest_versions = client.get_latest_versions(name=model_name)
    for version_ in latest_versions:
          version_logger.info(f"Model: {model_name}, Version: {version_.version}")
    
    # Transitioning model to `stage`
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    
    client.update_model_version(
        name=model_name,
        version=version,
        description=f"Model {version_.version} was transitioned to Production on {datetime.today().date()}"
    )
    
    return f"The model version {version_.version} was transitioned to Production on {datetime.today().date()}"