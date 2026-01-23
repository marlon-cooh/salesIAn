# Standard python libraries
from datetime import datetime
from tqdm import tqdm

# sklearn libraries
import pandas as pd
import numpy as numpy
from mlflow.models import infer_signature
from prefect import flow, task

# Utils library
from utils.pipeline import remove_undesired_columns, split_by_level, generate_reports, terms_to_remove as terms

@task(retries=3, retry_delay_seconds=2,
      name="Remove columns that are not part of academic subjects.",
      tags=["cleaning"]
)
def drop_rubbish(input_path) -> pd.DataFrame:
    df_without_rubbish_columns = remove_undesired_columns(path=input_path)
    return df_without_rubbish_columns

@task(retries=3, retry_delay_seconds=2,
      name="Split large dataset into several smaller dataframes according to level.",
      tags=["cleaning"]
)
def list_of_levels(total_df) -> list:
    groups = split_by_level(df=total_df)
    return groups

@task(retries=3, retry_delay_seconds=2,
      name="Adjusts columns values that could introduce errors, as long as, dealing with missing values.",
      tags=["cleaning"]
)
def structure_reports(datasets:dict, destiny_path:str="./students/") -> None:
    # Updating datasets.
    datasets = {
        group:generate_reports(students) for group, students in datasets.items()
    }
    # Generating .csv for each group.    
    for group_name, dt in datasets.items():
        safe_group = str(group_name).replace(" ", "_")
        csv_name = os.path.join(destiny_path, f"report_{safe_group}.csv")
        dt.to_csv(csv_name, index=False)
    
if __name__ == "__main__":
    pass