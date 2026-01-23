import pandas as pd #type:ignore
from pathlib import Path
import logging
import janitor
from re import search
import numpy as np #type:ignore

# Preprocessing.
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer #type:ignore
from sklearn.compose import ColumnTransformer #type:ignore
from sklearn.pipeline import Pipeline #type:ignore
from sklearn.impute import SimpleImputer #type:ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Levels in the institution.
levels = ["5-1", "6-1", "6-2", "6-3", "6-4", "7-1", "7-2", "7-3", "7-4", "8-1", "8-2", "8-3", "9-1", "9-2", "9-3", "10-1", "10-2", "10-3", "10-4", "11-1", "11-2", "11-3"]
columns_to_drop_in_datasets = ['esc_pad', 'electiva', 's', 'a', 'b', 'nan', 'b.1', 'nan', 'nan.1', 'nan.2', 'nan.3']
terms_to_clean = [
    "GRUPO", 
    "SUPERIOR", 
    "ALTO", 
    "BAJO", 
    "INSTITUCIIN",
    "AGRICOLA",
    "HOLANDA", 
    "CONSOLIDADO",
    "PROCESO",
    r"\(\s*[A-Za-z]\s*\)"
]

def remove_undesired_columns(path:str, terms_to_remove:list=terms_to_clean) -> pd.DataFrame:
    """
        Function that reads a CSV file, cleans specific rows based on given terms,
        and returns a cleaned DataFrame.
        Parameters:
        - path (str): Path to the CSV file.
        - terms_to_remove (list): List of terms to identify rows for removal.
        Returns:
        - pd.DataFrame: Cleaned DataFrame.
    """
    # Reading dataframe.
    data = pd.read_csv(path, encoding="cp1252", header=2)
    
    # Renaming columns for posterior analysis.
    data.rename(
        columns={f"Unnamed: {x}": f"{x}" for x in range(0, 24)},
        inplace=True
    )
    data.rename(
        columns={data.columns[2]:"process"},
        inplace=True
    )
    
    # Cleaning characters
    s_clean = (
        data.process.fillna('')
        .str.normalize('NFD')
        .str.replace(r'[\u0300-\u036f]', '', regex=True)
        .str.upper()
    )
    data["process"] = s_clean
    
    # Masks to remove undesired rows.
    patterns = "|".join(terms_to_remove)
    mask = data["process"].str.contains(patterns, regex=True, na=False)

    rows_to_remove = data[mask].index.tolist()
    data = data.drop(index=rows_to_remove)
    
    # Renaming columns based on first row values.
    data.columns = data.iloc[0].values.tolist()
    data = data.clean_names()

    return data

def split_by_level(df:pd.DataFrame) -> dict:
    """
        Splits a DataFrame into multiple DataFrames based on predefined levels.
        Args:
            df (pd.DataFrame): The input DataFrame to be split.
        Returns:
            dict: A dictionary where keys are levels and values are the corresponding DataFrames.
    """
    
    cleaned_data = df.copy() # To be reviewed.
    cleaned_data.reset_index(drop=True, inplace=True)
    
    # Rows to split each level.
    rows_for_each_level = cleaned_data[cleaned_data.codigo.str.contains("codigo", na=False, case=False)].index.tolist()
    rows_for_each_level.pop(-1) # This is removed because it does not make part of this dataset.
    
    # Matching levels with rows to split dataframe
    levels_dict = {x:y for x,y in zip(levels, rows_for_each_level)}
    
    # Assigning dictionary.
    list_cleaned_dfs = {}
    for number, key in enumerate(levels_dict):
        start = levels_dict[levels[number]]
        end = levels_dict[levels[number+1]] if number+1 < len(levels) else None
        list_cleaned_dfs[key] = cleaned_data.iloc[start:end, :]
    
    for key, dataset in list_cleaned_dfs.items():
        dataset = dataset.copy()
        
        dataset.columns = dataset.iloc[0].tolist()    
        dataset = dataset.clean_names()

        mask = dataset["no_lista"].astype(str).str.contains(r"\bNo\b", na=False)
        dataset = dataset.loc[~mask].copy()
        
        list_cleaned_dfs[key] = dataset
        
    return list_cleaned_dfs

def generate_reports(dataset:pd.DataFrame) -> pd.DataFrame:
    """
        Generates a cleaned and formatted report from the input dataset.
        Args:
            dataset (pd.DataFrame): The input dataset to be processed.
        Returns:
            pd.DataFrame: The cleaned and formatted report.
    """

    # Remove unregistered students or dropouts!
    rm_stud_idx = dataset[dataset.nombre.str.contains(
        r"\(\w+\)",
        regex=True,
        case=False,
        na=False        
    )].index.tolist()
    
    dropout_students_info = []
    for idx in rm_stud_idx: # indexers
        for subidx in range(idx, idx+4):
            dropout_students_info.append(subidx)
            
    dataset.loc[dropout_students_info, :]
    dataset.drop(index=dropout_students_info, inplace=True)
    
    # Replacing "pendie" for "PF"
    pendie_cond = dataset["periodo"] == "Pendie"
    dataset.loc[pendie_cond, "periodo"] = "PF"
    
    # Replacing empty spaces in nombre
    empty_cond = dataset["nombre"] == ""
    dataset.loc[empty_cond, "nombre"] = np.nan
    
    # Removing unnecessary columns.
    dataset.drop(columns=columns_to_drop_in_datasets, errors="ignore", inplace=True)
    
    # Treating missing values.
    dataset.dropna(axis=0, how="all", inplace=True)
    dataset[["codigo", "no_lista", "nombre"]] = dataset[["codigo", "no_lista", "nombre"]].ffill()
    dataset.iloc[:, 4:] = dataset.iloc[:, 4:].bfill()
    
    return dataset

def df_to_model(dt:pd.DataFrame) -> pd.DataFrame:
    """
    Prepares student grade DataFrames for machine learning model training by computing performance metrics
    and optionally categorizing students into performance bands.

    Args:
        input_dfs (list): List of pandas DataFrames containing student grades. Each DataFrame should:
            - Have student identifiers in columns 0-3
            - Contain subject grades in columns 4+
            - Be pre-processed using retrieve_grade_reports() and process_grades_columns()

    Returns:
        pd.DataFrame: A concatenated DataFrame with additional features:
            - performance: Sum of grades across all subjects
            - fundamental: Sum of grades in core subjects (math, spanish, critical reading)
            - band: Categorical performance level
            - Filtered to include only relevant subject columns based on grade level

    Features:
        - Automatically detects available subjects and adapts column selection
        - Handles missing core subjects when calculating fundamentals
        - Creates ordered categorical bands using percentile thresholds:
    MODIFY THIS ⚠️⚠️⚠️⚠️
            * EXCELLENT: >= 90th percentile
            * GOOD: >= 70th percentile
            * MEDIUM: >= 40th percentile 
            * LOW: < 40th percentile

    Example:
        >>> p1_df = process_grades_columns(retrieve_grade_reports("grade10.xls")["p1"])
        >>> p2_df = process_grades_columns(retrieve_grade_reports("grade10.xls")["p2"]) 
        >>> model_df = df_to_model([p1_df, p2_df], expose_band=True)
    """
    # Searching for columns to be removed.                
    cols_to_drop = list(set(['codigo', 'nombre', 'no_lista', 'periodo']))
    subject_cols = [col for col in dt.columns if col not in cols_to_drop]
    
    # Categorical variables
    categories_per_col = [['b', 'B', 'A', 'S']] * len(subject_cols)
    
    # Ordinal encoder pipeline.
    cat_ord_pipe = Pipeline(
    steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("ordinal", OrdinalEncoder(
            categories=categories_per_col,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            dtype=np.float64
            ))
        ]
    )
    
    # CT for ordinal pipeline.
    pre = ColumnTransformer(
    transformers=[
            ('cat', cat_ord_pipe, subject_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=True
    )
    
    pre.set_output(transform='pandas')
    processed_dt = pre.fit_transform(dt)
    processed_dt.columns = [c.split("__", 1)[-1] for c in processed_dt.columns]
    
    # Concatenating non-processed columns with processed ones.
    final_dt = pd.concat(
        [
            dt.loc[:, cols_to_drop],
            processed_dt
        ],
        axis=1
    )

    # Assigning 'fundamental' values (sum of transformed grades in español, matemáticas, lectura crítica), this is OPTIONAL
    fund_cols = ["mat", "esp", "lect"]
    final_dt['fundamental'] = final_dt.loc[:, list(set(fund_cols) & set(final_dt.columns))].sum(axis=1)

    # Assigning 'performance' values (sum of all transformed grades).
    final_dt['performance'] = final_dt.loc[:, subject_cols].sum(axis=1)
    
    # Cutpoints.
    q25, q50, q75, q90 = np.quantile(final_dt["performance"], [0.25, 0.5, 0.75, 0.9])
    final_dt['band'] = final_dt['performance'].apply(
        lambda x: (
            'EXCELLENT' if x >= q90 else
            'GOOD'      if (x >= q75 and x < q90) else
            'MEDIUM'    if (x >= q50 and x < q75) else
            'LOW'       if (x >= q25 and x < q50) else
            'VERY LOW'
        ) if pd.notnull(x) else pd.NA
    )
    
    # Merging categories (under review)
    final_dt['band'] = pd.Categorical(final_dt['band']) # Convert to categorical.

    final_dt.loc[final_dt['band'] == 'LOW', 'band'] = 'MEDIUM' # Merging LOW and MEDIUM
    final_dt.loc[final_dt['band'] == 'VERY LOW', 'band'] = 'LOW' #Merging VERY LOW and LOW
    final_dt['band'] = final_dt['band'].cat.add_categories('HIGH')
    final_dt.loc[final_dt['band'].isin(['GOOD', 'EXCELLENT']), 'band'] = 'HIGH' #Merging GOOD and EXCELLENT as new category HIGH.
    final_dt['band'] = final_dt['band'].cat.remove_unused_categories()
    
    students_order = ['LOW', 'MEDIUM', 'HIGH'] # Processing `band` column
    final_dt.band = pd.Categorical(
                values=final_dt.band,
                categories=students_order,
                ordered=True
        )
    
    return final_dt