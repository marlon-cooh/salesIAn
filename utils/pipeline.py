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

subjects_9 = ['codigo', 'no_lista', 'nombre', 'periodo', 'esp', 'ingl', 'edufi', 'art', 'soc', 'ere', 'mat', 'nat', 'tecn','esc_pad', 'compo']
subjects_10 = ['codigo', 'no_lista', 'nombre', 'periodo', 'lect', 'esp', 'mat', 'econ', 'ingl', 'nat', 'fis', 'filo', 'poli', 'ere', 'edufi', 'tecn', 'compo']
subjects_11 = ['codigo', 'no_lista', 'nombre', 'periodo', 'lect', 'esp', 'mat', 'econ', 'ingl', 'qui', 'fis', 'filo', 'poli', 'ere', 'edufi', 'tecn', 'compo']

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


def remove_unregistered_students(raw_df:pd.DataFrame) -> pd.DataFrame:
    """
        Cleans the raw dataframe by removing unnecessary columns and rows that do not contain grading information along with students that are not listed in the courses.
        This function is designed for those dataframes obtained from .parquet files after reading and cleaning HTMLs (look parsing_html.py).
        Args:
            raw_df (pd.DataFrame): The raw dataframe to be cleaned.
        Returns:
            pd.DataFrame: The cleaned dataframe.    
    """
    # Dropping unnecessary columns.
    
    # Unnecessary columns are those that do not relate to grading.
    cols_to_drop = ['Competencia', 'OBS  1', 'OBS  2', 'OBS  3', 'OBS  4', 'OBS  5']
    
    # Keeping only the first 13 columns that relate to grading.
    cleaned_df = raw_df.drop(columns=cols_to_drop).iloc[:, :13]
    
    # # Replacing "None" and pd.NA values with NaN, then dropping rows that are completely empty.
    cleaned_df = cleaned_df.replace({"None", pd.NA}, inplace=False).reset_index().rename(columns={'index' : 'ESTUDIANTE'})

    # Resetting index and dropping completely empty rows, then creating an auxiliar foreign key.
    cleaned_df = cleaned_df.reset_index().rename(columns={'index' : 'ID'})
    cleaned_df = cleaned_df.dropna(how="all", subset=cleaned_df.columns[2:13], inplace=False).dropna(subset=cleaned_df.columns[9], how="all")
    
    # Merge to non-grading relative columns.
    to_merge = raw_df.iloc[:, 19:].reset_index().rename(columns={'index' : 'ESTUDIANTE'}).reset_index().rename(columns={'index' : 'ID'})
    merged = cleaned_df.merge(
        to_merge,
        on=[
            'ID', 
            'ESTUDIANTE'
        ],
        how='inner'
    )
    
    # Columns where missing values must be binarized.
    cols = ['Rec P1', 'Rec P2', 'Rec P3', 'Rec PF']
    merged[cols] = merged[cols].notna().astype("Int64")
    
    # Transforming columns to categorical dtype.
    order = ["S", "A", "B", "b"]
    cat_cols = ['CONOCER', 'HACER', 'SER', 'CONVIVIR', 'Subtotal NIVEL', 'Nota P1', 'Nota P2', 'Nota P3', 'Nota PF']
    merged[cat_cols] = merged[cat_cols].replace(
        {"None" : pd.NA, "": pd.NA}
    ).apply(
        lambda s: s.str.strip() if s.dtype =="object" else s
    ).apply(
        lambda s : pd.Categorical(s, categories=order, ordered=True)
    )

    return merged

def clean_level_grades(df: pd.DataFrame, cols_to_present: list) -> pd.DataFrame:
    """Clean and format grade level data.
    
    Args:
        df: Input DataFrame containing grade information
        cols_to_present: List of columns to keep in output
    
    Returns:
        pd.DataFrame: Cleaned and formatted grade data
    """
    return (df.clean_names(
        case_type='snake',
        strip_underscores=True,
        remove_special=True        
    ).loc[-10:, cols_to_present].reset_index().rename(columns={'index':'idx'})
    )

def safe_drop(df:pd.DataFrame, colname:str) -> pd.DataFrame:
    return df.drop(columns=[colname], errors='ignore')

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

def retrieve_grade_reports(inpath:str, cols_to_present=None, **kwargs) -> dict:
    """
        This function returns a complete, cleaned, and ready-to-eda dataframe from grade reports taken in .xls format from database.
        (Args):
            * inpath: Directory where .xls file is stored,
            * cols_to_present: List of columns to display in the final dataframe, for instance, subjects as: 'nat' (Natural sciences), 'esp' (Spanish), and so on. 
        Returns:
            A dictionary of dataframes suitable to eda and train simple models.
    """
    
    # Reading .xsl files
    inpath_ = Path(inpath)
    try:
        if inpath_.exists():
            tables = pd.read_html(
                inpath_,
                attrs={"id": "consolidadonotas_periodo_tabla"}
            )
    except (FileExistsError, UnicodeDecodeError) as fe:
        print(f"{fe}: Inpath is not valid and/or does not contain a readable .xls file")
    
    # Removing multiindex.    
    try:
        df = tables[0]
    except UnboundLocalError as u:
        print(f"Inpath is not valid and/or does not contain a readable .xls file \n {u}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    # Removing retired students.
    df['Nombre'] = df['Nombre'].astype('str')
    
    indexes_to_drop = df[df['Nombre'].apply(lambda x: bool(search(r"\t", x)))].index.to_list()
    df.drop(index=indexes_to_drop, axis=0, inplace=True)
     
    # Cleaning information.
    try:
        if search(r"11(?=0)", inpath):
            cols_to_present = subjects_11
        elif search(r"10(?=0)", inpath):
            cols_to_present = subjects_10
        elif search(r"9(?=0)", inpath):
            cols_to_present = subjects_9
        else:
            raise ValueError("File name must contain grade level (9, 10, or 11)")
           
        level_grades = clean_level_grades(df, cols_to_present)
    
    except (KeyError) as ke:
        print(f"Column error: {ke}. Check if columns match grade level.")
        
    level_grades = level_grades.dropna(
        axis=0,
        subset=level_grades.columns[2:],
        how='all'
    )
    
    # To segment by period.
    allowed_periods = {"P1", "P2", "P3", "PF"}

    for key, value in kwargs.items():
        if isinstance(value, str) and value in allowed_periods:
            level_grades[f'{key}'] = value.strip()
        else:
            raise ValueError(
                "Additional parameters MUST be period indicators: 'P1', 'P2', 'P3', or 'PF'."
            )
                
    # Creating dataframes for terms. (This has to be updated!)
    
    level_grades_p1 = level_grades[level_grades['idx'] %2 == 0].drop(columns={'no_lista'}, axis=1)
    level_grades_p2 = level_grades[level_grades['idx'] %2 != 0]
    
    # Assigning columns depending on selected level.
    new_labels = ['idx', 'codigo_p1', 'nombre_p1']
    new_labels += [elem + "_p2" for elem in level_grades.columns[5:]]
    
    # Base dict of columns_to_replace
    columns_to_replace = {
        'codigo_p1' : 'codigo',
        'nombre_p1' : 'nombre'
    }
    
    columns_to_replace.update(
        {
            x : x.removesuffix("_p2").removesuffix("_p1") for x in new_labels
        }
    )
    
    # Creating p2 report.
    level_grades_p2.loc[:, 'idx'] -= 1
    level_grades_p2 = level_grades_p2.merge(
            level_grades_p1,
            on='idx',
            how='inner',
            suffixes=("_p2", "_p1")
        )[new_labels].rename(
        columns=columns_to_replace
    )
        
    # Assign value 'P2' to period column in level_grades_p2
    level_grades_p2['periodo'] = 'P2'
            
    # Removing unnecessary columns
    if 'esc_pad' in level_grades_p1.columns or 'esc_pad' in level_grades_p2.columns:
        level_grades_p1 = safe_drop(level_grades_p1, 'esc_pad')
        level_grades_p2 = safe_drop(level_grades_p2, 'esc_pad')
        
    # Renaming columns to standard format.
    level_grades_p1 = level_grades_p1.rename(
        columns={
            "nat" : "qui"
        }
    )
    level_grades_p2 = level_grades_p2.rename(
        columns={
            "nat" : "qui"
        }
    )
        
    dfs = {
            "p1" : level_grades_p1, 
            "p2" : level_grades_p2
        }
        
    return dfs

def process_grades_columns(df:pd.DataFrame) -> pd.DataFrame:
    """
    This function returns an ordinal encoded version of grades in student report, this version is useful to create visualization or train simple ML models.
        Args:
        df: Ready-to-eda DataFrame after being cleaned through utils/pipeline.py function, retrieve_grade_reports().
        
        Returns:
        df: Refined eda DataFrame.
    """
    # Rearranging columns
    # Filtering columns
    if 'qui' not in set(df.columns.tolist()) and 'lect' in set(df.columns.tolist()):
        df = df[['idx', 'codigo', 'nombre', 'periodo', 'lect', 'esp', 'mat', 'econ', 'ingl', 'nat', 'fis', 'filo', 'poli', 'ere', 'edufi', 'tecn', 'compo']]
    
    elif 'nat' not in (df.columns.tolist()):
        df = df[['idx', 'codigo', 'nombre', 'periodo', 'lect', 'esp', 'mat', 'econ', 'ingl', 'qui', 'fis', 'filo', 'poli', 'ere', 'edufi', 'tecn', 'compo']]
    
    elif 'qui' not in set(df.columns.tolist()) and 'lect' not in set(df.columns.tolist()):
        df = df[['idx', 'codigo', 'nombre', 'periodo', 'esp', 'ingl', 'edufi', 'art', 'soc', 'ere', 'mat', 'nat', 'tecn', 'compo']]
    
    # If there are columns to drop
    df.drop(columns=[], inplace=True)
    
    # If there are missing values (np.nan)
    df.dropna(subset=df.columns[:], how='any', inplace=True)
    
    # Categorical variables
    cols_to_classify = df.columns[4:].tolist()
    categories_per_col = [['b', 'B', 'A', 'S']] * len(cols_to_classify)
    
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
            ('cat', cat_ord_pipe, cols_to_classify)
        ],
        remainder='drop',
        verbose_feature_names_out=True
    )
    
    pre.set_output(transform='pandas')
    processed_df = pre.fit_transform(df)
    
    # Removing added label (cat_) in processed_df
    processed_df.columns = [c.split("__", 1)[-1] for c in processed_df.columns]
    
    processed_df = pd.concat(
        [df.iloc[:, :4], processed_df],
        axis=1,
        ignore_index=False
    )
    
    return processed_df

def df_to_model(input_dfs:list) -> pd.DataFrame:
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
    input_dfs = [df.copy() for df in input_dfs]
    
    # Assigning 'performance' values (sum of transformed grades of every subject), this is OPTIONAL.
    for df in input_dfs:
        df['performance'] = df[df.columns[4:].tolist()].sum(axis=1)
        
    # Assigning 'fundamental' values (sum of transformed grades in español, matemáticas, lectura crítica), this is OPTIONAL
    fund_cols = ["mat", "esp", "lect"]
    existing_cols = [col for col in fund_cols if col in input_dfs[0].columns]
    
    if not existing_cols:
        print("⚠️ No matching columns found for 'fundamental'. Skipping creation.")
    else:
        for df in input_dfs:
            df['fundamental'] = df[existing_cols].sum(axis=1)
        
    # Creating a band category based on performance. 
    for df in input_dfs:
        
        # Cutpoints.
        q25, q50, q75, q90 = np.quantile(df["performance"], [0.25, 0.5, 0.75, 0.9])
        df['band'] = df['performance'].apply(
            lambda x: (
                'EXCELLENT' if x >= q90 else
                'GOOD'      if (x >= q75 and x < q90) else
                'MEDIUM'    if (x >= q50 and x < q75) else
                'LOW'       if (x >= q25 and x < q50) else
                'VERY LOW'
            ) if pd.notnull(x) else pd.NA
        )
        
        # Quick reclassification of `band` column.
        
        df['band'] = pd.Categorical(df['band']) # Convert to categorical.
        
        df.loc[df['band'] == 'LOW', 'band'] = 'MEDIUM' # Merging LOW and MEDIUM
        df.loc[df['band'] == 'VERY LOW', 'band'] = 'LOW' #Merging VERY LOW and LOW
        df['band'] = df['band'].cat.add_categories('HIGH')
        df.loc[df['band'].isin(['GOOD', 'EXCELLENT']), 'band'] = 'HIGH' #Merging GOOD and EXCELLENT as new category HIGH.
        df['band'] = df['band'].cat.remove_unused_categories()

        students_order = ['LOW', 'MEDIUM', 'HIGH'] # Processing `band` column
        df.band = pd.Categorical(
                    values=df.band,
                    categories=students_order,
                    ordered=True
            )
    
    # Creating a wrapped df that contains every cleaned df.
    wrapped_df = pd.concat(
        objs = [df for df in input_dfs],
        axis = 0
    )
    
    # Filtering columns
    if 'qui' not in set(wrapped_df.columns.tolist()) and 'lect' in set(wrapped_df.columns.tolist()):
        wrapped_df.select(
                  'idx', 'codigo', 'nombre', 'periodo', 'lect', 'esp', 'mat', 'econ', 'ingl', 'nat', 'fis', 'filo', 'poli', 'ere', 'edufi', 'tecn', 'compo', 'fundamental','band'
        )
    elif 'nat' not in (wrapped_df.columns.tolist()) and 'lect' in set(wrapped_df.columns.tolist()):
        wrapped_df.select(
        'idx', 'codigo', 'nombre', 'periodo', 'lect', 'esp', 'mat', 'econ', 'ingl', 'qui', 'fis', 'filo', 'poli', 'ere', 'edufi', 'tecn', 'compo', 'fundamental','band'
    )
    elif 'qui' not in set(wrapped_df.columns.tolist()) and 'lect' not in set(wrapped_df.columns.tolist()):
        wrapped_df.select(
        'idx', 'codigo', 'nombre', 'periodo', 'esp', 'ingl', 'edufi', 'art', 'soc', 'ere', 'mat', 'nat', 'tecn', 'compo' ,'fundamental','band'
    )
        
    # Codigo column to int
    wrapped_df['codigo'] = wrapped_df['codigo'].astype('int')
    
    return wrapped_df

if __name__ == "__main__":
    
    # Run cleaning pipeline
    
    # Paths
    grade_configs = [
        # {'grade': '9_2', 'path': '../consolidados/consolidado_902.xls', 'students_p1': 95, 'students_p2': 95},
        {'grade': '10_1', 'path': '../consolidados/consolidado_1001.xls', 'students_p1': 81, 'students_p2': 81},
        {'grade': '10_2', 'path': '../consolidados/consolidado_1002.xls', 'students_p1': 81, 'students_p2': 81},
        {'grade': '10_3', 'path': '../consolidados/consolidado_1003.xls', 'students_p1': 85, 'students_p2': 85},
        {'grade': '10_4', 'path': '../consolidados/consolidado_1004.xls', 'students_p1': 82, 'students_p2': 83},
        {'grade': '11_1', 'path': '../consolidados/consolidado_1101.xls', 'students_p1': 81, 'students_p2': 81},
        {'grade': '11_2', 'path': '../consolidados/consolidado_1102.xls', 'students_p1': 79, 'students_p2': 79},
        {'grade': '11_3', 'path': '../consolidados/consolidado_1103.xls', 'students_p1': 81, 'students_p2': 81},
    ]
        
    # Processed dataframes.
    processed_data = {}
    logger.info("--- Running cleaning pipeline ---")
    
    for config in grade_configs:
        # Setting parameters
        grade_name = config['grade']
        path = config['path']
        
        # Running retrieve_grades_reports.
        logger.info(f"Processing grade: {grade_name}...")
        processed_data_1 = process_grades_columns(
                                                retrieve_grade_reports(
                                                    inpath=path, 
                                                    period='P1'
                                                    )['p1']
                                                ).rename(columns={"nat":"qui"})
        processed_data_2 = process_grades_columns(
                                                    retrieve_grade_reports(
                                                        inpath=path, 
                                                        period='P2'
                                                    )['p2']
                                                ).rename(columns={"nat":"qui"})
        processed_df = df_to_model(input_dfs=[processed_data_1, processed_data_2])
        processed_data[grade_name] = processed_df

    df = pd.concat(
        objs = processed_data.values(),
        axis = 0
    )
       
    logger.info(f"Dataframes: {df.isna().sum()}")
    
    df.to_parquet("../cleaned_data/grade_summary.parquet")
    logger.info(f"Dataframe saved.")