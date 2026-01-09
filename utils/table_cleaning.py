import pandas as pd
import os

# Cleaned data reading.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'mlops/cleaned_data', 'grade_summary.parquet'))

df = pd.read_parquet(MAIN_DIR)

# Subject columns.
subject_cols = df.columns[4:17].tolist()
subjects = (
    pd.DataFrame({"component" : subject_cols})
    .reset_index()
    .rename(columns={"index" : "id"})
)
subjects["id"] += 1

# Students table
student_base = df[
    [
        'codigo', #id
        'nombre', #name
        #age,
        #social_data,
        'band', #overall
        'fundamental', #fundamental
        'compo'
    ]
].reset_index().drop(columns=['index'])

# Hardcoded age and social data for testing purposes
student_base['age'] = 16
student_base['social_data'] = None

student_base = student_base[['codigo', 'nombre', 'age', 'social_data', 'fundamental', 'band', 'compo']]
student_base.rename(
    columns={
        "codigo" : "code",
        "nombre" : "name",
        "fundamental" : "fundamental_comp",
        "band" : "overall_grade",
        "compo" : "behavioral_comp"
    },
    inplace=True
)

student_base["disabilities"] = None
student_base["id"] = student_base.reset_index().rename(columns={"index": "id"}).loc[:, "id"].values + 1

student_base = student_base[
    [
        "id", 
        "code", "name", "age", "social_data", "fundamental_comp", "overall_grade", "behavioral_comp", "disabilities"
    ]
]

# Student to subject
student_to_subject = (
    df.melt(
        id_vars=["codigo", "periodo"],
        value_vars=subject_cols,
        var_name="component",
        value_name="grade"
    )
    .merge(subjects, on="component", how="left")
    .rename(columns={"codigo" : "student_id", "id": "subject_id", "periodo" : "term_id"})
)

student_to_subject.term_id = student_to_subject.term_id.map({"P1" : 1, "P2" : 2})
student_to_subject.drop(columns=["component"], inplace=True)

student_to_subject = pd.merge(
    student_base,
    student_to_subject,
    left_on="code",
    right_on="student_id",
    how="inner"
).rename(
    columns={
        "student_id" : "student_code"
    }
).rename(
    columns={"id" : "student_id"}
)

student_to_subject = student_to_subject[['student_id', 'subject_id', 'term_id', 'grade']]

# Term table
terms = pd.DataFrame(
    {
        "id" : [1, 2],
        "code" : ["P1", "P2"],
        "label" : ["Periodo 1", "Periodo 2"],
        "order" : [1, 2]
    }
)
# Save to JSON files
subjects.to_json(os.path.join(BASE_DIR, 'subjects.json'), orient='records', indent=4)
student_base.to_json(os.path.join(BASE_DIR, 'student_info.json'), orient='records', indent=4)
student_to_subject.to_json(os.path.join(BASE_DIR, 'student_to_subject.json'), orient='records', indent=4)
terms.to_json(os.path.join(BASE_DIR, 'terms.json'), orient='records', indent=4)
