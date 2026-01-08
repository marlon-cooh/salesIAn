from sqlmodel import SQLModel, Field, Relationship

class StudentBase(SQLModel):
    code : int | None = None
    name : str | None = None
    age : int
    social_data : int | None = None
    overall_grade : str
    fundamental_comp : int
    behavioral_comp : int
    disabilities : str | None
    
class StudentSubjectLink(SQLModel, table=True):
    __tablename__ = "student_to_subject"
    student_id : int = Field(foreign_key="student.id", primary_key=True)
    subject_id : int = Field(foreign_key="subject.id", primary_key=True)
    term_id : int = Field(foreign_key="term.id", primary_key=True)
    grade : int # (1-7-2026, just for means of testing, this will be int type)
    term : 'Term' = Relationship(back_populates='student_links') 
    
class Term(SQLModel, table=True):
    __tablename__ = "term"
    id : int | None = Field(default=None, primary_key=True)
    code : str # e.g., P1, P2, P3, PF.
    label : str # e.g., "Primer periodo", "Segundo periodo", ...
    order : int  # e.g., 1, 2, 3, 4 (used for sorting)
    student_links : list[StudentSubjectLink] = Relationship(back_populates='term')
    
class StudentCreate(StudentBase):
    pass

class StudentUpdate(StudentBase):
    pass

class Student(StudentBase, table=True):
    __tablename__ = "student"
    id : int | None = Field(default=None, primary_key=True)
    subjects : list['Subject'] = Relationship(back_populates='students', link_model=StudentSubjectLink)
    
class SubjectBase(SQLModel):
    component : str
    
class SubjectUpdate(StudentBase):
    pass

class Subject(SubjectBase, table=True):
    __tablename__ = "subject"
    id : int | None = Field(default=None, primary_key=True)
    students : list['Student'] = Relationship(back_populates='subjects', link_model=StudentSubjectLink)