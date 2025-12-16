from sqlmodel import SQLModel, Field, Relationship

class StudentBase(SQLModel):
    name : str | None
    age : int
    social_data : int | None
    overall_grade : str
    fundamental_comp : str
    behavioral_comp : str
    disabilities : str | None
    
class StudentSubjectLink(SQLModel, table=True):
    student_id : int | None = Field(default=None, foreign_key="student.id", primary_key=True)
    subject_id : int | None = Field(default=None, foreign_key="subject.id", primary_key=True)
    
class StudentCreate(StudentBase):
    pass

class StudentUpdate(StudentBase):
    pass

class Student(StudentBase, table=True):
    id : int = Field(default=None, primary_key=True)
    subjects : list['Subject'] = Relationship(back_populates='students', link_model=StudentSubjectLink)
    
class Subject(SQLModel, table=True):
    id : int = Field(default=None, primary_key=True)
    students : list['Student'] = Relationship(back_populates='subjects', link_model=StudentSubjectLink)
    component : str
    grade : str