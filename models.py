from sqlmodel import SQLModel, Field, Relationship

class StudentBase(SQLModel):
    name : str | None = None
    age : int
    social_data : int | None = None
    overall_grade : str
    fundamental_comp : str
    behavioral_comp : str
    disabilities : str | None
    
class StudentSubjectLink(SQLModel, table=True):
    __tablename__ = "student_to_subject"
    student_id : int = Field(foreign_key="student.id", primary_key=True)
    subject_id : int = Field(foreign_key="subject.id", primary_key=True)
    grade : str
    
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

class Subject(SubjectBase, table=True):
    __tablename__ = "subject"
    id : int | None = Field(default=None, primary_key=True)
    students : list['Student'] = Relationship(back_populates='subjects', link_model=StudentSubjectLink)