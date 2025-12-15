# Maria: AI-Driven Early Warning System 🎓

> **"A preventive academic system that helps reduce course failure and grade repetition, incorporating psychological, age-related, academic, and student predictors."**

---

## 📖 Overview

**Maria** is an intelligent agent designed to tackle a critical issue in secondary and post-secondary education: the disconnect between student struggles and timely intervention.

Current educational models often suffer from high dropout rates and academic failures due to a lack of real-time monitoring. **Maria** solves this by integrating diverse data sources—academic records, psychological profiles, and behavioral logs—to predict academic risk *before* a student fails.

## 🚩 The Problem

Secondary institutions face significant challenges regarding student performance:
* **Weak Comprehensive Formation:** Manifested in low grades and subject failures.
* **Reactive vs. Proactive:** Interventions often come too late, after failure has occurred.
* **Data Silos:** Critical information (behavioral, demographic, academic) is often disconnected, making it hard to see the "full picture" of a student at risk.

## 💡 The Solution

Maria serves as a bridge for the educational community (parents, teachers, administrators), offering:
1.  **Timely Alerts:** Identifying at-risk students based on predictive modeling.
2.  **Holistic Analysis:** Considering not just grades, but age, stratum (socioeconomic context), and disabilities.
3.  **Metric Optimization:** Aiming to improve institutional indicators like **ICFES scores** and retention rates.

---

## 🛠️ Tech Stack

This project is built using modern Python standards, leveraging asynchronous capabilities for high performance.

* **Language:** Python 3.13 🐍
* **Web Framework:** [FastAPI](https://fastapi.tiangolo.com/) (v0.121+)
* **Database ORM:** [SQLModel](https://sqlmodel.tiangolo.com/) (Interaction with SQLite/PostgreSQL)
* **Data Validation:** Pydantic (v2.12+)
* **Server:** Uvicorn (ASGI)
* **Machine Learning (Roadmap):** Scikit-Learn (Integration pending for risk scoring)

## ⚡ Key Features (Current & Planned)

- [x] **Student Profiling:** Detailed capture of demographic and contact info (JSONB support).
- [x] **Academic Tracking:** Recording of grades and behavioral components.
- [x] **Relational Modeling:** Complex mapping of Students ↔ Grades ↔ Risks.
- [ ] **ML Inference Endpoint:** Real-time risk prediction based on student feature vectors.
- [ ] **Alert Dashboard:** Visual indicators for "High" and "Critical" risk students.

---

## 🚀 Getting Started

Follow these steps to set up the development environment on your local machine.

### 1. Prerequisites
* Python 3.10 or higher (Developed on **3.13.10**)
* Conda (optional, but recommended for environment management)

### 2. Installation

Clone the repository and navigate to the project folder:

```bash
git clone [https://github.com/marlon-cooh/salesIAn.git](https://github.com/marlon-cooh/salesIAn.git)
cd salesIAn
```
