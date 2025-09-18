![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

# 🏥 Healthcare Data Analysis Project

## 📌 Overview
This project demonstrates the role of an **Information Analyst** by exploring hospital patient data to uncover operational insights. It includes data cleaning, exploratory data analysis (EDA), visualization, and reporting of key healthcare performance indicators. The dataset is **synthetic** (randomly generated for this project) and does not represent real patient records.

---

## 📂 Project Structure
Healthcare_Data_Analysis_Project/  
├── data/               # dataset(s)  
│   └── hospital_data.csv  
├── notebooks/          # analysis scripts / notebooks  
│   └── analysis.py  
├── reports/            # reports & figures  
│   ├── figs/           # saved charts  
│   └── Healthcare_Report.docx (optional)  
├── dashboard/          # Power BI / Tableau files  
└── README.md           # this file  

---

## 🔎 Key Questions Explored
- Which departments have the **highest admission volumes**?  
- What is the **average cost per patient** per department?  
- What are the **readmission rates** overall and by department?  
- Is there a correlation between **length of stay** and **cost**?  
- How do **admissions trend over time**?  

---

## 📊 Sample Insights
- Emergency admissions spike in certain months, suggesting a need for **seasonal staffing adjustments**.  
- Cardiology patients have the **highest average treatment cost**, with a higher-than-average readmission rate.  
- Length of stay has a positive correlation with cost (longer stays = higher expenses).  

---

## 🛠 Tools & Libraries
- **Python 3** (pandas, matplotlib)  
- **Power BI / Tableau** (for dashboarding)  
- **Git & GitHub** (for version control & collaboration)  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/ArmandoSNHU/Healthcare_Data_Analysis_Project.git
   cd Healthcare_Data_Analysis_Project
Make sure you have Python 3 installed.

Install dependencies:

bash
Copy code
pip install pandas matplotlib
Run the analysis:

bash
Copy code
python notebooks/analysis.py
View results:

Console summary in terminal.

Charts saved in reports/figs/.

📈 Example Outputs
## 📈 Example Outputs
![Admissions by Department](reports/figs/admissions_by_department.png)
![Average Cost by Department](reports/figs/avg_cost_by_department.png)
![Readmission Rates](reports/figs/readmission_yes_by_department.png)

📌 Notes
Dataset is synthetic and anonymized.

Project is intended for educational / portfolio purposes.

✨ Author
Armando Gomez
Graduate student in Computer Science / AI, with a focus on data analysis, visualization, and applied research.