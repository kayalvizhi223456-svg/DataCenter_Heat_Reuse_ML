# Data Center Heat-Reuse Suitability Prediction Using Machine Learning

## 🌐 Live Website

**Deployed Website:**  
[Open the Data Center Heat-Reuse Predictor](YOUR_DEPLOYED_WEBSITE_LINK)

> The live website link will be updated after deployment.

---

## 📌 Project Overview

Data centers consume large amounts of electricity and water, generating significant amounts of waste heat. This waste heat can potentially be reused for applications such as district heating, industrial processes, and other thermal-energy requirements.

This project develops a **machine-learning-based prediction system** that estimates the **heat-reuse suitability category of a data center** based on its operational and geographical characteristics.

The project combines:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Heat-reuse suitability scoring
- Machine learning classification
- Class-imbalance handling
- Model comparison
- Feature importance analysis
- Temporal validation
- Facility-level validation
- A Flask-based prediction website

The final system provides a web interface where users can enter data-center characteristics and receive a predicted heat-reuse suitability category.

---

## 🎯 Problem Statement

Data centers generate substantial amounts of waste heat, but the feasibility of reusing this heat varies depending on facility characteristics and operating conditions.

The objective of this project is to investigate whether machine learning can predict a predefined **heat-reuse suitability category** from available data-center characteristics while avoiding target leakage.

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze data-center operational and environmental characteristics.
2. Perform data cleaning and exploratory data analysis.
3. Develop a heat-reuse suitability scoring framework.
4. Categorize data centers into Low, Medium, and High suitability.
5. Build machine-learning classification models.
6. Compare different machine-learning approaches.
7. Address the severe class imbalance in the High-suitability category.
8. Identify important predictive features.
9. Evaluate the model using random, temporal, and facility-level validation.
10. Develop a web-based prediction interface using Flask.
11. Deploy the prediction system as a publicly accessible website.

---

## 📊 Dataset

The dataset contains **126,770 records** representing data-center observations from **2019 to 2025**.

### Original Dataset Features

| Feature | Description |
|---|---|
| Year | Observation year |
| Facility_ID | Unique facility identifier |
| Facility_Name | Name of the data center |
| Owner_Company | Facility owner |
| City | Facility city |
| Country | Facility country |
| Facility_Type | Type of data center |
| Estimated_Capacity_MW | Estimated facility capacity |
| PUE | Power Usage Effectiveness |
| Cooling_System_Type | Cooling technology |
| WUE_L_per_kWh | Water Usage Effectiveness |
| Daily_Electricity_Usage_MWh | Daily electricity consumption |
| Daily_Water_Usage_Gallons | Daily water consumption |
| Surrounding_Water_Stress_Tier | Water-stress classification |

### Dataset Summary

- Total records: **126,770**
- Unique facilities: **18,110**
- Years: **2019–2025**
- Average records per facility: approximately **7**
- Missing values: **None**
- Duplicate rows: **None**

---

# 🔥 Heat-Reuse Suitability Framework

A predefined heat-reuse score was developed using three components:

1. Cooling-system suitability
2. PUE
3. Estimated facility capacity

### Cooling-System Scores

| Cooling System | Score |
|---|---:|
| Liquid Cooled | 1.0 |
| Evaporative | 0.6 |
| Air Cooled | 0.2 |

### Normalization

PUE and facility capacity were normalized using Min-Max normalization.

The heat-reuse score was calculated as:

```text
Heat Reuse Score =
    0.5 × Cooling Score
  + 0.3 × PUE Normalized
  + 0.2 × Capacity Normalized

### Suitability Categories

| Score Range | Category |
|---|---|
| ≥ 0.66 | High |
| 0.33 – 0.65 | Medium |
| < 0.33 | Low |

### Target Distribution

| Category | Records | Percentage |
|---|---:|---:|
| Low | 52,595 | 41.49% |
| Medium | 73,723 | 58.15% |
| High | 452 | 0.36% |
| **Total** | **126,770** | **100%** |

---

# 🤖 Machine Learning

The machine-learning task is formulated as a multi-class classification problem.

### Target Variable

Heat_Reuse_Category

### Predictor Features

- Year
- Country
- City
- Facility_Type
- WUE_L_per_kWh
- Daily_Electricity_Usage_MWh
- Daily_Water_Usage_Gallons
- Surrounding_Water_Stress_Tier

### Features Excluded to Prevent Target Leakage

The following features were excluded because they were directly used to construct the heat-reuse score:

- Cooling_System_Type
- PUE
- Estimated_Capacity_MW
- Cooling_Score
- PUE_Normalized
- Capacity_Normalized
- Heat_Reuse_Score

Facility identifiers such as Facility_ID, Facility_Name, and Owner_Company were also excluded from the main prediction experiment.

---

# 🧠 Models Evaluated

The following machine-learning approaches were evaluated:

1. Random Forest without class weighting
2. Balanced Random Forest
3. Random Forest with increased High-class weight
4. SMOTE + Random Forest
5. CatBoost

---

# 📈 Model Results

| Model | Accuracy | Macro F1 | High F1 |
|---|---:|---:|---:|
| Random Forest – No Weights | 72.44% | 46.49% | 0.00% |
| Random Forest – Balanced | 72.90% | 53.52% | 10.17% |
| Random Forest – High Weight 10 | 74.70% | 55.06% | 18.18% |
| SMOTE + Random Forest | 73.35% | 53.94% | 10.95% |
| CatBoost | 94.62% | 80.82% | 53.00% |

---

# 🧪 Validation

Three validation strategies were used:

| Validation Method | Accuracy | Macro F1 | High F1 |
|---|---:|---:|---:|
| Random Stratified | 94.62% | 80.82% | 53.00% |
| 2025 Time-Based | 93.51% | 81.03% | 55.61% |
| Unseen Facilities | 89.88% | 67.46% | 22.01% |

---

# 🌐 Web Application

A Flask-based web application was developed for interactive prediction.

Users can enter:

- Year
- Country
- City
- Facility Type
- WUE
- Daily Electricity Usage
- Daily Water Usage
- Surrounding Water Stress Tier

The application returns:

- Predicted suitability category
- Prediction confidence
- Low probability
- Medium probability
- High probability

### Technologies

- Python
- Flask
- CatBoost
- Pandas
- NumPy
- HTML
- CSS
- JavaScript

---

# 📁 Project Structure

DataCenter_Heat_Reuse_ML/
│
├── models/
├── src/
├── website/
│   ├── app.py
│   ├── templates/
│   └── static/
├── results/
├── requirements.txt
├── .gitignore
└── README.md

---

# ⚠️ Limitations

- The heat-reuse category is based on a predefined scoring framework.
- It is not a direct measurement of recoverable waste heat.
- The High category represents only 0.36% of the dataset.
- Performance decreases when predicting completely unseen facilities.
- Feature importance does not establish causality.
- Real-world heat-reuse feasibility requires additional engineering and geographical information.

---

# 🔮 Future Work

- Use real-world heat-recovery measurements.
- Include heat-temperature information.
- Include nearby heat-demand information.
- Add district-heating and industrial heat-demand data.
- Apply SHAP-based explainability.
- Improve rare-class prediction.
- Perform external validation.
- Develop a larger monitoring/dashboard system.

---

# 📌 Conclusion

This project demonstrates a machine-learning approach for predicting data-center heat-reuse suitability from operational, geographical, and environmental characteristics.

Among the evaluated models, CatBoost achieved strong performance across the validation experiments. The project also provides a Flask-based web application for interactive prediction.

---

# 📜 Disclaimer

The prediction represents a category generated using the project's predefined heat-reuse suitability framework. It is not a direct measurement of recoverable waste heat and should not be treated as an engineering feasibility assessment without additional real-world analysis.