# Exploratory Data Analysis and Deep Learning-Based Prediction of Hotel Booking Cancellations

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.17](https://img.shields.io/badge/TensorFlow-2.17-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit 1.66](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎓 Academic Project Information
- **Course:** Exploratory Data Analysis and Visualization (Course Code: `U21ADP05`)
- **Assignment:** Assignment II
- **Department:** Department of Artificial Intelligence and Data Science
- **Institution:** KPR Institute of Engineering and Technology, Coimbatore, Tamil Nadu
- **Student Name:** Pooja Sri S.
- **Roll Number:** `711521104000`
- **Academic Year:** 2026–2027

---

## 📌 1. Project Overview
Reservation cancellations introduce severe revenue volatility, inventory perishability, and scheduling friction across the hospitality sector. Hotel operators routinely encounter cancellation rates exceeding 30–40%, impairing yield management, staffing allocation, and room rate optimization.

This repository delivers an end-to-end Data Science and Deep Learning solution that:
1. Conducts rigorous exploratory and statistical data analysis on **119,390 real-world hotel reservations**.
2. Implements a leak-free data preprocessing pipeline with IQR outlier auditing, mode and categorical imputation, duplicate resolution (31,980 exact duplicates removed), and stratified data partitioning (70% Train, 15% Validation, 15% Test).
3. Synthesizes the workflow using the canonical **Seven Stages of Data Visualization** (Ben Fry paradigm).
4. Trains and evaluates a **Deep Learning Multilayer Perceptron (MLP)** binary classifier in TensorFlow/Keras with early stopping and class weighting.
5. Interprets feature influences via **permutation feature importance** on held-out test data.
6. Deploys an interactive **Streamlit Intelligence Dashboard** (`app.py`) for real-time cancellation risk scoring.
7. Compiles a formal **Academic Project Report** in editable Microsoft Word (`.docx`) and Adobe PDF formats.

---

## ❓ 2. Problem Statement
In digital hotel booking platforms, how can hoteliers reliably predict whether a confirmed reservation will be canceled prior to the guest arrival date, **without relying on post-event outcome indicators (`reservation_status`, `reservation_status_date`) that introduce fatal target data leakage?**

Accurately solving this problem enables:
- Proactive engagement and re-confirmation workflows for high-risk guests.
- Dynamic, risk-calibrated overbooking quotas rather than static heuristics.
- Minimizing lost revenue from unsold vacant rooms while avoiding guest walk costs.

---

## 🎯 3. Project Objectives
- **Data Understanding & Quality Audit:** Ingest 119,390 raw transactions, evaluate schema types, identify missing values, and audit duplicate records.
- **Leak-Free Preprocessing:** Eliminate outcome indicator leakage, treat 31,980 duplicate rows, handle outliers using 1.5×IQR boundaries, and fit transformations strictly on training data.
- **Comprehensive Statistical EDA:** Produce 9 publication-grade figures spanning univariate spreads, bivariate contrasts, multivariate correlations, and seasonality trends.
- **Visualization Framework Integration:** Map the data pipeline to Ben Fry's Seven Stages (*Acquire, Parse, Filter, Mine, Represent, Refine, Interact*).
- **Deep Neural Network Modeling:** Design, regularize, and train a 5-layer Keras MLP classifier with early stopping and balanced class weighting.
- **Asymmetric Operational Evaluation:** Measure Accuracy, Precision, Recall, Specificity, F1-Score, and ROC-AUC on 13,085 isolated test records, evaluating the operational trade-offs of false positives versus false negatives.
- **Explainability & Attribution:** Compute permutation feature importances on test data to quantify predictive signals while clarifying that correlation does not establish causation.
- **Interactive Decision Support:** Provide an intuitive Streamlit dashboard allowing front-desk managers to assess booking cancellation risks in real time.

---

## 📊 4. Dataset Source & Attribution
- **Dataset Title:** Hotel Booking Demand Datasets
- **Authors:** Nuno Antonio, Ana de Almeida, and Luis Nunes (2019)
- **Canonical Citation:** Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. **Data in Brief**, 22, 41–49. [https://doi.org/10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126)
- **Public CSV Source:** [TidyTuesday Hotel Booking Dataset](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv)
- **Coverage:** July 1, 2015 to August 31, 2017 (26 months).
- **Properties:**
  - **City Hotel (H2):** 79,330 reservations in Lisbon, Portugal.
  - **Resort Hotel (H1):** 40,060 reservations in Algarve, Portugal.
- **Target Variable:** `is_canceled` (0 = Fulfilled / Check-Out, 1 = Canceled / No-Show).

---

## 🛠️ 5. Technologies & Stack
- **Core Language:** Python 3.11
- **Data Ingestion & Manipulation:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-Learn, Joblib
- **Data Visualization:** Matplotlib, Seaborn
- **Deep Learning Framework:** TensorFlow 2.17 / Keras
- **Interactive Web Dashboard:** Streamlit 1.66
- **Academic Report Generation:** python-docx, docx2pdf
- **Interactive Cloud Analysis:** Jupyter Notebook, Google Colab

---

## 📁 6. Repository Structure
```text
hotel-booking-cancellation-analysis/
│
├── data/                               # Dataset repository
│   ├── README.md                       # Provenance, citation & download guide
│   └── hotels.csv                      # Canonical dataset (auto-downloaded on first run)
│
├── notebooks/                          # Interactive Jupyter & Colab Notebooks
│   ├── Hotel_Booking_Cancellation_Analysis_Assignment.ipynb # Submission-ready Google Colab notebook
│   └── hotel_booking_eda.ipynb         # Interactive EDA and modeling development notebook
│
├── outputs/                            # Generated artifacts, figures & models
│   ├── eda_01_cancellation_count.png
│   ├── eda_02_lead_time_distribution.png
│   ├── eda_03_cancellation_by_hotel_type.png
│   ├── eda_04_correlation_heatmap.png
│   ├── eda_05_adr_distribution_boxplot.png
│   ├── eda_06_cancellation_by_market_segment.png
│   ├── eda_07_cancellation_by_deposit_type.png
│   ├── eda_08_monthly_booking_patterns.png
│   ├── eda_09_special_requests_impact.png
│   ├── eval_01_training_validation_loss.png
│   ├── eval_02_training_validation_accuracy.png
│   ├── eval_03_confusion_matrix.png
│   ├── eval_04_roc_curve.png
│   ├── eval_05_feature_importance.png
│   ├── hotel_cancellation_model.keras  # Saved Deep Learning model weights
│   ├── preprocessor.joblib              # Fitted ColumnTransformer pipeline
│   ├── model_metrics.csv               # Verified test evaluation metrics
│   ├── feature_importance.csv          # Ranked permutation feature importances
│   ├── feature_names.json              # 96 transformed feature names
│   ├── preprocessing_summary.json      # Preprocessing metadata & split sizes
│   └── training_history.json           # 30-epoch training & validation history
│
├── reports/                            # Generated formal academic reports
│   ├── hotel_booking_cancellation_report.docx # Formal Word report
│   └── hotel_booking_cancellation_report.pdf  # Publication-ready PDF report
│
├── scripts/                            # Automation & helper scripts
│   ├── build_colab_notebook.py         # Colab notebook builder
│   └── create_notebook.py              # Notebook generation utility
│
├── src/                                # Core modular Python package
│   ├── data_loading.py                 # Automated download, schema check, inspection
│   ├── preprocessing.py                # IQR outlier filter, leakage prevention, split
│   ├── visualization.py                # 9 publication-grade statistical charts
│   ├── model_training.py               # Keras MLP training with early stopping
│   ├── evaluation.py                   # Test metrics, confusion matrix, ROC-AUC, PFI
│   └── report_generator.py             # Academic Word/PDF report compiler
│
├── app.py                              # Streamlit intelligence dashboard & predictor
├── main.py                             # Master automated orchestration script
├── requirements.txt                    # Pinned project dependencies
├── .gitignore                          # Git ignore configuration
└── README.md                           # Documentation & execution guide
```

---

## ⚙️ 7. Data Preprocessing Methodology
1. **Target Data Leakage Prevention:** The attributes `reservation_status` and `reservation_status_date` directly disclose the final booking outcome (`Check-Out`, `Canceled`, `No-Show`). Retaining these attributes would lead to artificial 100% test accuracy that fails in production. Both columns are **strictly excluded** from input features.
2. **Duplicate Record Resolution:** The raw dataset contains **31,980 exact duplicate rows** (26.8%). Retaining identical rows across training and testing partitions causes severe data leakage (the model memorizes training records identical to test instances). Exact duplicates were dropped, yielding **87,228 unique records**.
3. **Invalid Record Filtering:** Removed 180 zero-guest bookings (`adults + children + babies == 0`) and 2 extreme/negative ADR records.
4. **IQR Outlier Auditing:** Continuous features were audited using the 1.5×IQR boundary rule. Genuine behavioral extremes (e.g., advance lead times up to 737 days) were retained to reflect realistic distribution tails.
5. **Missing Value Imputation:**
   - `children` (4 missing): Imputed with mode (`0`).
   - `country` (488 missing): Imputed as `'Unknown'`, then grouped into top 15 origin countries and `'Other'`.
   - `agent` (16,340 missing) & `company` (112,593 missing): Imputed with `0`, accompanied by engineered binary flags `has_agent` and `has_company`.
6. **Domain Feature Engineering:** `total_stay`, `total_guests`, `room_type_changed`, and `country_grouped`.
7. **Leak-Free Stratified Partitioning:**
   - **Training Set (70%):** 61,059 samples (27.52% baseline cancellation rate)
   - **Validation Set (15%):** 13,084 samples (27.52% baseline cancellation rate)
   - **Testing Set (15%):** 13,085 samples (27.53% baseline cancellation rate)
   - `ColumnTransformer` (StandardScaler for 22 numerics + OneHotEncoder for 10 categoricals) was **strictly fitted only on training data**.

---

## 📈 8. EDA Visualizations & Key Findings
All 9 publication-grade figures are stored in `outputs/`:

| Figure | Chart Type | Key Empirical Finding & Actionable Insight |
| :--- | :--- | :--- |
| **Figure 1** (`eda_01`) | Countplot | 72.48% fulfilled vs 27.52% canceled in unique records; moderate ~2.6:1 imbalance. |
| **Figure 2** (`eda_02`) | Histogram + KDE | Canceled bookings show median lead time of **113 days** vs **45 days** for fulfilled bookings. |
| **Figure 3** (`eda_03`) | Grouped Bar | City Hotels experience higher cancellations (**30.0%**) than Resort Hotels (**23.5%**). |
| **Figure 4** (`eda_04`) | Correlation Heatmap | `lead_time` exhibits strongest positive correlation ($r = +0.18$); special requests negative ($r = -0.12$). |
| **Figure 5** (`eda_05`) | Boxplot + Density | Canceled reservations consistently exhibit higher median Average Daily Rates (€94.50 median). |
| **Figure 6** (`eda_06`) | Horizontal Bar | Online Travel Agencies drive the highest churn volume (**32.8%**); Direct bookings are most stable (**15.3%**). |
| **Figure 7** (`eda_07`) | Bar Chart | Non-refundable bookings paradoxically exhibit **94.8%** cancellation due to high-risk group targeting. |
| **Figure 8** (`eda_08`) | Dual-Axis Seasonality | Arrival volume peaks in July–August; cancellation rate mirrors summer rate spikes. |
| **Figure 9** (`eda_09`) | Bar Chart | Guests submitting 0 requests experience **33.9%** cancellation; $\ge 2$ requests drop below **15%**. |

---

## 🧠 9. Deep Learning Model Architecture
A genuine Deep Learning Multilayer Perceptron (MLP) was constructed in TensorFlow/Keras to model non-linear interactions across 96 transformed input features:

```
Input Layer (96 Features)
       │
       ▼
Dense Layer (128 Neurons, ReLU Activation)
       │
       ▼
Dropout Regularization (Rate = 0.30)
       │
       ▼
Dense Layer (64 Neurons, ReLU Activation)
       │
       ▼
Dropout Regularization (Rate = 0.20)
       │
       ▼
Dense Layer (32 Neurons, ReLU Activation)
       │
       ▼
Output Layer (1 Neuron, Sigmoid Activation)
```

- **Total Trainable Parameters:** 22,785 (91.14 KB)
- **Optimizer:** Adam ($\alpha = 0.001$)
- **Loss Function:** Binary Cross-Entropy
- **Batch Size:** 256
- **Class Weights:** Balanced weighting ($w_0 = 0.69$, $w_1 = 1.82$)
- **Convergence:** 30 epochs executed. Best validation loss of **0.3886** achieved at **Epoch 28**, with best weights restored via `EarlyStopping`.

---

## 📊 10. Verified Evaluation Metrics & Feature Importance

### Held-Out Test Evaluation (13,085 Test Samples)
All metrics independently verified against `outputs/model_metrics.csv` and actual test predictions:

| Metric | Score | Percentage | Operational Interpretation |
| :--- | :---: | :---: | :--- |
| **Accuracy** | `0.8115` | **81.15%** | Overall correct cancellation and fulfillment predictions. |
| **Precision** | `0.6134` | **61.34%** | 3,070 true cancellations flagged out of 5,005 predicted cancellations. |
| **Recall (Sensitivity)** | `0.8523` | **85.23%** | 3,070 true cancellations detected out of 3,602 total test cancellations. |
| **Specificity** | `0.7960` | **79.60%** | 7,548 fulfilled stays correctly identified out of 9,483 fulfilled bookings. |
| **F1-Score** | `0.7134` | **71.34%** | Harmonic mean balancing precision and recall under class imbalance. |
| **ROC-AUC Score** | `0.9080` | **90.80%** | Outstanding discriminative capability across all decision thresholds. |

### Test Set Confusion Matrix Breakdown
- **True Negatives (TN):** 7,548 (79.60% of 9,483 fulfilled bookings)
- **False Positives (FP):** 1,935 (20.40% of 9,483 fulfilled bookings)
- **False Negatives (FN):** 532 (14.77% of 3,602 actual cancellations)
- **True Positives (TP):** 3,070 (85.23% of 3,602 actual cancellations)

### Top Permutation Features (ROC-AUC Degradation)
1. `country_grouped_PRT` ($\Delta\text{AUC} = 0.0950$)
2. `adr` (Average Daily Rate) ($\Delta\text{AUC} = 0.0748$)
3. `required_car_parking_spaces` ($\Delta\text{AUC} = 0.0668$)
4. `lead_time` ($\Delta\text{AUC} = 0.0546$)
5. `arrival_date_year` ($\Delta\text{AUC} = 0.0489$)
6. `room_type_changed` ($\Delta\text{AUC} = 0.0389$)
7. `total_of_special_requests` ($\Delta\text{AUC} = 0.0387$)

*Scientific Clarification: Ranked features represent predictive statistical signals and proxies, not proof of direct causation.*

---

## 💻 11. Installation Instructions (Windows PowerShell)

### 1. Clone the Repository
```powershell
git clone https://github.com/<your-username>/hotel-booking-cancellation-analysis.git
cd hotel-booking-cancellation-analysis
```

### 2. Create and Activate a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 12. Execution: Automated Pipeline
To execute the complete end-to-end data pipeline from scratch (automated data check, 9 EDA figures, leak-free preprocessing, MLP model training, test evaluation, and Word/PDF report generation):
```powershell
python main.py
```

---

## 🌐 13. Execution: Interactive Streamlit Dashboard
To launch the interactive intelligence dashboard:
```powershell
streamlit run app.py
```
The application opens automatically in your browser at `http://localhost:8501`. Features include:
- Interactive dataset explorer with dynamic property type and market segment filters.
- Complete gallery of all 9 high-resolution EDA figures.
- Deep Learning performance curves (Loss, Accuracy, Confusion Matrix, ROC-AUC).
- Ranked permutation feature importance table and chart.
- **Real-Time Cancellation Risk Predictor:** Live reservation form returning calibrated cancellation probabilities and operational advisories.

---

## 📓 14. Interactive Google Colab Notebook
The complete submission notebook is located at:
```
notebooks/Hotel_Booking_Cancellation_Analysis_Assignment.ipynb
```
### How to Run in Google Colab:
1. Open [colab.research.google.com](https://colab.research.google.com/).
2. Select **File $\to$ Upload notebook**, and select `Hotel_Booking_Cancellation_Analysis_Assignment.ipynb`.
3. Click **Runtime $\to$ Run all** (`Ctrl + F9`). The notebook automatically fetches the dataset from the public repository and executes all 17 sections self-contained in the cloud.
4. Click **File $\to$ Save a copy in Drive** and use **Share $\to$ Anyone with the link** for submission.

---

## 📑 15. Academic Project Reports
The formal academic report is compiled and available in both formats:
- **Adobe PDF:** [`reports/hotel_booking_cancellation_report.pdf`](file:///c:/Users/Pooja%20Sri/OneDrive/Documents/Desktop/hotel-booking-cancellation-analysis/reports/hotel_booking_cancellation_report.pdf) (1.43 MB)
- **Microsoft Word:** [`reports/hotel_booking_cancellation_report.docx`](file:///c:/Users/Pooja%20Sri/OneDrive/Documents/Desktop/hotel-booking-cancellation-analysis/reports/hotel_booking_cancellation_report.docx) (2.11 MB)

Contains all 14 academic sections, including Title Page, Abstract, Objectives, Dataset Schema, Leak-Free Preprocessing, 9 EDA Figures with 3-part structured explanations, Seven Stages of Visualization, Deep Learning Architecture, Evaluation Metrics Table, Precision vs. Recall Trade-Off, Feature Importance, Limitations, 6 genuine APA citations, and Appendix.

---

## ⚠️ 16. Limitations and Future Improvements
- **Geographic Concentration:** Data represents two Portuguese hotel properties (Lisbon and Algarve); seasonal patterns may vary in Asian or North American metropolitan centers.
- **Absence of Real-Time Covariates:** External factors such as flight delays, macroeconomic currency shifts, and extreme weather are absent from static booking transactions.
- **Future Improvements:**
  - Recurrent Neural Networks (LSTMs or Temporal Transformers) to model booking modification sequences.
  - Automated dynamic pricing integration coupling cancellation probabilities to real-time room tariffs.
  - Transfer learning across international hotel groups with domain adaptation.

---

## 📚 17. References & Dataset Citation
1. Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126
2. Antonio, N., de Almeida, A., & Nunes, L. (2017). Predicting hotel booking cancellations with a machine learning approach. *Journal of Hospitality and Tourism Technology*, 8(2), 266–281. https://doi.org/10.1108/JHTT-08-2016-0043
3. Sánchez-Medina, A. J., & C-Sánchez, E. (2020). Using machine learning and big data for efficient forecasting of hotel booking cancellations. *International Journal of Hospitality Management*, 89, 102546. https://doi.org/10.1016/j.ijhm.2020.102546
4. Fry, B. (2008). *Visualizing Data: Exploring and Explaining Data with the Processing Environment*. O'Reilly Media. ISBN: 978-0-596-51455-6.
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. http://www.deeplearningbook.org
6. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---

## ⚖️ License
This project is licensed under the MIT License. Academic and educational use is encouraged with appropriate attribution.
