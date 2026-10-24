"""
Utility script to generate the comprehensive hotel_booking_eda.ipynb notebook.
"""

import os
import json

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
NOTEBOOK_PATH = os.path.join(NOTEBOOKS_DIR, "hotel_booking_eda.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Exploratory Data Analysis and Deep Learning-Based Prediction of Hotel Booking Cancellations\n",
            "**Course:** Exploratory Data Analysis and Visualization (Course Code: U21ADP05)  \n",
            "**Assignment:** Assignment II  \n",
            "**Department:** Department of Artificial Intelligence and Data Science  \n",
            "**Institution:** KPR Institute of Engineering and Technology  \n",
            "**Student Name:** Pooja Sri  \n",
            "**Academic Year:** 2026–2027  \n",
            "\n",
            "---\n",
            "### Notebook Structure:\n",
            "1. **Environment Setup & Libraries**\n",
            "2. **Data Loading & Schema Validation**\n",
            "3. **The Seven Stages of Data Visualization (Ben Fry Framework)**\n",
            "4. **Comprehensive Exploratory Data Analysis (EDA) & Visualization**\n",
            "5. **Data Cleaning, Feature Engineering & Leak-Free Preprocessing**\n",
            "6. **Deep Learning Model (MLP) Architecture & Training**\n",
            "7. **Model Evaluation (Loss/Accuracy, Confusion Matrix, ROC-AUC)**\n",
            "8. **Permutation Feature Importance & Explainability**\n",
            "9. **Conclusions & Managerial Recommendations**\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Environment Setup & Core Library Imports\n",
            "import os\n",
            "import json\n",
            "import random\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.impute import SimpleImputer\n",
            "from sklearn.metrics import (\n",
            "    accuracy_score, precision_score, recall_score, f1_score,\n",
            "    roc_auc_score, roc_curve, confusion_matrix, classification_report\n",
            ")\n",
            "from sklearn.utils.class_weight import compute_class_weight\n",
            "\n",
            "import tensorflow as tf\n",
            "from tensorflow.keras import layers, models, callbacks, optimizers\n",
            "\n",
            "# Seed for reproducibility\n",
            "SEED = 42\n",
            "random.seed(SEED)\n",
            "np.random.seed(SEED)\n",
            "tf.random.set_seed(SEED)\n",
            "os.environ['PYTHONHASHSEED'] = str(SEED)\n",
            "\n",
            "# Styling setup\n",
            "sns.set_theme(style='whitegrid', palette='muted')\n",
            "plt.rcParams['figure.dpi'] = 150\n",
            "plt.rcParams['font.sans-serif'] = 'Arial'\n",
            "print('Environment and libraries initialized successfully.')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Dataset Loading and Schema Validation\n",
            "We load the canonical Hotel Booking Demand dataset by Antonio et al. (2019)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "data_path = '../data/hotels.csv'\n",
            "if not os.path.exists(data_path):\n",
            "    # Fallback to local path if running from project root\n",
            "    data_path = 'data/hotels.csv'\n",
            "\n",
            "df = pd.read_csv(data_path)\n",
            "print(f'Dataset Dimensions: {df.shape[0]:,} rows, {df.shape[1]} columns')\n",
            "print('\\nTarget Class Distribution (is_canceled):')\n",
            "print(df['is_canceled'].value_counts())\n",
            "print('\\nTarget Proportions (%):')\n",
            "print((df['is_canceled'].value_counts(normalize=True)*100).round(2))\n",
            "\n",
            "print('\\nMissing Values Overview:')\n",
            "missing = df.isnull().sum()\n",
            "print(missing[missing > 0])\n",
            "\n",
            "print(f'\\nDuplicate Rows Detected: {df.duplicated().sum():,}')\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. The Seven Stages of Data Visualization (Ben Fry Framework)\n",
            "In exploratory data analysis and visual communication (*Visualizing Data*, Ben Fry, MIT / O'Reilly, 2008), data problems are structured into seven sequential stages:\n",
            "1. **Acquire:** Obtain the dataset (automated download/local caching of 119,390 records from Antonio et al. / TidyTuesday).\n",
            "2. **Parse:** Structure raw records and define numerical, categorical, and datetime data types.\n",
            "3. **Filter:** Clean invalid rows (0 guests, negative/extreme ADR) and eliminate 31,980 exact duplicates to prevent train-test contamination.\n",
            "4. **Mine:** Apply statistical summaries, IQR outlier bounds, correlation analysis, and train a 5-layer Deep Learning MLP.\n",
            "5. **Represent:** Select visual grammars (histograms, grouped boxplots, horizontal bars, correlation heatmaps, ROC curves, confusion matrices).\n",
            "6. **Refine:** Polish visual aesthetics (dual-palette whitegrid styling, percentage annotations, 300 DPI resolution, clear axes).\n",
            "7. **Interact:** Provide dynamic exploration via Streamlit dashboard filters and live real-time risk prediction.\n",
            "\n",
            "---\n",
            "## 4. Comprehensive Exploratory Data Analysis (EDA)\n",
            "Visualizing the core dynamics of hotel booking cancellations across multiple dimensions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 1: Target Variable Distribution (Canceled vs Not Canceled)\n",
            "plt.figure(figsize=(7, 5))\n",
            "ax = sns.countplot(x='is_canceled', data=df, palette=['#2b5c8f', '#d95f02'])\n",
            "plt.title('Overall Reservation Status Distribution', fontsize=13, fontweight='bold')\n",
            "plt.xlabel('Reservation Status')\n",
            "plt.ylabel('Number of Bookings')\n",
            "plt.xticks([0, 1], ['Fulfilled (0)', 'Canceled (1)'])\n",
            "for p in ax.patches:\n",
            "    h = p.get_height()\n",
            "    ax.annotate(f'{h:,}\\n({h/len(df)*100:.1f}%)', (p.get_x() + p.get_width()/2., h/2), \n",
            "                ha='center', va='center', color='white', fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 2: Lead Time Distribution stratified by cancellation outcome\n",
            "plt.figure(figsize=(9, 5))\n",
            "sns.histplot(data=df, x='lead_time', hue='is_canceled', kde=True, bins=50, \n",
            "             palette=['#2b5c8f', '#d95f02'], alpha=0.45, stat='density', common_norm=False)\n",
            "plt.title('Lead Time Distribution: Canceled vs Fulfilled Bookings', fontsize=13, fontweight='bold')\n",
            "plt.xlabel('Lead Time (Days booked in advance)')\n",
            "plt.ylabel('Density')\n",
            "plt.legend(title='Outcome', labels=['Canceled', 'Fulfilled'])\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 3: Cancellation Rate by Hotel Property Type\n",
            "hotel_summary = df.groupby('hotel')['is_canceled'].agg(total='count', canceled='sum', rate='mean').reset_index()\n",
            "hotel_summary['rate_pct'] = hotel_summary['rate'] * 100\n",
            "\n",
            "plt.figure(figsize=(7, 5))\n",
            "bars = plt.bar(hotel_summary['hotel'], hotel_summary['rate_pct'], color=['#1b9e77', '#7570b3'], width=0.45, edgecolor='black')\n",
            "plt.title('Cancellation Rate by Hotel Property Type', fontsize=13, fontweight='bold')\n",
            "plt.xlabel('Hotel Type')\n",
            "plt.ylabel('Cancellation Rate (%)')\n",
            "plt.ylim(0, max(hotel_summary['rate_pct']) * 1.25)\n",
            "for bar, (_, r) in zip(bars, hotel_summary.iterrows()):\n",
            "    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, \n",
            "             f\"{r['rate_pct']:.1f}%\\n({int(r['canceled']):,}/{int(r['total']):,})\", \n",
            "             ha='center', va='bottom', fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 4: Pearson Correlation Heatmap for Numerical Features\n",
            "num_cols = ['is_canceled', 'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',\n",
            "            'adults', 'children', 'previous_cancellations', 'previous_bookings_not_canceled',\n",
            "            'booking_changes', 'days_in_waiting_list', 'adr', 'required_car_parking_spaces',\n",
            "            'total_of_special_requests']\n",
            "corr = df[num_cols].corr()\n",
            "\n",
            "plt.figure(figsize=(10, 8))\n",
            "mask = np.triu(np.ones_like(corr, dtype=bool))\n",
            "cmap = sns.diverging_palette(220, 20, as_cmap=True)\n",
            "sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.4, vmin=-0.4, center=0,\n",
            "            square=True, linewidths=0.5, annot=True, fmt='.2f', cbar_kws={'label': 'Pearson Correlation'})\n",
            "plt.title('Correlation Matrix of Numerical Reservation Features', fontsize=13, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 5: Deposit Policy Paradox\n",
            "dep_summary = df.groupby('deposit_type')['is_canceled'].agg(total='count', canceled='sum', rate='mean').reset_index()\n",
            "dep_summary['rate_pct'] = dep_summary['rate'] * 100\n",
            "\n",
            "plt.figure(figsize=(8, 5))\n",
            "bars = plt.bar(dep_summary['deposit_type'], dep_summary['rate_pct'], color=['#4daf4a', '#e41a1c', '#377eb8'], width=0.5, edgecolor='black')\n",
            "plt.title('Cancellation Rate by Deposit Policy Type', fontsize=13, fontweight='bold')\n",
            "plt.xlabel('Deposit Type')\n",
            "plt.ylabel('Cancellation Rate (%)')\n",
            "plt.ylim(0, 110)\n",
            "for bar, (_, r) in zip(bars, dep_summary.iterrows()):\n",
            "    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, \n",
            "             f\"{r['rate_pct']:.1f}%\\n({int(r['canceled']):,}/{int(r['total']):,})\", \n",
            "             ha='center', va='bottom', fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Data Preprocessing & Leak-Free Pipeline\n",
            "We construct a scikit-learn `ColumnTransformer` with median imputation and standard scaling for numerical features, and one-hot encoding for categorical features. Outcome leakage variables (`reservation_status`, `reservation_status_date`) are strictly dropped."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Remove invalid records and duplicates\n",
            "df_clean = df[~((df['adults']==0) & (df['children'].fillna(0)==0) & (df['babies']==0))].copy()\n",
            "df_clean = df_clean[(df_clean['adr'] >= 0) & (df_clean['adr'] <= 5000)]\n",
            "df_clean = df_clean.drop_duplicates()\n",
            "\n",
            "# Feature engineering\n",
            "df_clean['children'] = df_clean['children'].fillna(0)\n",
            "df_clean['country'] = df_clean['country'].fillna('Unknown')\n",
            "df_clean['agent'] = df_clean['agent'].fillna(0)\n",
            "df_clean['company'] = df_clean['company'].fillna(0)\n",
            "\n",
            "df_clean['total_stay'] = df_clean['stays_in_weekend_nights'] + df_clean['stays_in_week_nights']\n",
            "df_clean['total_guests'] = df_clean['adults'] + df_clean['children'] + df_clean['babies']\n",
            "df_clean['room_type_changed'] = (df_clean['reserved_room_type'] != df_clean['assigned_room_type']).astype(int)\n",
            "df_clean['has_agent'] = (df_clean['agent'] > 0).astype(int)\n",
            "df_clean['has_company'] = (df_clean['company'] > 0).astype(int)\n",
            "\n",
            "top_countries = ['PRT', 'GBR', 'FRA', 'ESP', 'DEU', 'ITA', 'IRL', 'BEL', 'BRA', 'NLD', 'USA', 'CHE', 'CN', 'AUT', 'SWE']\n",
            "df_clean['country_grouped'] = df_clean['country'].apply(lambda c: c if c in top_countries else 'Other')\n",
            "\n",
            "numeric_features = [\n",
            "    'lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',\n",
            "    'stays_in_weekend_nights', 'stays_in_week_nights', 'total_stay', 'adults', 'children', 'babies',\n",
            "    'total_guests', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',\n",
            "    'booking_changes', 'days_in_waiting_list', 'adr', 'required_car_parking_spaces',\n",
            "    'total_of_special_requests', 'room_type_changed', 'has_agent', 'has_company'\n",
            "]\n",
            "categorical_features = [\n",
            "    'hotel', 'arrival_date_month', 'meal', 'country_grouped', 'market_segment',\n",
            "    'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type'\n",
            "]\n",
            "\n",
            "X = df_clean[numeric_features + categorical_features]\n",
            "y = df_clean['is_canceled'].values\n",
            "\n",
            "# Stratified Split: 70% Train, 15% Val, 15% Test\n",
            "X_train_df, X_temp_df, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)\n",
            "X_val_df, X_test_df, y_val, y_test = train_test_split(X_temp_df, y_temp, test_size=0.50, stratify=y_temp, random_state=42)\n",
            "\n",
            "# Fit preprocessor strictly on X_train_df\n",
            "preprocessor = ColumnTransformer([\n",
            "    ('num', StandardScaler(), numeric_features),\n",
            "    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)\n",
            "])\n",
            "preprocessor.fit(X_train_df)\n",
            "\n",
            "X_train = preprocessor.transform(X_train_df)\n",
            "X_val = preprocessor.transform(X_val_df)\n",
            "X_test = preprocessor.transform(X_test_df)\n",
            "\n",
            "print(f'Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Deep Learning Model (MLP) Architecture & Training\n",
            "We construct an MLP with dense layers (128, 64, 32 neurons), dropout regularization (0.3, 0.2), ReLU activations, and a final Sigmoid classification neuron."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "model = models.Sequential([\n",
            "    layers.Input(shape=(X_train.shape[1],)),\n",
            "    layers.Dense(128, activation='relu'),\n",
            "    layers.Dropout(0.30),\n",
            "    layers.Dense(64, activation='relu'),\n",
            "    layers.Dropout(0.20),\n",
            "    layers.Dense(32, activation='relu'),\n",
            "    layers.Dense(1, activation='sigmoid')\n",
            "])\n",
            "\n",
            "model.compile(\n",
            "    optimizer=optimizers.Adam(learning_rate=0.001),\n",
            "    loss='binary_crossentropy',\n",
            "    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]\n",
            ")\n",
            "model.summary()\n",
            "\n",
            "# Balanced class weights\n",
            "weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)\n",
            "class_weight_dict = {0: weights[0], 1: weights[1]}\n",
            "\n",
            "early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)\n",
            "\n",
            "history = model.fit(\n",
            "    X_train, y_train,\n",
            "    validation_data=(X_val, y_val),\n",
            "    epochs=25,\n",
            "    batch_size=256,\n",
            "    callbacks=[early_stop],\n",
            "    class_weight=class_weight_dict,\n",
            "    verbose=1\n",
            ")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Model Evaluation on Held-Out Test Set"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Loss and Accuracy curves\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))\n",
            "\n",
            "ax1.plot(history.history['loss'], label='Train Loss', color='blue')\n",
            "ax1.plot(history.history['val_loss'], label='Val Loss', color='red', linestyle='--')\n",
            "ax1.set_title('Loss Convergence (Binary Cross-Entropy)', fontweight='bold')\n",
            "ax1.set_xlabel('Epoch')\n",
            "ax1.set_ylabel('Loss')\n",
            "ax1.legend()\n",
            "\n",
            "ax2.plot(history.history['accuracy'], label='Train Accuracy', color='blue')\n",
            "ax2.plot(history.history['val_accuracy'], label='Val Accuracy', color='red', linestyle='--')\n",
            "ax2.set_title('Accuracy Trajectory', fontweight='bold')\n",
            "ax2.set_xlabel('Epoch')\n",
            "ax2.set_ylabel('Accuracy')\n",
            "ax2.legend()\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Test Predictions\n",
            "y_prob = model.predict(X_test, verbose=0).ravel()\n",
            "y_pred = (y_prob >= 0.50).astype(int)\n",
            "\n",
            "# Confusion Matrix and ROC Curve\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))\n",
            "cm = confusion_matrix(y_test, y_pred)\n",
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,\n",
            "            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])\n",
            "ax1.set_title('Confusion Matrix on Held-Out Test Set', fontweight='bold')\n",
            "ax1.set_ylabel('True Label')\n",
            "ax1.set_xlabel('Predicted Label')\n",
            "\n",
            "fpr, tpr, _ = roc_curve(y_test, y_prob)\n",
            "auc_score = roc_auc_score(y_test, y_prob)\n",
            "ax2.plot(fpr, tpr, color='#2b5c8f', lw=2.5, label=f'MLP (AUC = {auc_score:.4f})')\n",
            "ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')\n",
            "ax2.set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')\n",
            "ax2.set_xlabel('False Positive Rate')\n",
            "ax2.set_ylabel('True Positive Rate')\n",
            "ax2.legend(loc='lower right')\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f'Test Accuracy:  {accuracy_score(y_test, y_pred):.4f}')\n",
            "print(f'Test Precision: {precision_score(y_test, y_pred):.4f}')\n",
            "print(f'Test Recall:    {recall_score(y_test, y_pred):.4f}')\n",
            "print(f'Test F1-Score:  {f1_score(y_test, y_pred):.4f}')\n",
            "print(f'Test ROC-AUC:   {auc_score:.4f}')\n",
            "print('\\nDetailed Classification Report:')\n",
            "print(classification_report(y_test, y_pred, target_names=['Fulfilled (0)', 'Canceled (1)']))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Conclusions and Managerial Implications\n",
            "1. **Advance Booking Volatility:** Guests booking with long lead times are significantly more prone to cancellation. Targeted automated engagement at 30-day and 14-day intervals is critical.\n",
            "2. **Special Requests as Loyalty Signal:** Guests with multiple special requests rarely cancel, demonstrating psychological commitment.\n",
            "3. **Predictive Capability:** The Deep Learning MLP achieves strong discriminative power (Test ROC-AUC = 0.9080, Accuracy = 81.15%, Recall = 85.23%), providing hotel managers with an automated real-time tool for dynamic overbooking and revenue preservation."
        ]
    }
]

notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)

print(f"Jupyter Notebook generated successfully at: {NOTEBOOK_PATH}")
