"""
Interactive Streamlit Dashboard for Hotel Booking Cancellation Analysis.
Enterprise Decision-Support Application for Hospitality Revenue Management.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F4E78;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1F4E78;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .status-badge-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
    .status-badge-low {
        background-color: #d4edda;
        color: #155724;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "hotels.csv")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "hotel_cancellation_model.keras")
PREPROCESSOR_PATH = os.path.join(OUTPUTS_DIR, "preprocessor.joblib")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "model_metrics.csv")
FEAT_IMP_PATH = os.path.join(OUTPUTS_DIR, "feature_importance.csv")

TOP_COUNTRIES = ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA", "NLD", "USA", "CHE", "CN", "AUT", "SWE"]


@st.cache_data
def load_cached_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    return None


@st.cache_resource
def load_cached_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    return None, None


df_raw = load_cached_data()
model, preprocessor = load_cached_model()

# Header
st.markdown("<div class='main-header'>🏨 Hotel Booking Cancellation Intelligence Dashboard</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select Section",
    [
        "📊 Project Overview & EDA",
        "🧠 Deep Learning Model Evaluation",
        "🎯 Feature Importance & Drivers",
        "🔮 Live Cancellation Predictor",
    ],
)

# -------------------------------------------------------------
# TAB 1: OVERVIEW & EDA
# -------------------------------------------------------------
if app_mode == "📊 Project Overview & EDA":
    st.subheader("1. Project Summary & Dataset Overview")
    st.markdown(
        "This project analyzes the benchmark **Hotel Booking Demand** dataset (Antonio et al., 2019) containing real-world transactions "
        "from a City Hotel (Lisbon) and a Resort Hotel (Algarve). The goal is to discover patterns correlating with reservation cancellations "
        "and deploy a Deep Learning MLP to forecast future cancellations."
    )

    if df_raw is None:
        st.error(
            f"Dataset not found at `{DATA_PATH}`. Please run `python main.py` or download `hotels.csv` into the `data/` folder."
        )
    else:
        # Top Metrics
        total_rows = len(df_raw)
        total_cols = len(df_raw.columns)
        canc_count = int(df_raw["is_canceled"].sum())
        canc_rate = (canc_count / total_rows) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Bookings", f"{total_rows:,}")
        col2.metric("Total Variables", f"{total_cols}")
        col3.metric("Canceled Bookings", f"{canc_count:,}")
        col4.metric("Baseline Cancellation Rate", f"{canc_rate:.2f}%")

        st.markdown("---")
        st.subheader("Interactive Dataset Explorer")

        # Filters in Sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("Dataset Filters")
        hotel_list = ["All"] + sorted(list(df_raw["hotel"].unique()))
        selected_hotel = st.sidebar.selectbox("Filter by Hotel Type", hotel_list)

        market_list = ["All"] + sorted(list(df_raw["market_segment"].dropna().unique()))
        selected_market = st.sidebar.selectbox("Filter by Market Segment", market_list)

        df_filtered = df_raw.copy()
        if selected_hotel != "All":
            df_filtered = df_filtered[df_filtered["hotel"] == selected_hotel]
        if selected_market != "All":
            df_filtered = df_filtered[df_filtered["market_segment"] == selected_market]

        if selected_hotel != "All" or selected_market != "All":
            f_rows = len(df_filtered)
            f_canc = int(df_filtered["is_canceled"].sum())
            f_rate = (f_canc / f_rows * 100) if f_rows > 0 else 0
            st.info(f"**Filtered Subset Active:** {f_rows:,} bookings | **Cancellations:** {f_canc:,} ({f_rate:.2f}%)")
        else:
            st.caption(f"Showing all {total_rows:,} records (No filter applied).")

        with st.expander("🔍 View Raw Data Sample (First 10 Rows)", expanded=False):
            st.dataframe(df_filtered.head(10), width="stretch")

        with st.expander("⚠️ Missing Values & Data Quality Summary", expanded=False):
            missing_series = df_raw.isnull().sum()
            missing_df = missing_series[missing_series > 0].reset_index()
            missing_df.columns = ["Column", "Missing Count"]
            missing_df["Missing Percentage"] = (missing_df["Missing Count"] / len(df_raw) * 100).round(2).astype(str) + "%"
            st.table(missing_df)

        st.markdown("---")
        st.subheader("Exploratory Data Analysis Visualizations")

        eda_tabs = st.tabs([
            "1. Cancellation Count",
            "2. Lead Time Spread",
            "3. Hotel Comparison",
            "4. Correlation Heatmap",
            "5. Pricing (ADR)",
            "6. Market Segment",
            "7. Deposit Paradox",
            "8. Seasonality Trends",
            "9. Special Requests",
        ])

        chart_map = [
            ("eda_01_cancellation_count.png", "Overall Cancellation Breakdown"),
            ("eda_02_lead_time_distribution.png", "Lead Time Distribution (Histogram + KDE)"),
            ("eda_03_cancellation_by_hotel_type.png", "Cancellation Rate by Hotel Property"),
            ("eda_04_correlation_heatmap.png", "Pearson Correlation Heatmap"),
            ("eda_05_adr_distribution_boxplot.png", "Average Daily Rate (ADR) Spread"),
            ("eda_06_cancellation_by_market_segment.png", "Cancellation Rate by Market Segment"),
            ("eda_07_cancellation_by_deposit_type.png", "Deposit Policy Paradox"),
            ("eda_08_monthly_booking_patterns.png", "Monthly Booking Volume & Cancellation Rate"),
            ("eda_09_special_requests_impact.png", "Special Requests vs Cancellation Rate"),
        ]

        for tab, (fig_name, title) in zip(eda_tabs, chart_map):
            with tab:
                st.markdown(f"#### {title}")
                img_path = os.path.join(OUTPUTS_DIR, fig_name)
                if os.path.exists(img_path):
                    st.image(img_path, width="stretch")
                else:
                    st.warning(f"Chart `{fig_name}` not found. Run `python main.py` to generate all high-resolution figures.")

# -------------------------------------------------------------
# TAB 2: MODEL EVALUATION
# -------------------------------------------------------------
elif app_mode == "🧠 Deep Learning Model Evaluation":
    st.subheader("2. Deep Learning Multilayer Perceptron (MLP) Evaluation")
    st.markdown(
        "A Deep Neural Network with dense layers, ReLU activations, dropout regularization, and an Adam optimizer "
        "was trained on 70% of the data and evaluated on an isolated held-out 15% test partition (13,085 bookings)."
    )

    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)
        m_cols = st.columns(len(metrics_df))
        for col, (_, row) in zip(m_cols, metrics_df.iterrows()):
            col.metric(row["Metric"], row["Percentage"])
    else:
        st.info("Model metrics file not yet generated. Please run `python main.py`.")

    st.markdown("---")
    st.markdown("### Performance & Convergence Curves")

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.markdown("#### Loss Convergence (Binary Cross-Entropy)")
        f_loss = os.path.join(OUTPUTS_DIR, "eval_01_training_validation_loss.png")
        if os.path.exists(f_loss):
            st.image(f_loss, width="stretch")
        else:
            st.warning("Loss curve pending.")

    with row1_c2:
        st.markdown("#### Training vs Validation Accuracy")
        f_acc = os.path.join(OUTPUTS_DIR, "eval_02_training_validation_accuracy.png")
        if os.path.exists(f_acc):
            st.image(f_acc, width="stretch")
        else:
            st.warning("Accuracy curve pending.")

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.markdown("#### Test Set Confusion Matrix")
        f_cm = os.path.join(OUTPUTS_DIR, "eval_03_confusion_matrix.png")
        if os.path.exists(f_cm):
            st.image(f_cm, width="stretch")
        else:
            st.warning("Confusion matrix pending.")

    with row2_c2:
        st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
        f_roc = os.path.join(OUTPUTS_DIR, "eval_04_roc_curve.png")
        if os.path.exists(f_roc):
            st.image(f_roc, width="stretch")
        else:
            st.warning("ROC curve pending.")

# -------------------------------------------------------------
# TAB 3: FEATURE IMPORTANCE
# -------------------------------------------------------------
elif app_mode == "🎯 Feature Importance & Drivers":
    st.subheader("3. Explainability: Permutation Feature Importance")
    st.markdown(
        "Permutation importance quantifies the influence of each feature by measuring the drop in test set ROC-AUC "
        "when its values are randomly shuffled. This maintains model-agnostic interpretability without claiming causation."
    )

    f_imp_chart = os.path.join(OUTPUTS_DIR, "eval_05_feature_importance.png")
    if os.path.exists(f_imp_chart):
        st.image(f_imp_chart, width="stretch")

    if os.path.exists(FEAT_IMP_PATH):
        df_imp = pd.read_csv(FEAT_IMP_PATH)
        with st.expander("📋 View Complete Feature Importance Table", expanded=False):
            st.dataframe(df_imp, width="stretch")

    st.info(
        "**Key Managerial Takeaways (Signals vs Causation):**\n\n"
        "1. **Guest Origin & Domestic Signals:** Domestic bookings (`PRT`) exhibit distinct booking patterns and higher overall churn compared to international fly-in travelers.\n"
        "2. **Average Daily Rate (ADR):** High tariff bookings face greater price sensitivity and shopping around before arrival.\n"
        "3. **Parking Space Demands:** Reserving parking is an empirical proxy for arrival commitment (driving guests rarely cancel).\n"
        "4. **Lead Time:** Far-in-advance bookings have the highest churn risk. Automating milestone confirmation emails at 30 and 15 days out can significantly mitigate sudden cancellations.\n"
        "5. **Special Requests:** Engagement is an empirical signal of commitment. Guests requesting specific amenities rarely cancel."
    )

# -------------------------------------------------------------
# TAB 4: LIVE PREDICTOR FORM
# -------------------------------------------------------------
elif app_mode == "🔮 Live Cancellation Predictor":
    st.subheader("4. Real-Time Booking Cancellation Risk Predictor")
    st.markdown(
        "Enter reservation details below. The input is transformed using the exact fitted preprocessing pipeline "
        "and evaluated by the trained Deep Learning MLP model."
    )

    if model is None or preprocessor is None:
        st.error(
            "Model or Preprocessor pipeline not found! Please run `python main.py` first to train the model and save the pipeline."
        )
    else:
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
                lead_time = st.number_input("Lead Time (Days in Advance)", min_value=0, max_value=750, value=65)
                arrival_date_month = st.selectbox(
                    "Arrival Month",
                    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
                    index=6,
                )
                arrival_date_year = st.selectbox("Arrival Year", [2015, 2016, 2017], index=1)
                arrival_date_week_number = st.slider("Arrival Week Number", 1, 53, 28)
                arrival_date_day_of_month = st.slider("Arrival Day of Month", 1, 31, 15)

            with c2:
                stays_in_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=15, value=1)
                stays_in_week_nights = st.number_input("Week Nights", min_value=0, max_value=30, value=3)
                adults = st.number_input("Adult Guests", min_value=1, max_value=10, value=2)
                children = st.number_input("Children", min_value=0, max_value=10, value=0)
                babies = st.number_input("Babies", min_value=0, max_value=5, value=0)
                meal = st.selectbox("Meal Package", ["BB", "HB", "FB", "SC", "Undefined"])
                country = st.selectbox("Guest Origin Country", TOP_COUNTRIES + ["Other"], index=0)

            with c3:
                market_segment = st.selectbox("Market Segment", ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups", "Complementary"])
                distribution_channel = st.selectbox("Distribution Channel", ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"])
                deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
                customer_type = st.selectbox("Customer Type", ["Transient", "Transient-Party", "Contract", "Group"])
                reserved_room_type = st.selectbox("Reserved Room Type", ["A", "B", "C", "D", "E", "F", "G", "H"])
                assigned_room_type = st.selectbox("Assigned Room Type", ["A", "B", "C", "D", "E", "F", "G", "H"])
                adr = st.number_input("Average Daily Rate (EUR)", min_value=0.0, max_value=1000.0, value=105.0)

            c4, c5, c6 = st.columns(3)
            with c4:
                is_repeated_guest = st.selectbox("Is Repeated Guest?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=20, value=0)
            with c5:
                previous_bookings_not_canceled = st.number_input("Previous Fulfilled Bookings", min_value=0, max_value=50, value=0)
                booking_changes = st.number_input("Number of Booking Changes", min_value=0, max_value=15, value=0)
            with c6:
                required_car_parking_spaces = st.selectbox("Parking Space Required?", [0, 1])
                total_of_special_requests = st.slider("Total Special Requests", 0, 5, 1)

            submit_btn = st.form_submit_button("⚡ Predict Cancellation Risk", width="stretch")

        if submit_btn:
            # Construct dictionary matching exact preprocessing requirements
            total_stay = stays_in_weekend_nights + stays_in_week_nights
            total_guests = adults + children + babies
            room_type_changed = int(reserved_room_type != assigned_room_type)
            has_agent = 1 if distribution_channel in ["TA/TO", "GDS"] or market_segment in ["Online TA", "Offline TA/TO", "Groups"] else 0
            has_company = 1 if market_segment == "Corporate" or distribution_channel == "Corporate" else 0
            country_grouped = country if country in TOP_COUNTRIES else "Other"

            input_dict = {
                "lead_time": [lead_time],
                "arrival_date_year": [arrival_date_year],
                "arrival_date_week_number": [arrival_date_week_number],
                "arrival_date_day_of_month": [arrival_date_day_of_month],
                "stays_in_weekend_nights": [stays_in_weekend_nights],
                "stays_in_week_nights": [stays_in_week_nights],
                "total_stay": [total_stay],
                "adults": [adults],
                "children": [children],
                "babies": [babies],
                "total_guests": [total_guests],
                "is_repeated_guest": [is_repeated_guest],
                "previous_cancellations": [previous_cancellations],
                "previous_bookings_not_canceled": [previous_bookings_not_canceled],
                "booking_changes": [booking_changes],
                "days_in_waiting_list": [0],
                "adr": [adr],
                "required_car_parking_spaces": [required_car_parking_spaces],
                "total_of_special_requests": [total_of_special_requests],
                "room_type_changed": [room_type_changed],
                "has_agent": [has_agent],
                "has_company": [has_company],
                "hotel": [hotel],
                "arrival_date_month": [arrival_date_month],
                "meal": [meal],
                "country_grouped": [country_grouped],
                "market_segment": [market_segment],
                "distribution_channel": [distribution_channel],
                "reserved_room_type": [reserved_room_type],
                "assigned_room_type": [assigned_room_type],
                "deposit_type": [deposit_type],
                "customer_type": [customer_type],
            }

            input_df = pd.DataFrame(input_dict)

            # Apply fitted preprocessor
            transformed_input = preprocessor.transform(input_df)

            # Predict with deep learning model
            prob = float(model.predict(transformed_input, verbose=0)[0][0])
            pred_class = int(prob >= 0.50)

            st.markdown("---")
            st.markdown("### 🎯 Model Prediction Results")

            res_c1, res_c2 = st.columns([1, 2])
            with res_c1:
                if pred_class == 1:
                    st.markdown("<div class='status-badge-high'>🚨 HIGH CANCELLATION RISK</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='status-badge-low'>✅ CONFIRMED / LOW RISK</div>", unsafe_allow_html=True)

                st.metric("Predicted Cancellation Probability", f"{prob*100:.2f}%")
                st.progress(prob)

            with res_c2:
                if pred_class == 1:
                    st.warning(
                        f"**Operational Advisory:** This booking carries a **{prob*100:.1f}% probability of cancellation**.\n\n"
                        "• Send an automated re-confirmation message or itinerary check-in 14 days before arrival.\n"
                        "• Authorize cautious overbooking quota if hotel capacity is constrained.\n"
                        "• Offer value-added dining or spa incentives to secure guest commitment."
                    )
                else:
                    st.success(
                        f"**Operational Advisory:** This booking is assessed as **stable ({prob*100:.1f}% risk)**.\n\n"
                        "• Standard room allocation and front-desk preparation are advised.\n"
                        "• Eligible for early check-in or room upgrade promotions."
                    )
