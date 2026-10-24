"""
Main Execution Script for Hotel Booking Cancellation Analysis.
Course: Exploratory Data Analysis and Visualization (U21ADP05)
Assignment II
Institution: KPR Institute of Engineering and Technology

This script executes the complete data science and deep learning workflow:
1. Load and validate the dataset.
2. Perform comprehensive data inspection and missing value analysis.
3. Clean and preprocess data with a leak-free ColumnTransformer pipeline.
4. Stratified splitting (70% Train, 15% Validation, 15% Test).
5. Generate high-resolution publication-grade EDA visualizations.
6. Train and tune the Deep Learning MLP model with early stopping.
7. Evaluate the model on the held-out test set and plot evaluation curves.
8. Compute permutation-based feature importance.
9. Automatically compile the formal Academic Project Report (.docx / .pdf).
"""

import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loading import load_and_inspect_data
from src.preprocessing import split_and_preprocess
from src.visualization import generate_all_eda_visualizations
from src.model_training import train_model
from src.evaluation import evaluate_model_pipeline
from src.report_generator import generate_academic_report


def main():
    start_time = time.time()
    print("\n" + "#" * 70)
    print(" HOTEL BOOKING CANCELLATION ANALYSIS - ASSIGNMENT II PIPELINE")
    print(" KPR Institute of Engineering and Technology | Dept of AI & DS")
    print("#" * 70 + "\n")

    # Step 1 & 2: Load and Inspect Dataset
    print("[STEP 1/6] Loading and Validating Dataset...")
    df, summary = load_and_inspect_data()

    # Step 3: Exploratory Data Analysis and Visualizations
    print("\n[STEP 2/6] Generating Exploratory Data Analysis (EDA) Visualizations...")
    eda_metadata = generate_all_eda_visualizations(df)

    # Step 4: Preprocessing and Stratified Splitting
    print("\n[STEP 3/6] Data Preprocessing and Leak-Free Stratified Splitting...")
    data_dict = split_and_preprocess(
        df,
        target_col="is_canceled",
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        drop_duplicates=True,
    )

    # Step 5: Deep Learning Model Training
    print("\n[STEP 4/6] Deep Learning MLP Model Training & Validation...")
    model, history, train_summary = train_model(
        X_train=data_dict["X_train"],
        y_train=data_dict["y_train"],
        X_val=data_dict["X_val"],
        y_val=data_dict["y_val"],
        batch_size=256,
        max_epochs=30,
        learning_rate=0.001,
        use_class_weights=True,
    )

    # Step 6: Model Evaluation & Feature Importance
    print("\n[STEP 5/6] Model Evaluation and Permutation Feature Importance...")
    eval_results = evaluate_model_pipeline(
        model=model,
        history=history,
        X_test=data_dict["X_test"],
        y_test=data_dict["y_test"],
        feature_names=data_dict["feature_names"],
    )

    # Step 7: Academic Report Generation
    print("\n[STEP 6/6] Compiling Academic Project Report (.docx / .pdf)...")
    docx_path, pdf_path = generate_academic_report(
        student_name="Pooja Sri",
        roll_number="711521104000",
        submission_date="October 2026",
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("         WORKFLOW EXECUTION COMPLETED SUCCESSFULLY!")
    print(f" Total Execution Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(" Generated Artifacts:")
    print("   - Processed Model: outputs/hotel_cancellation_model.keras")
    print("   - Preprocessor:    outputs/preprocessor.joblib")
    print("   - Test Metrics:    outputs/model_metrics.csv")
    print("   - Feature Imp.:    outputs/feature_importance.csv")
    print("   - Word Report:     reports/hotel_booking_cancellation_report.docx")
    if pdf_path and os.path.exists(pdf_path):
        print(f"   - PDF Report:      {pdf_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
