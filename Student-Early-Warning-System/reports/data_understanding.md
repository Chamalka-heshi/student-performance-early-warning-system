# Data Understanding — Phase 2

**Dataset:** Students Performance Dataset (Rabie El Kharoua, Kaggle, CC BY 4.0)
**Rows:** [fill in from df.shape]
**Target variable:** GradeClass (categorical grade classification)

## Why this dataset
Matches our project's planned features (attendance/absences, weekly study time, 
GPA, tutoring status) and is large enough and clean enough to build a full 
pipeline without excessive cleaning overhead.

## Important note
This dataset is synthetic (generated for educational/ML purposes), not collected 
from a real institution. This is disclosed transparently in the final report — 
the goal of this project is to demonstrate the ML/software engineering pipeline, 
not to make real-world dropout claims.

## Initial observations
- [fill in: any missing values found?]
- [fill in: is GradeClass balanced or imbalanced?]
- [fill in: any surprising ranges in Absences / StudyTimeWeekly / GPA?]

## Cleaning Summary (Phase 3)
- Dropped StudentID (identifier, no predictive value)
- Missing values: none found across all 15 columns
- Duplicates: [fill in from Step 2]
- Value ranges verified against documentation: Age (15-18), Absences (0-29), 
  StudyTimeWeekly (0-20), GPA (0.0-4.0) — all within expected bounds
- Class imbalance identified in target (GradeClass): class 4.0 (lowest 
  performance) represents 50.6% of students, class 0.0 (highest) only 4.5%.
  This will be addressed during modeling (Phase 6) via class weighting 
  and evaluated using per-class precision/recall/F1, not accuracy alone.
- Final cleaned shape: 2392 rows x 14 columns
- Saved to: data/processed/students_cleaned.csv

## Correlation Findings (Phase 4)
- Absences shows the strongest relationship with academic outcomes 
  (r=-0.92 with GPA, r=0.73 with GradeClass) — by far the dominant signal.
- GPA correlates strongly with GradeClass (r=-0.78), as expected since 
  GradeClass is derived from GPA bands — GPA will be EXCLUDED as a model 
  input to avoid leakage; only behavioral/demographic features will be used.
- StudyTimeWeekly, Tutoring, and ParentalSupport show only weak linear 
  correlations individually (|r| < 0.2) but are retained as features, 
  since tree-based models can capture non-linear interactions that 
  correlation alone doesn't reveal.

  ## Feature Engineering Summary (Phase 5)
- Excluded GPA from features (leakage risk — GradeClass is derived from GPA bands)
- One-hot encoded Ethnicity (nominal category, no inherent order)
- Kept ParentalEducation and ParentalSupport as ordinal numeric (existing 
  encoding reflects meaningful order)
- Engineered HighAbsenceFlag (Absences > 15, roughly the dataset median) 
  as an interpretable risk indicator for later explainability
- Final feature count: [fill in from X.shape]
- Saved to: data/processed/X_features.csv, data/processed/y_target.csv