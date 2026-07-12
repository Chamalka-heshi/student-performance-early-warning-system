## Model Evaluation & Selection (Phase 7)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.62 | 0.47 |
| Decision Tree | 0.58 | 0.43 |
| Random Forest | 0.67 | 0.53 |
| Neural Network | 0.73 | 0.56 |

**Selected model: Neural Network**

Rationale: Best overall accuracy and macro F1, and strongest performance 
specifically on GradeClass 3.0/4.0 (the at-risk classes this system is 
designed to flag). Its weakest class (0.0) represents top-performing 
students, which is not the target population for early intervention.

Limitation: class weighting was applied to LR/Decision Tree/Random Forest 
via class_weight='balanced', but not to the Neural Network's loss function. 
Future work could explore weighted loss to further improve rare-class recall.

## Explainability Findings (Phase 8)
Top features driving high-risk (GradeClass 4.0) predictions, per SHAP analysis:
1. Absences (dominant factor, consistent with EDA correlation of r=0.73)
2. StudyTimeWeekly
3. HighAbsenceFlag (engineered feature - confirms Phase 5 decision was well-founded)
4. ParentalSupport
Remaining features (Tutoring, Sports, Extracurricular, Gender, Ethnicity, etc.) 
contribute comparatively little to risk predictions.