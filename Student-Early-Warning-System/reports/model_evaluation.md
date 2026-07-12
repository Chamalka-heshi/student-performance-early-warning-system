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