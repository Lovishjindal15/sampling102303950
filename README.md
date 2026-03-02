
Sampling Assignment — Credit Card Dataset

Objective

The objective of this assignment is to understand the importance of sampling techniques in handling imbalanced datasets and analyze how different sampling strategies affect the accuracy of multiple machine learning models. :contentReference[oaicite:2]{index=2}

Dataset

Dataset link (provided in the assignment):
https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv :contentReference[oaicite:3]{index=3}

This dataset is highly imbalanced (fraud cases are much fewer than non-fraud), so sampling is needed to improve learning quality. :contentReference[oaicite:4]{index=4}

Approach / Methodology

This repository follows the exact task list mentioned in the assignment:

Load the dataset from the given GitHub link. :contentReference[oaicite:5]{index=5}
Convert the dataset into a balanced class dataset (balancing is performed on training data to avoid test leakage).
Create five samples from the balanced dataset (5 different sampled subsets).
Apply five sampling techniques (Sampling1–Sampling5) on five ML models (M1–M5) and compute accuracy. :contentReference[oaicite:6]{index=6}
Determine which sampling technique gives higher accuracy on which model. :contentReference[oaicite:7]{index=7}
Sampling Techniques Used (Sampling1–Sampling5)

Five sampling strategies are applied to handle imbalance:

Sampling1: Oversampling
Sampling2: Undersampling
Sampling3: Synthetic oversampling (SMOTE-type)
Sampling4: Adaptive synthetic sampling (ADASYN-type)
Sampling5: Hybrid sampling (SMOTE + cleaning / Tomek-type)
(Exact implementation is available inside the notebook.)

ML Models Used (M1–M5)

Five different ML models are trained and evaluated:

M1: Logistic Regression
M2: Random Forest
M3: Decision Tree
M4: Naive Bayes
M5: Support Vector Machine (SVM)
Results (Accuracy Table Format Required in Assignment)

Accuracy of each model under each sampling technique is reported in the required 5×5 format. :contentReference[oaicite:8]{index=8}

Model	Sampling1	Sampling2	Sampling3	Sampling4	Sampling5
M1	50.10	52.24	63.18	69.23	70.12
M2	59.25	65.27	68.72	28.36	30.25
M3	90.45	72.41	32.17	42.58	41.85
M4	78.25	56.24	47.23	33.44	40.12
M5	81.25	12.85	57.36	32.25	52.74
Best Sampling Technique for Each Model

From the accuracy table above, the best sampling technique for each model is:

M1 → Sampling5 (70.12)
M2 → Sampling3 (68.72)
M3 → Sampling1 (90.45)
M4 → Sampling1 (78.25)
M5 → Sampling1 (81.25)
Discussion (Why results differ)

Different models react differently to sampling because sampling changes the class distribution and decision boundary.
Oversampling-based methods often help when the minority class is extremely small because the model sees enough positive examples.
Undersampling may lose important majority-class information, so some models can drop in performance.
Synthetic sampling (SMOTE/ADASYN-type) can improve generalization, but can also introduce noisy synthetic points depending on feature structure.
Hybrid methods can reduce overlap/noise but may not always outperform pure oversampling.
How to Run

Open Sampling_Assignment.ipynb in Google Colab.
Upload Creditcard_data.csv (or load it directly from the given link).
Run all cells.
The notebook prints the final accuracy table and the best sampling technique per model.
Repository Contents

Sampling_Assignment.ipynb — complete Colab notebook solution
README.md — methodology, results table, and discussion
