Heart Disease Prediction System using Artificial Neural Networks (ANN) 

This project presents a complete end-to-end deep learning pipeline for predicting the likelihood of heart disease using clinical, physiological, and lifestyle parameters. Developed as part of our academic learning in Artificial Neural Networks (ANN) and deployment practices, the system integrates data preprocessing, exploratory data analysis, model building, evaluation, and deployment using Streamlit. The objective is to build a reliable, data-driven healthcare prediction model capable of assisting in early diagnosis and risk assessment.

 1. Introduction

Heart disease remains one of the leading causes of mortality globally. Early detection plays a critical role in preventing complications and improving patient outcomes. With advancements in artificial intelligence, predictive models have become powerful tools to support healthcare professionals by identifying early patterns associated with disease risk.

This project develops a binary classification ANN model capable of predicting whether a person is at risk of heart disease based on various medical and demographic factors. The model is designed using TensorFlow/Keras and deployed through an intuitive Streamlit web interface for real-time predictions.

2. Dataset Description

The dataset consists of multiple clinical features associated with cardiovascular health, including:

Age
Gender
Blood Pressure 
Cholesterol Level
Glucose Level
Heart Rate
BMI (Body Mass Index)
Physical Activity Indicators

The target variable is Heart Disease (0 = No Disease, 1 = Disease).

3. Exploratory Data Analysis (EDA)

The EDA performed in the Jupyter Notebook includes:
Checking for missing values
Evaluating statistical distributions
Identifying outliers using boxplots
Understanding variable correlations using a heatmap
Visualizing relationships using pairplots
Analyzing distribution skewness (left skew, right skew)

These steps helped us understand data quality, shape preprocessing decisions, and ensure model robustness.

4. Data Preprocessing

  The preprocessing pipeline included:

  4.1 Handling Missing Values

  Nulls were detected and appropriately handled.

  4.2 Outlier Detection and Treatment

  Using IQR (Q1, Q3, Min, Max)
  Capping extreme values

  4.3 Feature Scaling

  Standardization using StandardScaler
  Ensures ANN training stability and faster convergence

  4.4 Encoding

  Binary encoding applied wherever required

  4.5 Train-Test Split

  Typical 70:30 split used for unbiased evaluation

5. ANN Model Architecture

The ANN model was built using TensorFlow/Keras with the following structure:
Input Layer: Equal to number of features
Hidden Layers: Multiple Dense layers with ReLU activation
Output Layer: 1 neuron with Sigmoid activation

Optimizer: Adam
Loss Function: Binary Cross-Entropy
Metrics: Accuracy
EarlyStopping callback was applied to prevent overfitting and to retain the best model weights.

6. Model Evaluation

Metrics evaluated include:
Training Accuracy
Validation Accuracy
Loss 

The ANN achieved strong performance, demonstrating reliable predictive capability for real-world usage.

7. Deployment Using Streamlit

The model was deployed on a Streamlit web application, enabling users to:
Enter health parameters through input fields
See prediction results instantly
Experience an intuitive UI design

The Streamlit app loads:
The trained heart_disease_model.h5 model
The serialized scaler .pkl file
This allows seamless real-time inference.

8. Project Structure
├── Heart_disease_prediction.ipynb
├── heart_disease_model.h5
├── scaler.pkl
├── app.py (Streamlit UI)
├── Heart_Disease_and_Hospitals.csv
└── README.md

9. Team Contribution

Team Leader:
Dharmesh Kushwaha

Team Members:
Nitin Verma
Mandeep Kumar
Ronit Maurya
Shivam Maurya
Khyati Singh
Nishtha Agarwal
Abhay Mishra

Contributions included data analysis, preprocessing, ANN model development, UI design, documentation, and deployment.

10. Acknowledgments

We sincerely thank Vedant Sarraf for his mentorship, guidance, and continuous support throughout the project. His direction played a vital role in shaping our understanding of ANN and deployment.
We also extend our appreciation to the leadership and faculty of Invertis University for providing an environment that encourages innovation, practical learning, and teamwork:

11. Conclusion

This project demonstrates the complete lifecycle of a deep learning solution—from raw data to deployment. It enhanced our understanding of deep learning architectures, feature engineering, evaluation metrics, and real-time deployment using Streamlit. The model provides a practical, accessible, and effective approach to supporting early heart disease detection, contributing to the broader goal of integrating AI into healthcare.
