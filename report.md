# Influenza Risk Prediction Using Machine Learning


# 1. Problem Definition

## 1.1 Problem Statement

Seasonal influenza continues to affect millions of people each year, leading to hospital visits, missed workdays, and serious complications for vulnerable individuals. Identifying people who are likely to have the flu as early as possible can help ensure timely care and reduce the spread of infection.

This project explores a simple but important question:

Can machine learning be used to estimate a person’s likelihood of having the flu based on their reported symptoms and basic risk factors?


## 1.2 Context

In both clinical and community settings, influenza is often identified initially based on symptoms before laboratory tests are performed. However, many respiratory illnesses share similar symptoms, making it difficult to distinguish flu from other conditions at an early stage.

A data-driven predictive model can support early screening by analyzing symptom patterns and identifying individuals who may be at higher risk.

This project simulates a clinical decision-support tool that uses structured symptom and demographic data to estimate the likelihood of influenza infection.



## 1.3 Why This Problem Is Worth Solving

Identifying flu risk early can make a meaningful difference in both clinical and public health settings. Early risk assessment can:

* Support faster and more informed triage decisions
* Reduce unnecessary hospital visits and congestion
* Enable earlier medical intervention
* Help limit the spread of infection
* Strengthen public health response during peak flu seasons

By leveraging machine learning, healthcare providers can enhance traditional clinical assessment with data-driven pattern recognition, supporting more timely and efficient decision-making.



## 1.4 Who Benefits

**Healthcare Providers**:
Enhanced decision support during patient intake, enabling quicker and more informed triage decisions.

**Public Health Authorities**:
Improved early detection trends and more effective resource planning during flu seasons.

**Patients**:
Faster risk identification, supporting timely medical attention, and appropriate care.



## 1.5 What a Useful Outcome Looks Like

A successful outcome for this project includes:

* A predictive model that can accurately distinguish influenza-positive from non-influenza cases.
* High recall performance to minimize missed flu cases, which is especially important in healthcare settings.
* A deployable application that allows users to input symptoms and receive a real-time flu risk assessment.

In this project, the final tuned Gradient Boosting model achieved very high recall along with strong overall discrimination performance, demonstrating that this approach is both practical and effective for symptom-based flu risk prediction.



# 2. Approach and Tool Selection

## 2.1 Overall Approach

To address the problem of influenza risk prediction, a structured supervised machine learning workflow was followed. Since the objective was to determine whether an individual is likely to have influenza based on reported symptoms and risk factors, the task was framed as a binary classification problem (Flu vs. Non-Flu).

The project followed these key stages:

* Data familiarization and validation checks
* Exploratory Data Analysis (EDA)
* Data cleaning and preprocessing
* Feature engineering
* Model selection and benchmarking
* Hyperparameter tuning
* Model evaluation
* Deployment through a web application

This structured workflow ensured that model development was systematic, reproducible, and aligned with machine learning best practices.



## 2.2 Data Preparation and Feature Engineering

The dataset included symptom indicators and demographic variables. Since machine learning models require numerical input, categorical variables were encoded appropriately.

To enhance predictive performance, additional engineered features were created, including:

* Total symptom count
* Grouped symptom categories (e.g., respiratory, gastrointestinal, allergy-related)

Feature engineering allowed the model to capture higher-level symptom patterns that may not be immediately visible from individual symptom indicators alone.



## 2.3 Model Selection

Several classification algorithms were evaluated to determine the most suitable approach:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Gaussian Naive Bayes

Logistic Regression was used as a baseline model due to its interpretability and strong performance in binary classification problems.

Tree-based ensemble methods, particularly Gradient Boosting, were included because they perform well on structured tabular data and can capture non-linear interactions among symptoms.



## 2.4 Evaluation Strategy

Models were evaluated using multiple performance metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Given the healthcare context, recall was prioritized. Missing a flu-positive case (false negative) could lead to delayed treatment and increased transmission. Therefore, model selection emphasized maximizing recall while maintaining reasonable precision.



## 2.5 Hyperparameter Tuning

Hyperparameter tuning was conducted using **RandomizedSearchCV** to improve model performance. This method was chosen over GridSearchCV because it allows efficient exploration of the hyperparameter space while reducing computational cost.

The final tuned Gradient Boosting model achieved the highest recall along with strong ROC-AUC performance, leading to its selection as the final deployment model.



## 2.6 Deployment Tools

## 2.6 Deployment Tools

The project utilized the following tools:

* Python for development
* Pandas and NumPy for data manipulation
* Scikit-learn for machine learning
* Joblib for model serialization
* Streamlit for web application development
* OpenAI API for generative AI explanations
* GitHub for version control
* Streamlit Community Cloud for hosting

Streamlit was selected because it enables rapid deployment of interactive machine learning applications with minimal front-end development. The OpenAI API was integrated to generate natural-language explanations of the model's predictions, improving the interpretability of the system for non-technical users.



## 2.7 Alternative Approaches Considered

Several alternative approaches were considered:

* Deep learning models were not selected because the dataset is structured and tabular, where tree-based ensemble models typically perform better.
* XGBoost was considered but not implemented, as Gradient Boosting already achieved strong performance.
* Accuracy was not used as the primary evaluation metric due to the healthcare context, where recall is more critical for minimizing missed positive cases.

## 2.8 Generative AI Integration

To improve interpretability and user interaction, a generative AI component was integrated into the deployed application. While the machine learning model predicts the probability of influenza infection based on symptom inputs, generative AI is used to translate the prediction results into clear, human-readable explanations.

After the model calculates the flu risk probability, the application sends the prediction result and the reported symptoms to a generative AI model through the OpenAI API. The generative AI model then produces a contextual explanation that includes:

* An interpretation of the predicted flu risk score.

* A summary of the symptoms reported by the user.

* General health guidance and suggested next steps.

For example, if the model predicts a high probability of influenza based on symptoms such as fever, fatigue, and muscle aches, the generative AI system provides a natural-language explanation describing why these symptoms are commonly associated with flu-like illnesses and suggests general precautions such as rest, hydration, or consulting a healthcare professional.

This integration transforms the application from a simple prediction tool into an interactive decision-support system. By combining machine learning predictions with generative AI explanations, the system improves usability and accessibility for non-technical users while maintaining clear disclaimers that the tool does not provide medical diagnosis.



# 3. Reflection

## 3.1 What Worked Better Than Expected

One aspect that worked particularly well was the structured model comparison process. By evaluating multiple algorithms before selecting a final model, I was able to make a data-driven decision rather than relying on assumptions.

Gradient Boosting performed better than expected, especially after hyperparameter tuning, achieving very high recall and strong ROC-AUC performance.

The feature engineering process also contributed meaningfully to performance improvements. Creating aggregated symptom features enabled the model to capture broader clinical patterns beyond individual symptom indicators.

Additionally, deploying the model using Streamlit demonstrated that the solution could function outside of a notebook environment. This significantly strengthened the practical value of the project and reinforced the importance of thinking beyond experimentation toward real-world usability.

Another aspect that enhanced the project was the integration of generative AI into the deployed application. By incorporating the OpenAI API, the system was able to transform numerical prediction outputs into natural language explanations. This significantly improved the usability of the application by helping users understand the meaning of the predicted flu risk and the potential implications of their reported symptoms.



## 3.2 Challenges and Limitations

### Model Selection Complexity

Comparing multiple models required careful interpretation of evaluation metrics beyond accuracy. Determining when to prioritize recall over precision required deeper reflection on the healthcare context and the consequences of false negatives.

### Hyperparameter Tuning Time

Initial grid search attempts were computationally expensive. Transitioning to RandomizedSearchCV improved efficiency while maintaining strong performance improvements.

### Deployment Issues

During deployment, I encountered issues related to file paths, model serialization, and case sensitivity differences between local development (Windows) and cloud hosting environments (Linux). Troubleshooting these challenges reinforced the importance of environment awareness and production readiness.

### Model Limitations

The model relies solely on structured symptom data and does not incorporate laboratory results, detailed medical history, or real-time epidemiological trends. As such, it should be viewed as a decision-support tool rather than a diagnostic system.



## 3.3 What I Would Improve Next Time

If extending this project, I would:

* Implement probability threshold optimization to better balance recall and precision.
* Explore advanced ensemble methods such as XGBoost for additional performance comparison.
* Incorporate explainability techniques (e.g., SHAP values) to improve transparency and interpretability.
* Refactor the training pipeline into a fully modular structure to enhance reproducibility.
* Investigate cost-sensitive learning to explicitly penalize false negatives.



## 3.4 Key Lessons Learned

This project reinforced several important lessons about applying machine learning to real-world problems:

**Problem Context Drives Metric Selection**
Choosing recall over accuracy was not just a technical decision but a contextual one grounded in healthcare impact.

**Model Comparison Is Essential**
Evaluating multiple algorithms provided clarity and prevented premature conclusions.

**Hyperparameter Tuning Must Be Strategic**
Efficient search strategies like RandomizedSearchCV balance performance gains with computational cost.

**Deployment Is a Distinct Skill**
Developing a high-performing model is only part of the process. Deployment introduced additional considerations such as file management, version control, and environment compatibility.

**Documentation and Structure Matter**
Clear organization and reproducibility are as important as strong model performance.

Overall, this project strengthened my understanding of end-to-end machine learning workflows, from problem framing to deployment, and emphasized the importance of aligning technical decisions with real-world healthcare impact.

**Combining Predictive Models with Generative AI Enhances Interpretability**
Integrating generative AI into the application demonstrated how predictive machine learning models can be complemented with language models to produce user-friendly explanations. This hybrid approach improves accessibility and helps bridge the gap between technical model outputs and real-world decision-making.








