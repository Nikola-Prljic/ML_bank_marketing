# Maschine learing bank marketing
Machine learning Model with bank marketing data set to find out if client will subscribe.
For iMooX curse.

# Dataset
This dataset provides a practical scenario for applying various data science and machine learning techniques, making it a valuable resource for learning and experimentation.

## Bank marketing
The Bank Marketing dataset is commonly used in machine learning and data science for various analysis and predictive modeling tasks. It includes data related to a marketing campaign run by a Portuguese banking institution. The primary goal of the campaign was to convince potential customers to subscribe to a term deposit. 

### Source

The dataset is publicly available in the UCI Machine Learning Repository. It was donated by S. Moro, P. Cortez, and P. Rita.
Data Composition

The dataset consists of two versions: one with 45,211 records (full dataset) and a smaller version with 4,521 records (a 10% sample). Both datasets have the same structure, with 16 input features (attributes) and one target variable.

## X Features
The dataset includes the following attributes:

    Age: Numeric value representing the age of the client.
    Job: Type of job (categorical).
    Marital: Marital status (categorical: "married", "divorced", "single", etc.).
    Education: Education level (categorical).
    Default: Has credit in default? (binary: "yes", "no").
    Balance: Average yearly balance, in euros (numeric).
    Housing: Has a housing loan? (binary: "yes", "no").
    Loan: Has a personal loan? (binary: "yes", "no").
    Contact: Type of communication contact (categorical: "cellular", "telephone").
    Day: Last contact day of the month (numeric).
    Month: Last contact month of the year (categorical: "jan" to "dec").
    Duration: Last contact duration, in seconds (numeric).
    Campaign: Number of contacts performed during this campaign and for this client (numeric, includes last contact).
    Pdays: Number of days since the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted).
    Previous: Number of contacts performed before this campaign and for this client (numeric).
    Poutcome: Outcome of the previous marketing campaign (categorical: "success", "failure", "other", "unknown").

## Y Label target Variable

    Y: Has the client subscribed to a term deposit? (binary: "yes", "no").
### Key Points

The dataset is imbalanced, with more "no" responses than "yes".
Features like "duration" highly influence the output target (if the call lasts longer, the chance of a successful subscription increases).
The dataset can be used for classification tasks, particularly to predict whether a client will subscribe to a term deposit or not.

### Common Uses

Classification: Building models to predict the likelihood of a customer subscribing to a term deposit.
Exploratory Data Analysis (EDA): Understanding patterns and insights from the marketing data.
Feature Engineering: Creating new features to improve model performance.
Imbalance Handling: Techniques like SMOTE, undersampling, and oversampling can be applied due to the imbalanced nature of the dataset.


![scatter3D](https://github.com/Nikola-Prljic/ML_bank_marketing/assets/72382235/beb64d30-d148-4d2f-b9a5-8303dd3842a0)
![auc_bank_marketing](https://github.com/Nikola-Prljic/ML_bank_marketing/assets/72382235/19100db3-3da3-4b98-9f8b-93a906d2c751)
