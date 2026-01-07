# Predicting-Hotel-Reservation-Cancellations
Develop robust machine learning models to accurately predict hotel reservation cancellations. By utilizing these models, the company aims to improve revenue management, optimize resource allocation, and potentially implement targeted retention strategies.

## **Dataset Overview**
<img width="838" height="377" alt="image" src="https://github.com/user-attachments/assets/d4d7a797-8282-4e2a-ba1d-ff3f204dedcc" />

## **Data Preprocessing**
* **Checked for missing values:  No missing data was found.**
* **Detecting and removing Outliers.**
* **Converted categorical features to numerical using encoding (Label encoder).**
* **Standardized Numerical features.**
* **Split data into training (80%) and testing (20%) sets for model building.**

## **Exploratory Data Analysis**
<img width="512" height="400" alt="image" src="https://github.com/user-attachments/assets/940eb09d-6fd2-4560-8683-7c0aa1ce6730" />

## **Machine Learning Models**
All models were evaluated using accuracy, precision, recall, and F1-score to ensure robust performance comparison.

### **Models Used**
* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier**

### **Logistic Regression**
* **Interpreted linear relationships between features.**

### **Decision Tree Classifier**
* **Selected optimal parameters via grid search.**
* **Grid Search in Decision Tree: Accuracy - 85%.**
* **Max depth:10, minimum samples split: 10.**

### **Random Forest Classifier**
* **Explored ensemble learning for improved accuracy (Accuracy - 67%).**
* **Max depth:20, minimum samples split: 2.**
* **Grid search in random forest: Accuracy - 88%.**

## **Model Performance**
<img width="863" height="292" alt="image" src="https://github.com/user-attachments/assets/57b35ef4-b011-4173-aa5a-ffe39c6bf2da" />

## **Key Findings**
### **Most Influential Factors**
* **Lead Time (highest influence)**
* **Average Price per Room**
* **Number of Adults**
* **Arrival Month & Date (seasonality)**
* **Special Requests**
* **Repeated Guest Status**
### **Model Performance:**
The Random Forest model achieved 0.88%, providing strong accuracy in predicting Hotel reservation cancellations.

## **Recommendations**
Recommendations to reduce cancellations
* **Implement dynamic pricing based on lead time.**
* **Offer early-booking incentives to secure long-lead customers.**
* **Provide flexible or partially refundable booking options.**
* **Encourage customers to add special requests (food, room preference, etc.)**
* **Build loyalty programs to convert new guests into repeat customers.**
* **Automate follow-up reminders for risky bookings.**

## **Conclusion**
* **Lead time is the most influential factor in predicting cancellations longer lead times strongly increase cancellation probability.**
* **Room price and customer type, including repeated guests and special requests, significantly impact booking stability.**
* **Understanding seasonal trends helps hotels optimize pricing and resource allocation.**
* **Machine learning models (Random Forest & Decision Tree) provide high accuracy in identifying high-risk cancellations.**
* **Predictive insights can reduce operational uncertainty and improve revenue management.**

