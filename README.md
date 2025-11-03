# Spam Email Detection using Machine Learning
🧠 **Overview**

This project detects whether an SMS or email is Spam or Ham (Not Spam) using various Machine Learning algorithms.
It leverages the SMSSpamCollection dataset, performs text preprocessing with NLTK, and uses TF-IDF Vectorization for feature extraction.

⚙️ **Tech Stack**

| Tool                       | Purpose                       |
| -------------------------- | ----------------------------- |
| 🐍 **Python**              | Programming language          |
| 📊 **Pandas, NumPy**       | Data handling                 |
| 🧹 **NLTK**                | Text preprocessing            |
| 🤖 **Scikit-learn**        | Machine learning & evaluation |
| 🎨 **Matplotlib, Seaborn** | Data visualization            |



🔄 **Workflow**

1️⃣ Data Cleaning → Convert to lowercase, remove punctuation & stopwords

2️⃣ Feature Extraction → Convert text into numerical vectors using TF-IDF

3️⃣ Model Training → Train models like Naive Bayes, Logistic Regression, SVM, and Random Forest

4️⃣ Evaluation → Compare models using Accuracy, Precision, Recall, and F1-Score

5️⃣ Result Analysis → Visualize confusion matrices and compare performance

📊 **Model Performance**
| Model                   | Accuracy | Remarks                            |
| -------------------     | -------- | ---------------------------------- |
| **Naive Bayes**         | 0.977    | Excellent baseline                 |
| **Logistic Regression** | 0.982    | High precision                     |
| **SVM**                 | 0.986    | Best performance                   |
| **Random Forest**       | 0.975    | Stable but slower                  |
| **Decision Tree**       | 0.940    | Overfits slightly                  |
| **KNN**                 | 0.910    | Lower accuracy, not ideal for text |


🧾 **Evaluation Metrics**

-Accuracy = (TP + TN) / (TP + TN + FP + FN)

-Precision = TP / (TP + FP)

-Recall = TP / (TP + FN)

-F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

🌟 **Future Improvements**

-Integrate Deep Learning models like LSTM or BERT

-Develop a Streamlit or Flask Web App for real-time detection

-Deploy as an API for email platforms
