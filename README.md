# 🧠 Student Depression Prediction (Deep Learning)

## 📌 Overview

This project aims to predict whether a student is likely to experience depression based on lifestyle and academic factors using a **Deep Learning model**.

It is an end-to-end machine learning project covering:

* Data preprocessing
* Model building (Neural Network)
* Model evaluation
* Simple web app deployment

---

## 🎯 Problem Statement

Student mental health is a growing concern.
The goal is to build a predictive system that can classify students as:

* **Depressed (1)**
* **Not Depressed (0)**

based on features like:

* Sleep duration
* Study hours
* Stress level
* Lifestyle habits

---

## 🧠 Approach

### 1. Data Processing

* Removed irrelevant features (e.g., City, Profession, Degree)
* Handled missing values
* Encoded categorical variables
* Scaled numerical features

### 2. Model

* Implemented a **Feedforward Neural Network**
* Used:

  * ReLU activation
  * Sigmoid output layer
  * Binary Cross Entropy loss
  * Adam optimizer

### 3. Evaluation

* Accuracy
* F1 Score
* Confusion Matrix

---

## 📂 Project Structure

```
student-depression-dl/
│
├── data/
├── notebooks/
├── src/
├── app/
├── models/
├── reports/
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/student-depression-dl.git
cd student-depression-dl
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run training

```bash
python src/model/train.py
```

### 5. Run the app

```bash
streamlit run app/app.py
```

---

## 📊 Results

* Baseline ML Model (Logistic Regression): ~82% accuracy
* Deep Learning Model: (to be updated)

---

## 📈 Future Improvements

* Hyperparameter tuning
* Feature engineering improvements
* Try advanced architectures
* Deploy on cloud (AWS / GCP)

---

## 🧑‍💻 Author

Souvik

---

## ⭐ Key Learnings

* Difference between ML and Deep Learning
* Importance of preprocessing
* Building end-to-end AI systems
* Model evaluation and comparison
