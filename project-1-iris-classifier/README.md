# 🌸 Iris Flower Classifier - Machine Learning Project 1

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

**A hands-on introduction to supervised machine learning using the classic Iris dataset.** Built as part of my journey to become an AI Expert. This project demonstrates data exploration, model training, evaluation, and hyperparameter tuning with scikit-learn.

---

## 🎯 **Project Objectives**
- Master supervised classification basics
- Practice data preprocessing and visualization
- Compare Logistic Regression vs Decision Tree models
- Implement cross-validation and hyperparameter tuning
- Build a production-ready prediction script

**Key Learnings**: Petal features are most discriminative; Logistic Regression achieves 97%+ accuracy; Decision Trees visualize decision boundaries clearly.

---

## 📊 **Results**
| Model                | Test Accuracy | Cross-Validation Score | Best Params (Tree) |
|----------------------|---------------|-----------------------|-------------------|
| Logistic Regression  | **97.3%**     | **97.3%**             | N/A               |
| Decision Tree        | **96.7%**     | **95.8%**             | `max_depth=3, min_samples_split=2` |

*Detailed confusion matrix and plots in `iris_classifier.ipynb`*

---

## 🛠 **Tech Stack**
| Category         | Tools                          |
|------------------|--------------------------------|
| **Data**         | pandas, numpy                 |
| **ML Models**    | scikit-learn (LogisticRegression, DecisionTreeClassifier) |
| **Visualization**| matplotlib, seaborn           |
| **Evaluation**   | cross_val_score, GridSearchCV |

---

## 🚀 **Quick Start**

1. **Clone the repo**
   ```bash
   git clone https://github.com/tommoni98/ML-Journey.git
   cd ML-Journey/project-1-iris-classifier
   
1. **Set up virtual environment**
   ```bash
   python -m venv env
   # Windows: .\env\Scripts\activate
   # macOS/Linux: source env/bin/activate
   
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
1. **Run the project**
   ```bash
   jupyter notebook iris_classifier.ipynb    # Interactive exploration
   python predict.py                        # Quick prediction demo
  
---

## 📁 **Project Structure**
project-1-iris-classifier/

├── iris_classifier.ipynb     # Main notebook with all code + visuals

├── iris_classifier.py        

├── predict.py                # Standalone prediction script

├── requirements.txt          # Dependencies

├── README.md                 # You're reading it!

└── LICENSE                   # MIT License

---

## 🔍 **How It Works**

| Step | Description | Key Output |
|------|-------------|------------|
| **Data** | 150 iris flowers × 4 features (sepal/petal length & width) | `data.shape = (150, 5)` |
| **Preprocessing** | 80/20 train-test split + feature scaling | `X_train.shape = (120, 4)` |
| **Models** | Logistic Regression (linear) vs Decision Tree (non-linear) | 97% accuracy |
| **Evaluation** | Accuracy, confusion matrix, 5-fold cross-validation | CV Score: 97.3% |
| **Tuning** | GridSearchCV optimizes tree depth and split criteria | `max_depth=3` |

**Visual Highlights:**
<img src="screenshots/pairplot.png" width="300" alt="Pairplot"> **Feature relationships by class**  
<img src="screenshots/confusion_matrix.png" width="250" alt="Confusion Matrix"> **Model performance**  
<img src="screenshots/decision_tree.png" width="400" alt="Decision Tree"> **Interpretable boundaries**

---

## 📊 **Results**
| Model                | Test Accuracy | Cross-Validation Score |
|----------------------|---------------|-----------------------|
| Logistic Regression  | **97.3%**     | **97.3%**             |
| Decision Tree        | **96.7%**     | **95.8%**             |

---

## 💡 **Key Insights & Learnings**
- **Feature Importance**: Petal length/width >> Sepal dimensions
- **Overfitting**: Unconstrained trees hit 100% train accuracy but drop on test
- **Scaling**: Helps Logistic Regression converge 2x faster
- **Business Value**: 97% accuracy enables automated flower species identification



---

## 🚀 **Quick Start**
1. **Clone**: `git clone https://github.com/tommoni98/ML-Journey.git`
2. **Activate**: `.\env\Scripts\activate` (Windows) / `source env/bin/activate` (Mac/Linux)
3. **Install**: `pip install -r requirements.txt`
4. **Run**: `jupyter notebook iris_classifier.ipynb`

---

## 📄 **License**
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

---