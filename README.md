# Heart Disease Prediction Using Machine Learning 🫀📊

A comprehensive machine learning project that predicts the 10-year risk of coronary heart disease (CHD) using the famous Framingham Heart Study dataset. This project implements logistic regression to analyze various health factors and predict cardiovascular disease risk.

## 🎯 Project Overview

This project uses machine learning techniques to predict the likelihood of developing coronary heart disease within 10 years based on demographic, behavioral, and medical risk factors. The model achieves reliable predictions that can assist in early cardiovascular risk assessment.

## 📊 Dataset Information

**Dataset**: Framingham Heart Study Dataset
- **Source**: Framingham, Massachusetts cardiovascular study
- **Target Variable**: `TenYearCHD` (10-year coronary heart disease risk)
- **Features Used**:
  - `age`: Age of the patient
  - `Sex_male`: Gender (1 = Male, 0 = Female)
  - `cigsPerDay`: Number of cigarettes smoked per day
  - `totChol`: Total cholesterol level
  - `sysBP`: Systolic blood pressure
  - `glucose`: Glucose level

## 🛠️ Technologies & Libraries Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and tools
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

## 📋 Prerequisites

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 🚀 How to Run

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare the dataset**:
   - Download the Framingham dataset (`framingham.csv`)
   - Place it in the project directory
   - Update the file path in the code if necessary

4. **Run the analysis**:
```bash
python heart_disease_prediction.py
```

## 🔍 Project Workflow

### 1. **Data Preprocessing**
- **Data Loading**: Import Framingham Heart Study dataset
- **Feature Selection**: Remove unnecessary columns (education)
- **Data Cleaning**: Handle missing values using dropna()
- **Feature Renaming**: Rename columns for clarity (male → Sex_male)

### 2. **Exploratory Data Analysis**
- **Target Distribution**: Analyze the distribution of CHD cases
- **Data Visualization**: 
  - Count plots showing CHD distribution
  - Statistical visualizations using Seaborn

### 3. **Feature Engineering**
- **Feature Selection**: Choose 6 key predictive features
- **Data Standardization**: Apply StandardScaler for feature normalization
- **Data Splitting**: 70-30 train-test split with random state for reproducibility

### 4. **Model Development**
- **Algorithm**: Logistic Regression
- **Training**: Fit model on training data
- **Prediction**: Generate predictions on test set

### 5. **Model Evaluation**
- **Accuracy Score**: Overall model performance
- **Confusion Matrix**: Detailed prediction analysis
- **Classification Report**: Precision, recall, and F1-score metrics
- **Visualization**: Heatmap of confusion matrix

## 📈 Key Features

- **Comprehensive Data Analysis**: Complete EDA with visualizations
- **Robust Preprocessing**: Proper handling of missing data and feature scaling
- **Model Interpretability**: Clear evaluation metrics and confusion matrix
- **Visualization**: Multiple plots for data understanding and result interpretation
- **Reproducible Results**: Fixed random state for consistent outputs

## 📊 Model Performance Metrics

The model provides detailed performance analysis including:
- **Accuracy Score**: Overall prediction accuracy
- **Confusion Matrix**: True vs. predicted classifications
- **Classification Report**: 
  - Precision: Accuracy of positive predictions
  - Recall: Sensitivity of the model
  - F1-Score: Harmonic mean of precision and recall
  - Support: Number of samples in each class

## 🎨 Visualizations

The project includes several informative visualizations:
- **Count Plot**: Distribution of CHD cases in the dataset
- **Line Plot**: Trend analysis of CHD cases
- **Confusion Matrix Heatmap**: Visual representation of model performance

## 📁 Project Structure

```
heart-disease-prediction/
│
├── heart_disease_prediction.py    # Main analysis script
├── framingham.csv                 # Dataset file
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
└── results/                       # Output visualizations
    ├── chd_distribution.png
    ├── confusion_matrix.png
    └── model_performance.txt
```

## 🔬 Medical Significance

This project addresses a critical healthcare challenge:
- **Early Detection**: Identify high-risk patients before symptoms appear
- **Preventive Care**: Enable proactive healthcare interventions
- **Resource Allocation**: Help healthcare providers prioritize patient care
- **Risk Stratification**: Categorize patients based on cardiovascular risk levels

## 🚀 Future Enhancements

- [ ] **Advanced Algorithms**: Implement Random Forest, SVM, Neural Networks
- [ ] **Feature Engineering**: Add derived features and interaction terms
- [ ] **Cross-Validation**: Implement k-fold cross-validation
- [ ] **Hyperparameter Tuning**: Optimize model parameters using GridSearch
- [ ] **Model Deployment**: Create web application for real-time predictions
- [ ] **Additional Metrics**: ROC curve, AUC score, precision-recall curves
- [ ] **Feature Importance**: Analyze which factors contribute most to predictions

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- Model improvement and optimization
- Additional visualization features
- Code optimization and refactoring
- Documentation enhancements
- Test case development

## 📄 Dataset Citation

```
The Framingham Heart Study dataset is a long-term, ongoing cardiovascular 
cohort study of residents of the city of Framingham, Massachusetts. 
The classification goal is to predict whether the patient has a 10-year 
risk of future coronary heart disease (CHD).
```

## ⚠️ Disclaimer

This project is for educational and research purposes only. The predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult healthcare professionals for medical decisions.

## 👨‍💻 Developer Information

**Developer**: [Your Name]  
**Email**: [your.email@example.com]  
**GitHub**: [https://github.com/yourusername](https://github.com/yourusername)  
**LinkedIn**: [https://linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  

### About the Developer
[Add information about your background in data science, healthcare analytics, or machine learning]

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact via email
- Check documentation for troubleshooting

## 🙏 Acknowledgments

- **Framingham Heart Study** for providing the dataset
- **Scikit-learn** community for excellent ML tools
- **Healthcare Research Community** for domain knowledge
- **Open Source Contributors** for inspiration and support

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Predicting Heart Disease, Saving Lives! 🫀💙**

*Using data science to contribute to better cardiovascular health outcomes.*
