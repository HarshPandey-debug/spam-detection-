# Spam Detection using Machine Learning

## Overview
This project implements a spam detection model using **Scikit-Learn** and **Naïve Bayes classifier**. The model is trained on the **SMS Spam Collection Dataset** to classify messages as either spam or ham (not spam).

## Features
- Data preprocessing using **TF-IDF vectorization**
- Model training with **Multinomial Naïve Bayes**
- Performance evaluation with **accuracy, classification report, and confusion matrix**
- Visualization of results using **Seaborn**

## Dataset
The dataset used is the **SMS Spam Collection Dataset**, which contains labeled SMS messages for spam detection. The dataset is automatically downloaded and extracted from the UCI Machine Learning Repository.

## Installation & Usage
### Prerequisites
Ensure you have **Python 3.x** and the following libraries installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-detection.git
   cd spam-detection
   ```
2. Run the Python script:
   ```bash
   python spamdetection.py
   ```

## Model Performance
The trained model achieves high accuracy on spam detection. Results include:
- **Accuracy Score**
- **Classification Report**
- **Confusion Matrix**

## Example Output
```
Accuracy: 0.98
Classification Report:
              precision    recall  f1-score   support

         ham       0.99      0.99      0.99      965
        spam       0.95      0.96      0.95      150

    accuracy                           0.98     1115
   macro avg       0.97      0.97      0.97     1115
weighted avg       0.98      0.98      0.98     1115
```

## License
This project is licensed under the **MIT License**.

## Contributing
Feel free to submit pull requests for improvements or additional features.

## Author
Harsh Pandey

