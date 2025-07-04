# Emotion Detection Project

This project implements a comprehensive emotion detection system using various machine learning and deep learning approaches, including traditional ML models and BERT-based transformers.

## Project Structure

```
Final_submission/
├── config.py               # All paths, labels, constants, and model parameters
├── main.py                 # Predict function using trained model
├── train.py                # Preprocesses, trains, and saves the model
├── evaluate.py             # Evaluates the model and prints classification metrics
├── requirements.txt        # Python dependencies
│
├── data/
│   ├── track-a.csv
│   ├── eng.csv
│   └── nrc_lexicon.txt
│
├── objects/
│   ├── dataset.py
│   └── distilbert.py
│
├── preprocess/
│   ├── data_augmentation.py
│   ├── data_balancing.py
│   └── data_clearing.py
│
└── utils/
    ├── lexicons.py
    └── nltk_resources.py
```

## Features

### Data Processing
- Text preprocessing with NLTK (tokenization, stopword removal, lemmatization)
- Multi-label emotion classification support
- Data augmentation using synonym replacement
- Dataset balancing techniques

### Models Supported
- **Traditional ML Models:**
  - Support Vector Machine (SVM) with calibration
  - Logistic Regression with elastic net regularization
  - Random Forest
  - Gradient Boosting
  - Multi-layer Perceptron (MLP)
  - Ensemble methods

- **Deep Learning Models:**
  - DistilBERT for emotion classification
  - RoBERTa for emotion classification
  - Specialized emotion BERT model
  - Hierarchical Attention Networks
  - Custom multi-label BERT classifier with attention mechanisms

### Model Strategy: DistilBERT with Linguistic Feature Augmentation

The model is based on `DistilBERT` (`distilbert-base-uncased`) and enhanced with additional linguistic features to improve multi-label emotion classification. Two types of features were concatenated with the [CLS] token output:

- **Handcrafted features**: Character and word counts capturing basic text structure.
- **Lexicon-based features**: Emotion vectors derived from the NRC Emotion Lexicon for five emotions (anger, fear, joy, sadness, surprise).

The combined 775-dimensional input passes through two fully connected layers with batch normalization, ReLU activation, and dropout, followed by a sigmoid output layer. Post-training, class-specific thresholds were tuned using F1 score optimization on validation data.


### Evaluation Features
- Cross-validation for model selection
- Threshold optimization for multi-label classification
- Comprehensive evaluation metrics (F1-macro, F1-micro, F1-weighted)
- External test set evaluation
- Detailed classification reports

## Installation

1. Clone or download the project files
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to set your data paths and parameters:

```python
# Data paths
CSV_FILE_PATH = 'path/to/your/training_data.csv'
TEST_FILE_PATH = 'path/to/your/test_data.csv'

# Model parameters
EMOTIONS = ['anger', 'fear', 'joy', 'sadness', 'surprise']
# ... other parameters
```

## Usage

### Running the Complete Pipeline

Simply run the main script:
```bash
python main.py
```

This will:
1. Load and preprocess the data
2. Perform data augmentation
3. Train multiple models
4. Compare model performance
5. Evaluate on external test set (if available)
6. Save trained models and results

### Using the Predict Function (Required for Submission)

The main `predict` function is located in `main.py` and can be used as follows:

```python
from main import predict

# Make predictions on a CSV file
predictions = predict('path/to/your/test_data.csv')

# predictions is a numpy array of shape (n_samples, 5)
# Each row contains binary predictions for [anger, fear, joy, sadness, surprise]
```

**Important**: The `predict` function:
- Takes a CSV file path as input
- Returns a numpy array of predictions
- Automatically loads the best trained model
- Handles both BERT and traditional ML models
- Applies optimized thresholds for better performance

### Additional Prediction Functions

```python
from prediction import predict_with_probabilities, save_predictions_to_csv

# Get both predictions and probabilities
predictions, probabilities = predict_with_probabilities('test_data.csv')

# Save predictions to a new CSV file
output_file = save_predictions_to_csv('test_data.csv', 'output_predictions.csv')
```

### Training Specific Models

To train only specific models, modify the `model_subset` parameter in `main.py`:

```python
# Train only specific models
model_subset = ['RoBERTa', 'SVM', 'LogisticRegression']
trained_models, tfidf_vectorizer, results, test_data, best_model_name = train_and_compare_enhanced_models(
    augmented_data, model_subset
)
```

### Using Individual Components

You can also use individual components separately:

```python
from data_preprocessing import load_and_preprocess_data
from training import train_and_compare_enhanced_models

# Load data
data = load_and_preprocess_data('your_data.csv')

# Train models
trained_models, tfidf, results, test_data, best_model = train_and_compare_enhanced_models(data)
```

## Output Files

The system saves several files after training:
- `trained_models.pkl`: All trained models
- `tfidf_vectorizer.pkl`: Fitted TF-IDF vectorizer
- `results.pkl`: Training and evaluation results
- `best_model_name.pkl`: Name of the best performing model

## Model Performance

The system automatically compares all models and identifies the best performer based on F1-weighted score. Results are displayed in a comparison table showing:
- Cross-validation F1-macro scores
- Test set F1-macro, F1-micro, and F1-weighted scores
- Detailed classification reports for each emotion class

## Submission Requirements

This project meets the submission requirements:

✅ **Contains `main.py`** with the required `predict` function  
✅ **Predict function signature**: `predict(csv_file_path)` → returns numpy array  
✅ **Handles CSV files** with the same structure as training data  
✅ **Returns predictions** for each item in the CSV file  
✅ **Maintains all existing functionality** for training and evaluation  

## Customization

### Adding New Models
Add new models in `models.py` within the `create_enhanced_models()` function:

```python
def create_enhanced_models():
    models = {}
    # ... existing models
    
    # Add your custom model
    models['YourModel'] = YourCustomModel()
    
    return models
```

### Modifying Data Augmentation
Customize augmentation strategies in `data_augmentation.py`:

```python
def custom_augmentation_strategy(data):
    # Your custom augmentation logic
    return augmented_data
```

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Transformers 4.30+
- Scikit-learn 1.3+
- NLTK 3.8+
- Pandas 1.5+
- NumPy 1.24+
- imbalanced-learn 0.11+

## Notes

- BERT models require significant computational resources and may take longer to train
- The system automatically detects CUDA availability for GPU acceleration
- Cross-validation is skipped for BERT models to reduce training time
- Threshold optimization is performed for better multi-label classification performance
- The `predict` function automatically uses the best performing model from training

## Troubleshooting

1. **NLTK Data Issues**: The system automatically downloads required NLTK data to the specified directory
2. **Memory Issues**: Reduce batch size for BERT models or train fewer models simultaneously
3. **CUDA Issues**: The system falls back to CPU if CUDA is not available
4. **File Path Issues**: Ensure all paths in `config.py` are correct and accessible
5. **Missing Model Files**: Run the complete training pipeline first before using the predict function

## License

This project is for educational and research purposes.
