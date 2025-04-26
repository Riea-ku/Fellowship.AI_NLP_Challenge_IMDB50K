import tensorflow as tf
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from time import time
import logging


# Configuration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    MAX_TOKENS = 20000
    BATCH_SIZE = 64
    EPOCHS = 15
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    DATA_PATH = 'IMDB Dataset.csv'
    SAMPLE_SIZE = 10000  # Set to None for full dataset


# Text Cleaning 

class TextCleaner:
    """Enhanced text preprocessing with more thorough cleaning"""
    def __init__(self):
        self.contractions = {
            "can't": "cannot", "won't": "will not", "n't": " not",
            "'re": " are", "'s": " is", "'d": " would", 
            "'ll": " will", "'ve": " have", "'m": " am",
            "it's": "it is", "that's": "that is"
        }
        # Common internet slang and abbreviations
        self.slang_map = {
            "brb": "be right back", "lol": "laugh out loud",
            "omg": "oh my god", "btw": "by the way",
            "imho": "in my humble opinion"
        }

    def clean(self, text):
        """Perform comprehensive text cleaning"""
        if not isinstance(text, str):
            text = str(text)
            
        # Lowercase and expand contractions
        text = text.lower()
        for c, full in self.contractions.items():
            text = text.replace(c, full)
            
        # Replace internet slang
        for slang, full in self.slang_map.items():
            text = re.sub(rf'\b{slang}\b', full, text)
            
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        # Handle repeated punctuation
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


# Model Constructor

def build_model(train_text_ds):
    """Build an enhanced text classification model"""
    text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text_input')

    # Corrected text vectorization
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=Config.MAX_TOKENS,
        output_mode='tf_idf',
        ngrams=2,
        name='text_vectorization'
    )
    vectorizer.adapt(train_text_ds)

    x = vectorizer(text_input)
    x = tf.keras.layers.Dense(256, activation='relu', 
                             kernel_regularizer=tf.keras.regularizers.l2(0.01),
                             name='dense_1')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_1')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_norm_1')(x)
    x = tf.keras.layers.Dense(128, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.01),
                             name='dense_2')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout_2')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

    model = tf.keras.Model(inputs=text_input, outputs=output)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model, vectorizer


# Visualization Tools

def plot_training_history(history):
    """Enhanced training history visualization"""
    plt.figure(figsize=(14, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Enhanced confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve with AUC score"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_class_distribution(y_true, y_pred):
    """Plot the distribution of predicted classes"""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_true)
    plt.title('True Class Distribution')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x=y_pred)
    plt.title('Predicted Class Distribution')
    
    plt.tight_layout()
    plt.show()


# Data Analysis Tools

def analyze_misclassifications(df, y_true, y_pred):
    """Perform detailed analysis of misclassified examples"""
    misclassified = df[y_true != y_pred].copy()
    misclassified['prediction'] = y_pred[y_true != y_pred]
    
    # Calculate error rate by review length
    misclassified['length'] = misclassified['cleaned'].apply(len)
    misclassified['error_type'] = np.where(
        (misclassified['sentiment'] == 0) & (misclassified['prediction'] == 1),
        'False Positive',
        'False Negative'
    )
    
    print("\nMisclassification Analysis:")
    print(f"Total misclassified: {len(misclassified)}")
    print(f"False Positives: {sum(misclassified['error_type'] == 'False Positive')}")
    print(f"False Negatives: {sum(misclassified['error_type'] == 'False Negative')}")
    
    # Plot error distribution by review length
    plt.figure(figsize=(10, 5))
    sns.histplot(data=misclassified, x='length', hue='error_type', 
                 bins=30, kde=True, alpha=0.6)
    plt.title('Error Distribution by Review Length')
    plt.xlabel('Review Length (characters)')
    plt.ylabel('Count')
    plt.show()
    
    return misclassified

def generate_classification_report(y_true, y_pred):
    """Generate comprehensive classification report"""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                               target_names=['Negative', 'Positive']))
    
    # Calculate additional metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate: {fp / (fp + tn):.4f}")
    print(f"False Negative Rate: {fn / (fn + tp):.4f}")


# Main Routine

def main():
    logger.info("Starting IMDB sentiment analysis...")
    start_time = time()
    
    # Initialize text cleaner
    cleaner = TextCleaner()
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    try:
        df = pd.read_csv(Config.DATA_PATH, nrows=Config.SAMPLE_SIZE)
        logger.info(f"Successfully loaded {len(df)} samples.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Detect key columns
    text_col = next((col for col in df.columns if df[col].dtype == 'object'), None)
    target_col = next((col for col in df.columns if df[col].nunique() == 2), None)
    
    if not text_col or not target_col:
        logger.error("Could not identify text/sentiment columns.")
        return
    
    # Preprocess data
    df['sentiment'] = df[target_col].map({'positive': 1, 'negative': 0, '1': 1, '0': 0})
    df['cleaned'] = df[text_col].apply(cleaner.clean)
    
    # Check for class imbalance
    class_dist = df['sentiment'].value_counts(normalize=True)
    logger.info(f"Class distribution:\n{class_dist}")
    
    # Data split
    train_df, test_df = train_test_split(
        df, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE,
        stratify=df['sentiment']
    )
    
    # Reset indices to avoid index mismatch later
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # TF Datasets
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_df['cleaned'], train_df['sentiment'])
    ).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_df['cleaned'], test_df['sentiment'])
    ).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Build and train model
    logger.info("Building and training model...")
    model, vectorizer = build_model(train_ds.map(lambda x, y: x))
    
    # Model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, 
            restore_best_weights=True,
            monitor='val_auc',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Training
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    logger.info("\nModel Evaluation:")
    evaluation = model.evaluate(test_ds, return_dict=True)
    for metric, value in evaluation.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Predictions
    y_pred_proba = model.predict(test_ds).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    y_true = test_df['sentiment'].values
    
    # Visualizations
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_pred_proba)
    plot_class_distribution(y_true, y_pred)
    
    # Analytics
    generate_classification_report(y_true, y_pred)
    misclassified_df = analyze_misclassifications(test_df, y_true, y_pred)
    
    # Display sample misclassifications
    logger.info("\nSample Misclassifications:")
    sample_mistakes = misclassified_df.sample(
        min(3, len(misclassified_df)), 
        random_state=Config.RANDOM_STATE
    )
    
    for _, row in sample_mistakes.iterrows():
        logger.info(f"\nReview: {row['cleaned'][:200]}...")
        logger.info(f"True: {'Positive' if row['sentiment'] else 'Negative'} | "
                   f"Predicted: {'Positive' if row['prediction'] else 'Negative'}")
        logger.info(f"Prediction Confidence: {y_pred_proba[row.name]:.2%}")
    
    # Execution time
    end_time = time()
    logger.info(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")

if __name__ == '__main__':
    main()
