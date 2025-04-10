# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# def compute_confusion_matrix(y_true, y_pred, num_classes):
#     y_true = tf.constant(y_true, dtype=tf.int32)
#     y_pred = tf.constant(y_pred, dtype=tf.int32)
    
#     confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)
#     return confusion_matrix.numpy()

# def plot_confusion_matrix(confusion_matrix, class_names):
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.show()

# # Example usage
# if __name__ == "__main__":
#     y_true = [1, 0, 1, 1, 0]
#     y_pred = [1, 0, 1, 0, 0]
#     num_classes = 2
#     class_names = ['Class 0', 'Class 1']
    
#     confusion_matrix = compute_confusion_matrix(y_true, y_pred, num_classes)
#     print("Confusion Matrix:")
#     print(confusion_matrix)
    
#     plot_confusion_matrix(confusion_matrix, class_names)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import load_model

def compute_confusion_matrix(y_true, y_pred, num_classes):
    y_true = tf.constant(y_true, dtype=tf.int32)
    y_pred = tf.constant(y_pred, dtype=tf.int32)
    
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)
    return confusion_matrix.numpy()

def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate_model(model_path, test_dir):
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    # Check if test directory exists
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at: {test_dir}")
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    print(f"Loading test data from: {test_dir}")
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(75, 75),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    
    print("Generating predictions...")
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    class_names = list(test_generator.class_indices.keys())
    num_classes = len(class_names)
    
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    plot_confusion_matrix(cm, class_names)
    
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    # Use absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    model_path = os.path.join(project_dir, 'best_model.h5')
    test_dir = os.path.join(project_dir, 'DATASET', 'TEST')
    
    print(f"Model path: {model_path}")
    print(f"Test directory: {test_dir}")
    
    evaluate_model(model_path, test_dir)