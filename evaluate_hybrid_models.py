"""
Evaluate Hybrid Eyes and Gills Models on True Test Set
- Loads models from backend/results
- Evaluates on data/processed/eyes_split/test and gills_split/test
- Prints accuracy and classification report
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import cv2
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Import GLCM extractor from backend
sys.path.insert(0, str(Path(__file__).parent / 'backend'))
from app.models_service.glcm_extractor import GLCMExtractor

FRESHNESS_CLASSES = ['fresh', 'less_fresh', 'starting_to_rot', 'rotten']
IMG_SIZE = (224, 224)

# Feature extractor models
resnet_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
mobilenet_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, pooling='avg')
glcm_extractor = GLCMExtractor()

def flatten_glcm_features(glcm_dict):
    features = []
    if 'basic' in glcm_dict and glcm_dict['basic']:
        basic = glcm_dict['basic']
        features.extend([
            basic.get('contrast', 0.0),
            basic.get('dissimilarity', 0.0),
            basic.get('homogeneity', 0.0),
            basic.get('energy', 0.0),
            basic.get('correlation', 0.0),
            basic.get('ASM', 0.0),
        ])
    else:
        features.extend([0.0] * 6)
    if 'multi_scale' in glcm_dict and glcm_dict['multi_scale']:
        for scale in [1, 2, 3]:
            if scale in glcm_dict['multi_scale']:
                scale_props = glcm_dict['multi_scale'][scale]
                features.extend([
                    scale_props.get('contrast', 0.0),
                    scale_props.get('dissimilarity', 0.0),
                    scale_props.get('homogeneity', 0.0),
                    scale_props.get('energy', 0.0),
                    scale_props.get('correlation', 0.0),
                    scale_props.get('ASM', 0.0),
                ])
            else:
                features.extend([0.0] * 6)
    else:
        features.extend([0.0] * 18)
    if 'directional_variance' in glcm_dict and glcm_dict['directional_variance']:
        dv = glcm_dict['directional_variance']
        features.extend([
            dv.get('contrast', 0.0),
            dv.get('dissimilarity', 0.0),
            dv.get('homogeneity', 0.0),
            dv.get('energy', 0.0),
            dv.get('correlation', 0.0),
        ])
    else:
        features.extend([0.0] * 5)
    return np.array(features[:29], dtype=np.float32)

def extract_features(image):
    # EXACTLY match training script: preprocess, extract, and return as separate arrays
    # ResNet50 expects images in RGB, float32, range [-1, 1] after preprocess_input
    img_resnet = tf.keras.applications.resnet50.preprocess_input((image * 255).astype(np.float32))
    resnet_feat = resnet_model.predict(np.expand_dims(img_resnet, axis=0), verbose=0)[0]
    # MobileNet expects images in RGB, float32, range [-1, 1] after preprocess_input
    img_mobilenet = tf.keras.applications.mobilenet.preprocess_input((image * 255).astype(np.float32))
    mobilenet_feat = mobilenet_model.predict(np.expand_dims(img_mobilenet, axis=0), verbose=0)[0]
    # GLCM features
    glcm_dict = glcm_extractor.compute_glcm_summary(image)
    glcm_feat = flatten_glcm_features(glcm_dict)
    return resnet_feat, mobilenet_feat, glcm_feat

def load_test_data(data_dir, split_folder):
    X_resnet, X_mobilenet, X_glcm, y = [], [], [], []
    for class_idx, class_name in enumerate(FRESHNESS_CLASSES):
        class_path = Path(data_dir) / split_folder / 'test' / class_name
        if not class_path.exists():
            continue
        image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        print(f"  {class_name}: {len(image_files)} images")
        for img_path in image_files:
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMG_SIZE)
                image = image.astype(np.float32) / 255.0
                resnet_feat, mobilenet_feat, glcm_feat = extract_features(image)
                X_resnet.append(resnet_feat)
                X_mobilenet.append(mobilenet_feat)
                X_glcm.append(glcm_feat)
                y.append(class_idx)
            except Exception as e:
                print(f"⚠ Error processing {img_path}: {e}")
                continue
    X_resnet = np.array(X_resnet)
    X_mobilenet = np.array(X_mobilenet)
    X_glcm = np.array(X_glcm)
    y = np.array(y)
    print(f"✓ Loaded {len(y)} test samples with extracted features")
    X_cnn = np.concatenate([X_resnet, X_mobilenet], axis=1)
    print(f"[DEBUG] X_cnn shape: {X_cnn.shape}, X_glcm shape: {X_glcm.shape}, y shape: {y.shape}")
    return X_cnn, X_glcm, y

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name} Model', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    output_path = Path(output_dir) / f'confusion_matrix_{model_name.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix: {output_path}")

def plot_classification_metrics(y_true, y_pred, class_names, model_name, output_dir):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    ax.set_xlabel('Freshness Class')
    ax.set_ylabel('Score')
    ax.set_title(f'Classification Metrics - {model_name} Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    output_path = Path(output_dir) / f'classification_metrics_{model_name.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved classification metrics: {output_path}")

def evaluate_model(model_path, data_dir, split_folder, model_name, output_dir):
    print(f"\n=== Evaluating {model_name} model on TEST set ===")
    model = load_model(model_path, compile=False)
    X_cnn, X_glcm, y_test = load_test_data(data_dir, split_folder)
    if X_cnn.shape[0] == 0:
        print("No test data found!")
        return
    print(f"[DEBUG] Predict input shapes: {[X_cnn.shape, X_glcm.shape]}")
    y_pred_probs = model.predict([X_cnn, X_glcm], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print(f"[DEBUG] First 10 predictions: {y_pred[:10]}")
    print(f"[DEBUG] First 10 true labels: {y_test[:10]}")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=FRESHNESS_CLASSES, digits=4))
    # Visualizations
    os.makedirs(output_dir, exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, FRESHNESS_CLASSES, model_name, output_dir)
    plot_classification_metrics(y_test, y_pred, FRESHNESS_CLASSES, model_name, output_dir)

def main():
    data_dir = 'data/processed'
    output_dir = 'evaluation_results'
    eyes_model_path = 'backend/results/hybrid_eyes_model.h5'
    gills_model_path = 'backend/results/hybrid_gills_model.h5'
    evaluate_model(eyes_model_path, data_dir, 'eyes_split', 'Eyes', output_dir)
    evaluate_model(gills_model_path, data_dir, 'gills_split', 'Gills', output_dir)

if __name__ == '__main__':
    main()
