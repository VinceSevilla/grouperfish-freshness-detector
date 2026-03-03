"""
Model Evaluation and Visualization
Generates comprehensive visualizations for trained models
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import cv2
from datetime import datetime

# Import GLCM extractor from backend
sys.path.insert(0, str(Path(__file__).parent / 'backend'))
from app.models_service.glcm_extractor import GLCMExtractor


class ModelEvaluator:
    """Evaluates and visualizes trained fish freshness models"""
    
    FRESHNESS_CLASSES = ['fresh', 'less_fresh', 'starting_to_rot', 'rotten']
    CLASS_NAMES = ['Fresh', 'Less Fresh', 'Starting to Rot', 'Rotten']
    COLORS = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']  # Green, Yellow, Orange, Red
    
    def __init__(self, models_dir='backend/results', data_dir='data/processed', output_dir='evaluation_results'):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load feature extractors
        print("Loading feature extractors...")
        self.resnet_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        self.mobilenet_model = tf.keras.applications.MobileNet(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        print("✓ Feature extractors loaded")
    
    def load_images_and_extract_features(self, folder_type):
        """Load images and extract features (same as training)"""
        print(f"\n[LOADER] Loading {folder_type} images...")
        
        resnet_features_list = []
        mobilenet_features_list = []
        glcm_features_list = []
        labels_list = []
        image_paths = []
        
        folder_path = self.data_dir / folder_type
        if not folder_path.exists():
            print(f"⚠ Folder not found: {folder_path}")
            return None, None, None, None
        
        for class_idx, class_name in enumerate(self.FRESHNESS_CLASSES):
            class_path = folder_path / class_name
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
                    
                    # Extract ResNet50 features
                    processed_resnet = preprocess_input(image.astype(np.float32))
                    batch_resnet = np.expand_dims(processed_resnet, axis=0)
                    resnet_feat = self.resnet_model.predict(batch_resnet, verbose=0)
                    resnet_features_list.append(resnet_feat[0])
                    
                    # Extract MobileNetV1 features
                    processed_mobile = mobilenet_preprocess(image.astype(np.float32))
                    batch_mobile = np.expand_dims(processed_mobile, axis=0)
                    mobilenet_feat = self.mobilenet_model.predict(batch_mobile, verbose=0)
                    mobilenet_features_list.append(mobilenet_feat[0])
                    
                    # Extract GLCM features
                    glcm_dict = GLCMExtractor.compute_glcm_summary(image)
                    glcm_feat = self._flatten_glcm_features(glcm_dict)
                    glcm_features_list.append(glcm_feat)
                    
                    labels_list.append(class_idx)
                    image_paths.append(img_path)
                
                except Exception as e:
                    print(f"⚠ Error processing {img_path}: {e}")
                    continue
        
        if not resnet_features_list:
            return None, None, None, None
        
        X_resnet = np.array(resnet_features_list)
        X_mobilenet = np.array(mobilenet_features_list)
        X_cnn = np.concatenate([X_resnet, X_mobilenet], axis=1)
        X_glcm = np.array(glcm_features_list)
        y = np.array(labels_list)
        
        print(f"✓ Loaded {len(y)} images")
        return X_cnn, X_glcm, y, image_paths
    
    def _flatten_glcm_features(self, glcm_dict):
        """Flatten GLCM dict into 29-feature vector"""
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
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.CLASS_NAMES, 
                    yticklabels=self.CLASS_NAMES,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name.upper()} Model', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        output_path = self.output_dir / f'confusion_matrix_{model_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix: {output_path}")
        plt.close()
    
    def plot_classification_metrics(self, y_true, y_pred, model_name):
        """Plot precision, recall, F1-score per class"""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
        
        x = np.arange(len(self.CLASS_NAMES))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        ax.bar(x, recall, width, label='Recall', color='#2ecc71')
        ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xlabel('Freshness Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Classification Metrics - {model_name.upper()} Model', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.CLASS_NAMES)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f'classification_metrics_{model_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved classification metrics: {output_path}")
        plt.close()
    
    def plot_class_distribution(self, y_true, y_pred, model_name):
        """Plot true vs predicted class distributions"""
        true_counts = np.bincount(y_true, minlength=4)
        pred_counts = np.bincount(y_pred, minlength=4)
        
        x = np.arange(len(self.CLASS_NAMES))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, true_counts, width, label='True Labels', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, pred_counts, width, label='Predictions', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Freshness Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Class Distribution - {model_name.upper()} Model', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.CLASS_NAMES)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f'class_distribution_{model_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved class distribution: {output_path}")
        plt.close()
    
    def plot_sample_predictions(self, X_cnn, X_glcm, y_true, image_paths, model, model_name, num_samples=12):
        """Plot sample predictions with images"""
        # Get predictions
        y_pred_probs = model.predict([X_cnn, X_glcm], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Sample images (mix of correct and incorrect)
        correct_idx = np.where(y_pred == y_true)[0]
        incorrect_idx = np.where(y_pred != y_true)[0]
        
        # Select samples
        sample_idx = []
        if len(correct_idx) >= num_samples // 2:
            sample_idx.extend(np.random.choice(correct_idx, num_samples // 2, replace=False))
        else:
            sample_idx.extend(correct_idx)
        
        if len(incorrect_idx) >= num_samples // 2:
            sample_idx.extend(np.random.choice(incorrect_idx, num_samples // 2, replace=False))
        else:
            sample_idx.extend(incorrect_idx)
        
        if len(sample_idx) < num_samples:
            remaining = num_samples - len(sample_idx)
            all_idx = set(range(len(y_true))) - set(sample_idx)
            sample_idx.extend(np.random.choice(list(all_idx), min(remaining, len(all_idx)), replace=False))
        
        sample_idx = sample_idx[:num_samples]
        
        # Plot
        rows = 3
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        fig.suptitle(f'Sample Predictions - {model_name.upper()} Model', fontsize=18, fontweight='bold', y=0.995)
        
        for i, idx in enumerate(sample_idx):
            if i >= rows * cols:
                break
            
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Load and display image
            img = cv2.imread(str(image_paths[idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            ax.imshow(img)
            
            # Get prediction info
            true_class = self.CLASS_NAMES[y_true[idx]]
            pred_class = self.CLASS_NAMES[y_pred[idx]]
            confidence = y_pred_probs[idx][y_pred[idx]] * 100
            
            # Color based on correctness
            color = '#2ecc71' if y_pred[idx] == y_true[idx] else '#e74c3c'
            
            # Title
            title = f'True: {true_class}\nPred: {pred_class} ({confidence:.1f}%)'
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(len(sample_idx), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / f'sample_predictions_{model_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved sample predictions: {output_path}")
        plt.close()
    
    def evaluate_model(self, model_path, folder_type):
        """Complete evaluation of a model"""
        model_name = folder_type
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()} MODEL")
        print(f"{'='*60}")
        
        # Load model
        if not model_path.exists():
            print(f"✗ Model not found: {model_path}")
            return None
        
        model = load_model(str(model_path), compile=False)
        print(f"✓ Loaded model: {model_path.name}")
        
        # Load data
        X_cnn, X_glcm, y, image_paths = self.load_images_and_extract_features(folder_type)
        if X_cnn is None:
            print(f"✗ Failed to load data for {folder_type}")
            return None
        
        # Split data (same as training)
        X_cnn_train, X_cnn_test, X_g_train, X_g_test, y_train, y_test, paths_train, paths_test = train_test_split(
            X_cnn, X_glcm, y, image_paths, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n[EVALUATION] Test set: {len(y_test)} images")
        print(f"  Class distribution: {np.bincount(y_test)}")
        
        # Get predictions
        y_pred_probs = model.predict([X_cnn_test, X_g_test], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✓ Test Accuracy: {accuracy*100:.2f}%")
        
        # Print classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.CLASS_NAMES, digits=4))
        
        # Generate visualizations
        print(f"\n[VISUALIZATIONS] Generating plots...")
        self.plot_confusion_matrix(y_test, y_pred, model_name)
        self.plot_classification_metrics(y_test, y_pred, model_name)
        self.plot_class_distribution(y_test, y_pred, model_name)
        self.plot_sample_predictions(X_cnn_test, X_g_test, y_test, paths_test, model, model_name)
        
        return {
            'accuracy': accuracy,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs
        }
    
    def evaluate_all(self):
        """Evaluate all available models"""
        results = {}
        
        for folder_type in ['eyes', 'gills', 'eyes_and_gills']:
            model_path = self.models_dir / f'best_model_{folder_type}.h5'
            if model_path.exists():
                result = self.evaluate_model(model_path, folder_type)
                if result:
                    results[folder_type] = result
        
        # Summary
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        for model_name, result in results.items():
            print(f"{model_name.upper()}: Accuracy = {result['accuracy']*100:.2f}%")
        
        print(f"\n✓ All visualizations saved to: {self.output_dir}")
        print(f"✓ Evaluation complete!")


if __name__ == '__main__':
    evaluator = ModelEvaluator(
        models_dir='backend/results',
        data_dir='data/processed',
        output_dir='evaluation_results'
    )
    evaluator.evaluate_all()
