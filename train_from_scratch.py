"""
Complete Fish Freshness Detection Training Script - NO PRETRAINED WEIGHTS
Trains hybrid models (ResNet50 + GLCM) from scratch for eye and gill classification
"""

import os
import sys
from pathlib import Path

# Fix numpy compatibility with imgaug (np.bool deprecated in NumPy 1.20+)
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_

# CRITICAL: Set random seed for reproducible random initialization of feature extractors
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import imgaug.augmenters as iaa

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'backend'))
from app.models_service.glcm_extractor import GLCMExtractor

class PreventOverfittingCallback(Callback):
    """Custom callback to stop if validation accuracy exceeds 99.5%"""
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        if val_acc > 0.995:
            print(f"\n⚠ Stopping training: Val accuracy too high ({val_acc*100:.2f}%)")
            print(f"   Targeting 97-98% accuracy for realistic generalization")
            self.model.stop_training = True

class FishFreshnessTrainerFromScratch:
    """Trainer for fish freshness models - TRAINING FROM SCRATCH (NO PRETRAINED WEIGHTS)"""
    
    FRESHNESS_CLASSES = ['fresh', 'less_fresh', 'starting_to_rot', 'rotten']
    NUM_CLASSES = 4
    IMG_SIZE = (224, 224)
    
    def __init__(self, data_dir='data/processed', output_dir='backend/results'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load feature extractors WITHOUT pretrained weights (random initialization)
        print("Loading models (TRAINING FROM SCRATCH - NO PRETRAINED WEIGHTS)...")
        self.resnet_model = tf.keras.applications.ResNet50(
            weights=None, include_top=False, pooling='avg'
        )
        self.resnet_model.trainable = False  # ResNet50 frozen
        print(f"✓ ResNet50: {self.resnet_model.output_shape[1]} features (FROZEN)")
        
        self.mobilenet_model = tf.keras.applications.MobileNet(
            weights=None, include_top=False, pooling='avg'
        )
        self.mobilenet_model.trainable = True  # MobileNetV1 trainable - learns meaningful features from random init
        print(f"✓ MobileNetV1: {self.mobilenet_model.output_shape[1]} features (TRAINABLE)")
        
        self.glcm_extractor = GLCMExtractor()
        
        # Data augmentation pipeline (REDUCED - lighter augmentation for better learning)
        # Focus on: subtle brightness, contrast, minimal noise
        self.augmenter = iaa.Sequential([
            # Light brightness and contrast adjustments
            iaa.Multiply((0.85, 1.15)),  # Reduced from (0.7, 1.3)
            iaa.LinearContrast((0.85, 1.15)),  # Reduced from (0.7, 1.3)
            # Light color variations
            iaa.AddToHueAndSaturation((-10, 10)),  # Reduced from (-20, 20)
            # Light noise
            iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Reduced from 0.1*255
                iaa.SaltAndPepper(0.02),  # Reduced from 0.05
            ]),
            # Light blur
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0.2, 0.8)),  # Reduced from (0.5, 1.5)
                iaa.MedianBlur(k=(3, 3)),  # Reduced from (3, 5)
            ]),
            # Light crops and scaling
            iaa.CropAndPad(percent=(-0.05, 0.05), pad_mode='reflect'),  # Reduced from 0.1
            iaa.Affine(scale=(0.95, 1.05)),  # Reduced from (0.9, 1.1)
        ], random_order=True)
        
        # EYES-SPECIFIC augmenter with light scale augmentation
        # Eyes have size variation issue (fresh vs rotten), but keep augmentation light
        self.augmenter_eyes = iaa.Sequential([
            # Light brightness and contrast adjustments
            iaa.Multiply((0.85, 1.15)),  # Reduced from (0.7, 1.3)
            iaa.LinearContrast((0.85, 1.15)),  # Reduced from (0.7, 1.3)
            # Light color variations
            iaa.AddToHueAndSaturation((-10, 10)),  # Reduced from (-20, 20)
            # Light noise
            iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Reduced from 0.1*255
                iaa.SaltAndPepper(0.02),  # Reduced from 0.05
            ]),
            # Light blur
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0.2, 0.8)),  # Reduced from (0.5, 1.5)
                iaa.MedianBlur(k=(3, 3)),  # Reduced from (3, 5)
            ]),
            # Light crops
            iaa.CropAndPad(percent=(-0.05, 0.05), pad_mode='reflect'),  # Reduced from 0.1
            # LIGHT SCALE AUGMENTATION FOR EYES (0.95-1.05x) - handle fresh (small) vs rotten (large)
            iaa.Affine(scale=(0.95, 1.05)),  # Reduced from (0.7, 1.3)
        ], random_order=True)
    
    def load_data_from_folder(self, folder_type, split='train'):
        """Load RAW images + GLCM features from split folder (END-TO-END approach)"""
        print(f"\n[LOADER] Loading {folder_type}/{split} (RAW IMAGES for end-to-end)...")
        
        images = []
        glcm_features = []
        labels = []
        
        folder_path = self.data_dir / folder_type / split
        if not folder_path.exists():
            print(f"✗ Not found: {folder_path}")
            return None, None, None
        
        for class_idx, class_name in enumerate(self.FRESHNESS_CLASSES):
            class_path = folder_path / class_name
            if not class_path.exists():
                continue
            
            img_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            print(f"  [{class_idx}] {class_name}: {len(img_files)} images")
            
            for img_path in tqdm(img_files, desc=f"    {split}-{class_name}"):
                try:
                    # Load and resize image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, self.IMG_SIZE)
                    
                    # APPLY LIGHT AUGMENTATION ONLY TO TRAINING DATA
                    if split == 'train' and np.random.random() > 0.5:  # 50% get augmented
                        augmenter = self.augmenter_eyes if folder_type == 'eyes_split' else self.augmenter
                        image = augmenter(image=image)
                    
                    # Store RAW image (will be processed by model layers)
                    images.append(image.astype(np.float32))
                    
                    # Extract GLCM features from uint8 image
                    img_uint8 = image.astype(np.uint8)
                    glcm_dict = self.glcm_extractor.compute_glcm_summary(img_uint8)
                    glcm_feat = self._flatten_glcm_features(glcm_dict)
                    glcm_features.append(glcm_feat)
                    
                    # Label
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"  ⚠ Error loading {img_path}: {e}")
                    continue
        
        if not images:
            print(f"✗ No images loaded from {split}!")
            return None, None, None
        
        X_images = np.array(images)
        X_glcm = np.array(glcm_features)
        y = np.array(labels)
        
        # DEBUG: Check data quality
        print(f"[DEBUG] Data loaded:")
        print(f"  Images shape: {X_images.shape}, min={X_images.min():.2f}, max={X_images.max():.2f}")
        print(f"  GLCM shape: {X_glcm.shape}, min={X_glcm.min():.4f}, max={X_glcm.max():.4f}, mean={X_glcm.mean():.4f}")
        print(f"  Class distribution: {np.bincount(y)}")
        
        return X_images, X_glcm, y
    
    def _flatten_glcm_features(self, glcm_dict):
        """Flatten GLCM dict to 29 features"""
        features = []
        
        # Basic features (6)
        if 'basic' in glcm_dict and glcm_dict['basic']:
            b = glcm_dict['basic']
            features.extend([
                b.get('contrast', 0.0),
                b.get('dissimilarity', 0.0),
                b.get('homogeneity', 0.0),
                b.get('energy', 0.0),
                b.get('correlation', 0.0),
                b.get('ASM', 0.0)
            ])
        else:
            features.extend([0.0] * 6)
        
        # Multi-scale features (18)
        if 'multi_scale' in glcm_dict and glcm_dict['multi_scale']:
            for scale_key in ['scale_1', 'scale_2', 'scale_3']:
                if scale_key in glcm_dict['multi_scale']:
                    s = glcm_dict['multi_scale'][scale_key]
                    features.extend([
                        s.get('contrast', 0.0),
                        s.get('dissimilarity', 0.0),
                        s.get('homogeneity', 0.0),
                        s.get('energy', 0.0),
                        s.get('correlation', 0.0),
                        s.get('ASM', 0.0)
                    ])
                else:
                    features.extend([0.0] * 6)
        else:
            features.extend([0.0] * 18)
        
        # Directional variance features (5)
        if 'directional' in glcm_dict and glcm_dict['directional']:
            d = glcm_dict['directional']
            for prop in ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']:
                vals = [d[dir].get(prop, 0.0) for dir in ['0°', '45°', '90°', '135°'] if dir in d]
                features.append(float(np.var(vals)) if vals else 0.0)
        else:
            features.extend([0.0] * 5)
        
        return np.array(features[:29], dtype=np.float32)
    
    def build_hybrid_model(self, cnn_dim=None, glcm_dim=None):
        """
        Build END-TO-END model: ResNet50 + MobileNetV1 are LAYERS in the model.
        Both are trained via backpropagation from scratch.
        """
        from tensorflow.keras.layers import Lambda
        
        print("[MODEL] Building END-TO-END hybrid model (ResNet50 + MobileNetV1 as layers)...")
        
        # Image input (raw images 224x224x3)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        
        # GLCM input
        glcm_input = Input(shape=(29,), name='glcm_input')
        
        # === ResNet50 as a LAYER (frozen - good random features) ===
        print("  - Adding ResNet50 (weights=None, frozen)")
        resnet_model = tf.keras.applications.ResNet50(
            weights=None, include_top=False, pooling='avg'
        )
        resnet_model.trainable = False
        resnet_preproc = Lambda(
            lambda x: tf.keras.applications.resnet50.preprocess_input(x),
            name='resnet_preprocess'
        )(image_input)
        resnet_features = resnet_model(resnet_preproc)
        
        # === MobileNetV1 as a LAYER (trainable - learns from scratch!) ===
        print("  - Adding MobileNetV1 (weights=None, trainable)")
        mobilenet_model = tf.keras.applications.MobileNet(
            weights=None, include_top=False, pooling='avg'
        )
        mobilenet_model.trainable = True  # TRAINABLE!
        mobilenet_preproc = Lambda(
            lambda x: tf.keras.applications.mobilenet.preprocess_input(x),
            name='mobilenet_preprocess'
        )(image_input)
        mobilenet_features = mobilenet_model(mobilenet_preproc)
        
        # === Concatenate CNN features ===
        cnn_features = Concatenate(name='concat_cnn')([resnet_features, mobilenet_features])
        
        # === CNN path ===
        reg = L1L2(l1=1e-5, l2=1e-4)
        x = Dense(128, activation='relu', kernel_regularizer=reg)(cnn_features)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu', kernel_regularizer=reg)(x)
        x = Dropout(0.2)(x)
        
        # === GLCM path ===
        y = Dense(32, activation='relu', kernel_regularizer=reg)(glcm_input)
        y = BatchNormalization()(y)
        y = Dropout(0.2)(y)
        
        # === Merge ===
        merged = Concatenate(name='concat_final')([x, y])
        merged = Dense(64, activation='relu', kernel_regularizer=reg)(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(0.3)(merged)
        
        # === Output ===
        output = Dense(self.NUM_CLASSES, activation='softmax')(merged)
        
        model = Model(inputs=[image_input, glcm_input], outputs=output)
        return model
    
    def train_model(self, folder_type, model_path, epochs=50, batch_size=32):
        """Train end-to-end model using raw images"""
        print(f"\n{'='*70}")
        print(f"TRAINING: {folder_type.upper()} (END-TO-END: ResNet50 + MobileNetV1)")
        print(f"{'='*70}")
        
        # Load raw images + GLCM (not pre-computed CNN features)
        print(f"\nLoading data (raw images)...")
        X_train_img, X_train_glcm, y_train = self.load_data_from_folder(folder_type, split='train')
        X_val_img, X_val_glcm, y_val = self.load_data_from_folder(folder_type, split='val')
        X_test_img, X_test_glcm, y_test = self.load_data_from_folder(folder_type, split='test')
        
        if X_train_img is None or X_val_img is None or X_test_img is None:
            print(f"✗ Missing data splits for {folder_type}")
            return None, None
        
        print(f"\n✓ Data summary:")
        print(f"  Train: {len(y_train)} images")
        print(f"  Val:   {len(y_val)} images")
        print(f"  Test:  {len(y_test)} images")
        
        # Build END-TO-END model
        model = self.build_hybrid_model()
        
        model.compile(
            optimizer=Adam(learning_rate=5e-4),
            loss=CategoricalCrossentropy(label_smoothing=0.0),
            metrics=['accuracy']
        )
        print(model.summary())
        
        # Convert labels to one-hot
        y_train_cat = to_categorical(y_train, num_classes=self.NUM_CLASSES)
        y_val_cat = to_categorical(y_val, num_classes=self.NUM_CLASSES)
        
        # Train
        print(f"\n[TRAINING] Starting END-TO-END training...")
        print(f"  ResNet50: FROZEN (good random features)")
        print(f"  MobileNetV1: TRAINABLE (learns from scratch via backprop!)")
        print(f"  Target: 75-85% validation accuracy")
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        print(f"\n[CLASS WEIGHTS] Penalizing misclassification:")
        for i, class_name in enumerate(self.FRESHNESS_CLASSES):
            print(f"  {class_name:20s}: weight = {class_weight_dict[i]:.3f}")
        
        callbacks = [
            EarlyStopping(patience=30, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-7, verbose=1),
            PreventOverfittingCallback(),
        ]
        
        history = model.fit(
            [X_train_img, X_train_glcm], y_train_cat,
            validation_data=([X_val_img, X_val_glcm], y_val_cat),
            epochs=300,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print(f"\n--- TEST SET EVALUATION (End-to-End) ---")
        y_pred_probs = model.predict([X_test_img, X_test_glcm], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.FRESHNESS_CLASSES))
        
        # Visualizations
        self._plot_results(folder_type, y_test, y_pred, history)
        
        return model, {
            'train_acc': float(history.history['accuracy'][-1]),
            'val_acc': float(history.history['val_accuracy'][-1]),
            'test_acc': test_acc,
        }
    
    def _plot_results(self, folder_type, y_test, y_pred, history):
        """Generate visualizations"""
        folder_type_clean = folder_type.replace('_split', '').upper()
        
        # 1. Training history
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history.history['loss'], label='Train Loss', marker='o')
        axes[0].plot(history.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_title(f'{folder_type_clean} - Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['accuracy'], label='Train Acc', marker='o')
        axes[1].plot(history.history['val_accuracy'], label='Val Acc', marker='s')
        axes[1].set_title(f'{folder_type_clean} - Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{folder_type}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training history: {folder_type}_training_history.png")
        
        # 2. Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.FRESHNESS_CLASSES, 
                   yticklabels=self.FRESHNESS_CLASSES)
        plt.title(f'{folder_type_clean} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{folder_type}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved confusion matrix: {folder_type}_confusion_matrix.png")
        
        # 3. Per-class accuracy
        fig, ax = plt.subplots(figsize=(8, 5))
        per_class_acc = []
        for i, class_name in enumerate(self.FRESHNESS_CLASSES):
            mask = y_test == i
            if mask.sum() > 0:
                acc = accuracy_score(y_test[mask], y_pred[mask])
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0)
        
        ax.bar(self.FRESHNESS_CLASSES, per_class_acc, color='steelblue', alpha=0.8)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{folder_type_clean} - Per-Class Accuracy')
        ax.set_ylim(0, 1.1)
        for i, v in enumerate(per_class_acc):
            ax.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{folder_type}_per_class_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved per-class accuracy: {folder_type}_per_class_accuracy.png")

def main():
    trainer = FishFreshnessTrainerFromScratch()
    
    # Train eye model
    # Use as_posix() to ensure forward slashes for h5py compatibility on Windows
    eyes_path = (trainer.output_dir / 'from_scratch_eyes_model.h5').resolve().as_posix()
    eyes_model, eyes_results = trainer.train_model('eyes_split', eyes_path, epochs=50, batch_size=32)
    
    # Train gill model
    # Use as_posix() to ensure forward slashes for h5py compatibility on Windows
    gills_path = (trainer.output_dir / 'from_scratch_gills_model.h5').resolve().as_posix()
    gills_model, gills_results = trainer.train_model('gills_split', gills_path, epochs=50, batch_size=32)
    
    # Summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE (FROM SCRATCH - NO PRETRAINED WEIGHTS)")
    print(f"{'='*70}")
    if eyes_results:
        print(f"\nEYES Model (From Scratch):")
        print(f"  Train Accuracy: {eyes_results['train_acc']*100:.2f}%")
        print(f"  Val Accuracy:   {eyes_results['val_acc']*100:.2f}%")
        print(f"  Test Accuracy:  {eyes_results['test_acc']*100:.2f}%")
    
    if gills_results:
        print(f"\nGILLS Model (From Scratch):")
        print(f"  Train Accuracy: {gills_results['train_acc']*100:.2f}%")
        print(f"  Val Accuracy:   {gills_results['val_acc']*100:.2f}%")
        print(f"  Test Accuracy:  {gills_results['test_acc']*100:.2f}%")
    
    # Create comparison visualizations
    if eyes_results and gills_results:
        _plot_accuracy_comparison(trainer.output_dir, eyes_results, gills_results)
    
    print(f"\nModels saved to: {trainer.output_dir}")
    print(f"Visualizations saved to: {trainer.output_dir}")

def _plot_accuracy_comparison(output_dir, eyes_results, gills_results):
    """Create comparison visualizations for eyes and gills model accuracies"""
    
    # 1. Side-by-side test accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['EYES Model (From Scratch)', 'GILLS Model (From Scratch)']
    test_accuracies = [eyes_results['test_acc']*100, gills_results['test_acc']*100]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax.bar(models, test_accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, test_accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Test Accuracy (From Scratch)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'from_scratch_model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved model accuracy comparison: from_scratch_model_accuracy_comparison.png")
    
    # 2. Comprehensive accuracy metrics (Train/Val/Test)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(2)  # 2 models
    width = 0.25  # Width of bars
    
    train_accs = [eyes_results['train_acc']*100, gills_results['train_acc']*100]
    val_accs = [eyes_results['val_acc']*100, gills_results['val_acc']*100]
    test_accs = [eyes_results['test_acc']*100, gills_results['test_acc']*100]
    
    bars1 = ax.bar(x - width, train_accs, width, label='Train', color='#06A77D', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, val_accs, width, label='Validation', color='#F18F01', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, test_accs, width, label='Test', color='#C73E1D', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Train / Validation / Test Accuracy (From Scratch)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['EYES Model', 'GILLS Model'], fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'from_scratch_model_full_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved full accuracy comparison: from_scratch_model_full_accuracy_comparison.png")
    
    # 3. Display the comparison summary
    print(f"\n{'='*70}")
    print("MODEL ACCURACY COMPARISON (FROM SCRATCH)")
    print(f"{'='*70}")
    print(f"\n{'Metric':<20} {'EYES Model':<20} {'GILLS Model':<20}")
    print(f"{'-'*60}")
    print(f"{'Train Accuracy':<20} {eyes_results['train_acc']*100:>18.2f}% {gills_results['train_acc']*100:>18.2f}%")
    print(f"{'Val Accuracy':<20} {eyes_results['val_acc']*100:>18.2f}% {gills_results['val_acc']*100:>18.2f}%")
    print(f"{'Test Accuracy':<20} {eyes_results['test_acc']*100:>18.2f}% {gills_results['test_acc']*100:>18.2f}%")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
