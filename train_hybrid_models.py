def plot_per_class_accuracy(y_true, y_pred, class_names, model_name, output_dir):
    from sklearn.metrics import accuracy_score
    accs = []
    for i, cname in enumerate(class_names):
        idx = (y_true == i)
        if np.sum(idx) > 0:
            accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        else:
            accs.append(0)
    plt.figure(figsize=(8, 5))
    plt.bar(class_names, accs, color='#3498db')
    plt.ylim(0, 1.1)
    plt.title(f'Per-Class Accuracy - {model_name} (Val)', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'val_per_class_accuracy_{model_name.lower()}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved per-class accuracy: {out_path}")

def plot_true_vs_pred_distribution(y_true, y_pred, class_names, model_name, output_dir):
    true_counts = np.bincount(y_true, minlength=len(class_names))
    pred_counts = np.bincount(y_pred, minlength=len(class_names))
    x = np.arange(len(class_names))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, true_counts, width, label='True', color='#2ecc71')
    plt.bar(x + width/2, pred_counts, width, label='Predicted', color='#e74c3c', alpha=0.8)
    plt.xticks(x, class_names)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(f'True vs. Predicted Distribution - {model_name} (Val)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'val_true_vs_pred_dist_{model_name.lower()}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved true vs. predicted distribution: {out_path}")

def plot_sample_predictions(X_cnn, X_glcm, y_true, model, class_names, model_name, output_dir, num_samples=12):
    y_pred_probs = model.predict([X_cnn, X_glcm], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    idxs = np.random.choice(len(y_true), min(num_samples, len(y_true)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    for i, idx in enumerate(idxs):
        ax = axes[i]
        # No access to original images, so just show class info
        ax.axis('off')
        ax.set_title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}",
                     color='green' if y_true[idx]==y_pred[idx] else 'red', fontsize=12)
        ax.text(0.5, 0.5, f"Sample {idx}", fontsize=16, ha='center', va='center')
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    fig.suptitle(f'Sample Predictions - {model_name} (Val)', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, f'val_sample_predictions_{model_name.lower()}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved sample predictions: {out_path}")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
def plot_confusion_matrix_and_metrics(y_true, y_pred, class_names, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name} Model', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'confusion_matrix_{model_name.lower()}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix: {output_path}")
    # Classification metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    x = range(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width for i in x], precision, width, label='Precision', color='#3498db')
    ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    ax.bar([i + width for i in x], f1, width, label='F1-Score', color='#e74c3c')
    ax.set_xlabel('Freshness Class')
    ax.set_ylabel('Score')
    ax.set_title(f'Classification Metrics - {model_name} Model', fontsize=14, fontweight='bold')
    ax.set_xticks(list(x))
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'classification_metrics_{model_name.lower()}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved classification metrics: {output_path}")

def plot_model_summary_accuracy(eyes_acc, gills_acc, output_dir):
    """Create a summary visualization of model test accuracies"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Eyes accuracy visualization
    ax = axes[0]
    ax.barh(['Eye Model'], [eyes_acc * 100], color='#3498db', height=0.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Eyes Model - Test Accuracy', fontsize=14, fontweight='bold')
    ax.text(eyes_acc * 100 + 1, 0, f'{eyes_acc*100:.2f}%', va='center', fontsize=12, fontweight='bold')
    
    # Gills accuracy visualization
    ax = axes[1]
    ax.barh(['Gill Model'], [gills_acc * 100], color='#e74c3c', height=0.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Gills Model - Test Accuracy', fontsize=14, fontweight='bold')
    ax.text(gills_acc * 100 + 1, 0, f'{gills_acc*100:.2f}%', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_summary_test_accuracy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved model summary accuracy: {output_path}")

    # Create detailed summary report
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    summary_text = f"""
    FISH FRESHNESS DETECTION - MODEL SUMMARY
    ═════════════════════════════════════════
    
    EYES MODEL TEST ACCURACY
    ────────────────────────
    Accuracy: {eyes_acc*100:.2f}%
    
    GILLS MODEL TEST ACCURACY
    ─────────────────────────
    Accuracy: {gills_acc*100:.2f}%
    
    OVERALL PERFORMANCE
    ───────────────────
    Average Accuracy: {(eyes_acc + gills_acc)/2 * 100:.2f}%
    
    Classes Detected: fresh, less_fresh, starting_to_rot, rotten
    Prediction Method: Hybrid (ResNet50 + MobileNetV1 + GLCM Texture)
    """
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=13,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    output_path = os.path.join(output_dir, 'model_summary_report.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved model summary report: {output_path}")

import os
import numpy as np
# --- NumPy bool workaround for imgaug compatibility ---
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
class StopOnLowValLoss(Callback):
    def __init__(self, threshold=0.01):
        super().__init__()
        self.threshold = threshold
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.threshold:
            print(f"\nStopping training: val_loss < {self.threshold}")
            self.model.stop_training = True
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import cv2
from pathlib import Path
from tqdm import tqdm
import imgaug.augmenters as iaa
import sys
sys.path.insert(0, str(Path(__file__).parent / 'backend'))
from app.models_service.glcm_extractor import GLCMExtractor


class HybridFishFreshnessTrainer:

    # --- BEGIN: Test set methods ---
    def load_test_data(self, folder_type, use_white_balance=False):
        """
        Loads test data for a given folder_type (eyes_split or gills_split).
        Returns: X_cnn, X_glcm, y
        """
        print(f"\n[LOADER] Loading test data for {folder_type}...")
        resnet_features_list = []
        mobilenet_features_list = []
        glcm_features_list = []
        labels_list = []
        folder_path = self.data_dir / folder_type / 'test'
        if not folder_path.exists():
            print(f"⚠ Test folder not found: {folder_path}")
            return None, None, None
        total_images = 0
        for class_idx, class_name in enumerate(self.FRESHNESS_CLASSES):
            class_path = folder_path / class_name
            if not class_path.exists():
                print(f"⚠ Test class folder not found: {class_path}")
                continue
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')) + list(class_path.glob('*.jpeg'))
            print(f"  {class_name}: {len(image_files)} test images")
            for img_path in tqdm(image_files, desc=f"  {folder_type}-test-{class_name}"):
                try:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        print(f"⚠ Failed to load: {img_path}")
                        continue
                    if folder_type == 'gills_split' and use_white_balance:
                        image = self.apply_white_balance(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # No augmentation for test set
                    processed_resnet = tf.keras.applications.resnet50.preprocess_input(image.astype(np.float32))
                    batch = np.expand_dims(processed_resnet, axis=0)
                    resnet_feat = self.resnet_model.predict(batch, verbose=0)
                    resnet_features_list.append(resnet_feat[0])
                    processed_mobile = tf.keras.applications.mobilenet.preprocess_input(image.astype(np.float32))
                    batch_mobile = np.expand_dims(processed_mobile, axis=0)
                    mobilenet_feat = self.mobilenet_model.predict(batch_mobile, verbose=0)
                    mobilenet_features_list.append(mobilenet_feat[0])
                    glcm_dict = self.glcm_extractor.compute_glcm_summary(image)
                    glcm_feat = self._flatten_glcm_features(glcm_dict)
                    glcm_features_list.append(glcm_feat)
                    labels_list.append(class_idx)
                    total_images += 1
                except Exception as e:
                    print(f"⚠ Error processing {img_path}: {e}")
                    continue
        if not resnet_features_list:
            print(f"✗ No test images loaded for {folder_type}")
            return None, None, None
        X_resnet = np.array(resnet_features_list)
        X_mobilenet = np.array(mobilenet_features_list)
        X_cnn = np.concatenate([X_resnet, X_mobilenet], axis=1)
        X_glcm = np.array(glcm_features_list)
        y = np.array(labels_list)
        print(f"✓ Loaded {total_images} test images for {folder_type}")
        return X_cnn, X_glcm, y

    def evaluate_on_test(self, model, folder_type, use_white_balance=False):
        from sklearn.metrics import classification_report, confusion_matrix
        X_cnn, X_glcm, y = self.load_test_data(folder_type, use_white_balance=use_white_balance)
        if X_cnn is None:
            print(f"No test data for {folder_type}, skipping test evaluation.")
            return
        y_pred = np.argmax(model.predict([X_cnn, X_glcm], verbose=0), axis=1)
        print(f"\n[TEST] {folder_type} - Test Classification Report:")
        print(classification_report(y, y_pred, target_names=self.FRESHNESS_CLASSES, digits=4))
        print("Test Confusion Matrix:")
        print(confusion_matrix(y, y_pred))
    # --- END: Test set methods ---

    FRESHNESS_CLASSES = ['fresh', 'less_fresh', 'starting_to_rot', 'rotten']
    NUM_CLASSES = len(FRESHNESS_CLASSES)
    IMG_SIZE = (224, 224)

    def __init__(self, data_dir='data/processed', output_dir='results'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.resnet_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.resnet_model.trainable = False
        self.mobilenet_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, pooling='avg')
        self.mobilenet_model.trainable = False
        self.glcm_extractor = GLCMExtractor()
        self.augmenter = iaa.Sequential([
            iaa.Fliplr(0.5),  # Horizontal flip
            iaa.Flipud(0.5),  # Vertical flip
            iaa.OneOf([
                iaa.Affine(rotate=(-40, 40)),  # Stronger random rotation
                iaa.Affine(rotate=(-25, 25)),
            ]),
            iaa.CropAndPad(
                percent=(-0.25, 0.1), pad_mode='reflect', pad_cval=(0, 255)
            ),  # More aggressive cropping/padding
            iaa.Multiply((0.7, 1.3)),  # Stronger brightness adjustment
            iaa.LinearContrast((0.7, 1.3)),  # Stronger contrast adjustment
            iaa.AddToHueAndSaturation((-20, 20)),  # Stronger hue & saturation shift
            iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=(0, 0.08*255)),  # Stronger Gaussian noise
                iaa.SaltAndPepper(0.05),  # Salt-and-pepper noise
                iaa.CoarseDropout(0.05, size_percent=0.1),  # Randomly remove patches
            ]),
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0, 2.0)),  # Stronger blur
                iaa.MotionBlur(k=(3, 7)),
                iaa.MedianBlur(k=(3, 5)),
            ]),
            iaa.Affine(scale=(0.7, 1.3)),  # More aggressive scaling (zoom)
            iaa.CLAHE(clip_limit=(1, 6)),  # Stronger adaptive histogram equalization
        ], random_order=True)
        print("✓ Advanced data augmentation pipeline ready")

    def apply_white_balance(self, img):
        """Advanced white balance using LAB Gray World (from old script)"""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    def load_images_and_extract_features(self, folder_type):
        """
        Flexible loader for 'eyes', 'gills', or 'eyes_and_gills'.
        Looks in the 'train' subfolder (e.g., data/processed/gills_split/train/{class})
        Returns: X_cnn, X_glcm, y
        """
        print(f"\n[LOADER] Loading {folder_type} images...")
        resnet_features_list = []
        mobilenet_features_list = []
        glcm_features_list = []
        labels_list = []
        folder_path = self.data_dir / folder_type / 'train'
        if not folder_path.exists():
            print(f"⚠ Folder not found: {folder_path}")
            return None, None, None
        total_images = 0
        for class_idx, class_name in enumerate(self.FRESHNESS_CLASSES):
            class_path = folder_path / class_name
            if not class_path.exists():
                print(f"⚠ Class folder not found: {class_path}")
                continue
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')) + list(class_path.glob('*.jpeg'))
            print(f"  {class_name}: {len(image_files)} images")
            for img_path in tqdm(image_files, desc=f"  {folder_type}-{class_name}"):
                try:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        print(f"⚠ Failed to load: {img_path}")
                        continue
                    # White balance for gills (match 'gills' or 'gills_split')
                    if 'gills' in folder_type:
                        image = self.apply_white_balance(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Data augmentation (50% chance)
                    if np.random.random() > 0.5:
                        image = self.augmenter(image=image)
                    # Extract ResNet50 features
                    processed_resnet = tf.keras.applications.resnet50.preprocess_input(image.astype(np.float32))
                    batch = np.expand_dims(processed_resnet, axis=0)
                    resnet_feat = self.resnet_model.predict(batch, verbose=0)
                    resnet_features_list.append(resnet_feat[0])
                    # Extract MobileNetV1 features
                    processed_mobile = tf.keras.applications.mobilenet.preprocess_input(image.astype(np.float32))
                    batch_mobile = np.expand_dims(processed_mobile, axis=0)
                    mobilenet_feat = self.mobilenet_model.predict(batch_mobile, verbose=0)
                    mobilenet_features_list.append(mobilenet_feat[0])
                    # Extract GLCM features
                    glcm_dict = self.glcm_extractor.compute_glcm_summary(image)
                    glcm_feat = self._flatten_glcm_features(glcm_dict)
                    glcm_features_list.append(glcm_feat)
                    # Add label
                    labels_list.append(class_idx)
                    total_images += 1
                    if total_images % 50 == 0:
                        print(f"    Processed {total_images} images...")
                except Exception as e:
                    print(f"⚠ Error processing {img_path}: {e}")
                    continue
        if not resnet_features_list:
            print(f"✗ No images loaded for {folder_type}")
            return None, None, None
        X_resnet = np.array(resnet_features_list)
        X_mobilenet = np.array(mobilenet_features_list)
        X_cnn = np.concatenate([X_resnet, X_mobilenet], axis=1)
        X_glcm = np.array(glcm_features_list)
        y = np.array(labels_list)
        print(f"✓ Loaded {total_images} images")
        print(f"  ResNet50 features: {X_resnet.shape}")
        print(f"  MobileNetV1 features: {X_mobilenet.shape}")
        print(f"  Combined CNN features: {X_cnn.shape}")
        print(f"  GLCM features: {X_glcm.shape}")
        print(f"  Labels: {y.shape}")
        print(f"  Class distribution: {np.bincount(y)}")
        return X_cnn, X_glcm, y
    def print_confusion_matrix(self, y_true, y_pred):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

    def _flatten_glcm_features(self, glcm_dict):
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

    def extract_features(self, image):
        img_resnet = tf.keras.applications.resnet50.preprocess_input((image * 255).astype(np.float32))
        resnet_feat = self.resnet_model.predict(np.expand_dims(img_resnet, axis=0), verbose=0)[0]
        img_mobilenet = tf.keras.applications.mobilenet.preprocess_input((image * 255).astype(np.float32))
        mobilenet_feat = self.mobilenet_model.predict(np.expand_dims(img_mobilenet, axis=0), verbose=0)[0]
        glcm_dict = self.glcm_extractor.compute_glcm_summary(image)
        glcm_feat = self._flatten_glcm_features(glcm_dict)
        return resnet_feat, mobilenet_feat, glcm_feat

    def load_data(self, split_folder, use_white_balance=False):
        X_resnet, X_mobilenet, X_glcm, y = [], [], [], []
        class_counts = {}
        for class_idx, class_name in enumerate(self.FRESHNESS_CLASSES):
            class_path = self.data_dir / split_folder / 'train' / class_name
            if not class_path.exists():
                print(f"Warning: {class_name} folder missing!")
                continue
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            class_counts[class_name] = len(image_files)
            print(f"  {class_name}: {len(image_files)} images")
            for img_path in tqdm(image_files, desc=f"Loading {class_name}"):
                try:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, self.IMG_SIZE)
                    if use_white_balance:
                        image = self.apply_white_balance(image)
                    if 'train' in str(class_path):
                        if np.random.random() > 0.5:
                            image = self.augmenter(image=image)
                    image = image.astype(np.float32) / 255.0
                    resnet_feat, mobilenet_feat, glcm_feat = self.extract_features(image)
                    X_resnet.append(resnet_feat)
                    X_mobilenet.append(mobilenet_feat)
                    X_glcm.append(glcm_feat)
                    y.append(class_idx)
                except Exception as e:
                    print(f"⚠ Error processing {img_path}: {e}")
                    continue
        if not X_resnet:
            print(f"✗ No images loaded for {split_folder}")
            return None, None, None, None
        X_cnn = np.concatenate([np.array(X_resnet), np.array(X_mobilenet)], axis=1)
        X_glcm = np.array(X_glcm)
        y = np.array(y)
        print(f"✓ Loaded {len(y)} samples")
        print(f"  ResNet50 features: {np.array(X_resnet).shape}")
        print(f"  MobileNetV1 features: {np.array(X_mobilenet).shape}")
        print(f"  Combined CNN features: {X_cnn.shape}")
        print(f"  GLCM features: {X_glcm.shape}")
        print(f"  Labels: {y.shape}")
        print(f"  Class distribution: {np.bincount(y)}")
        return X_cnn, X_glcm, y

    def build_hybrid_model(self, input_cnn_dim=3072, input_glcm_dim=29, num_classes=4):
        cnn_input = Input(shape=(input_cnn_dim,), name='cnn_input')
        x1 = BatchNormalization()(cnn_input)
        x1 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x1)
        x1 = Dropout(0.7)(x1)
        x1 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x1)
        x1 = Dropout(0.6)(x1)
        x1 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x1)
        glcm_input = Input(shape=(input_glcm_dim,), name='glcm_input')
        x2 = BatchNormalization()(glcm_input)
        x2 = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x2)
        x2 = Dropout(0.6)(x2)
        x2 = Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x2)
        merged = Concatenate()([x1, x2])
        merged = BatchNormalization()(merged)
        merged = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(merged)
        merged = Dropout(0.7)(merged)
        merged = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(merged)
        merged = Dropout(0.6)(merged)
        output = Dense(num_classes, activation='softmax', name='output')(merged)
        model = Model(inputs=[cnn_input, glcm_input], outputs=output)
        return model

    def enable_fine_tuning(self):
        print("\n[FINE-TUNE] Unfreezing last layers...")
        # Unfreeze only top layers to prevent overfitting
        for layer in self.resnet_model.layers[-15:]:
            layer.trainable = True
        for layer in self.mobilenet_model.layers[-8:]:
            layer.trainable = True
        print(f"  ResNet50: {sum([1 for l in self.resnet_model.layers if l.trainable])} trainable layers")
        print(f"  MobileNetV1: {sum([1 for l in self.mobilenet_model.layers if l.trainable])} trainable layers")

    def train_and_save_model(self, folder_type, model_save_path, use_white_balance=False, batch_size=16):
        print(f"\n=== Training model for {folder_type} ===")
        X_cnn, X_glcm, y = self.load_images_and_extract_features(folder_type)
        if X_cnn is None:
            print(f"No data for {folder_type}, skipping training.")
            return None, None
        y_cat = to_categorical(y, num_classes=self.NUM_CLASSES)
        X_train_cnn, X_val_cnn, X_train_glcm, X_val_glcm, y_train, y_val = train_test_split(
            X_cnn, X_glcm, y_cat, test_size=0.20, random_state=42, stratify=y
        )
        print(f"Train: {X_train_cnn.shape[0]}, Val: {X_val_cnn.shape[0]}")
        model = self.build_hybrid_model(X_cnn.shape[1], X_glcm.shape[1], self.NUM_CLASSES)
        class_weights = compute_class_weight('balanced', classes=np.arange(self.NUM_CLASSES), y=np.argmax(y_cat, axis=1))
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        print(f"Class weights: {class_weight_dict}")

        # === STAGE 1: Train with frozen feature extractors ===
        print("\n[STAGE 1] Training with frozen feature extractors...")
        self.resnet_model.trainable = False
        self.mobilenet_model.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        callbacks_stage1 = [
            EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
            StopOnLowValLoss(0.05)
        ]
        history1 = model.fit(
            [X_train_cnn, X_train_glcm], y_train,
            validation_data=([X_val_cnn, X_val_glcm], y_val),
            epochs=20,
            batch_size=batch_size,
            callbacks=callbacks_stage1,
            class_weight=class_weight_dict,
            verbose=1
        )

        # === STAGE 2: Fine-tune with unfrozen layers ===
        print("\n[STAGE 2] Fine-tuning with unfrozen layers...")
        self.enable_fine_tuning()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        callbacks_stage2 = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            StopOnLowValLoss(0.05)
        ]
        history2 = model.fit(
            [X_train_cnn, X_train_glcm], y_train,
            validation_data=([X_val_cnn, X_val_glcm], y_val),
            epochs=40,
            batch_size=batch_size,
            callbacks=callbacks_stage2,
            class_weight=class_weight_dict,
            verbose=1
        )

        # Combine histories
        history = history1
        for key in history.history:
            history.history[key].extend(history2.history[key])

        print(f"✓ Model saved to {model_save_path}")
        val_preds = np.argmax(model.predict([X_val_cnn, X_val_glcm], verbose=0), axis=1)
        val_true = np.argmax(y_val, axis=1)
        print("Validation Classification Report:")
        print(classification_report(val_true, val_preds, target_names=self.FRESHNESS_CLASSES, digits=4))
        self.print_confusion_matrix(val_true, val_preds)
        return model, history


if __name__ == '__main__':
    trainer = HybridFishFreshnessTrainer()
    eyes_model_path = 'backend/results/hybrid_eyes_model.h5'
    gills_model_path = 'backend/results/hybrid_gills_model.h5'
    eyes_model, _ = trainer.train_and_save_model('eyes_split', eyes_model_path, use_white_balance=True, batch_size=32)
    gills_model, _ = trainer.train_and_save_model('gills_split', gills_model_path, use_white_balance=True, batch_size=32)
    print("\n✓ Training complete for both eyes and gills models!")

    # --- TEST SET EVALUATION AND VISUALIZATION ---
    print("\n=== EVALUATING ON TEST SETS ===")
    eyes_accuracy = 0.0
    gills_accuracy = 0.0
    
    # Eyes model
    print("\n--- TEST SET: EYES MODEL ---")
    X_cnn_eyes, X_glcm_eyes, y_eyes = trainer.load_test_data('eyes_split', use_white_balance=True)
    if X_cnn_eyes is not None:
        y_pred_eyes = eyes_model.predict([X_cnn_eyes, X_glcm_eyes], verbose=0)
        y_pred_eyes = np.argmax(y_pred_eyes, axis=1)
        eyes_accuracy = np.mean(y_pred_eyes == y_eyes)
        print(f"Eyes Model Test Accuracy: {eyes_accuracy*100:.2f}%")
        plot_confusion_matrix_and_metrics(y_eyes, y_pred_eyes, trainer.FRESHNESS_CLASSES, 'Eyes_Test', 'results')
    else:
        print("No test data for eyes model")
    
    # Gills model
    print("\n--- TEST SET: GILLS MODEL ---")
    X_cnn_gills, X_glcm_gills, y_gills = trainer.load_test_data('gills_split', use_white_balance=True)
    if X_cnn_gills is not None:
        y_pred_gills = gills_model.predict([X_cnn_gills, X_glcm_gills], verbose=0)
        y_pred_gills = np.argmax(y_pred_gills, axis=1)
        gills_accuracy = np.mean(y_pred_gills == y_gills)
        print(f"Gills Model Test Accuracy: {gills_accuracy*100:.2f}%")
        plot_confusion_matrix_and_metrics(y_gills, y_pred_gills, trainer.FRESHNESS_CLASSES, 'Gills_Test', 'results')
    else:
        print("No test data for gills model")
    
    # Create summary visualization
    print("\n--- GENERATING MODEL SUMMARY ---")
    if eyes_accuracy > 0 and gills_accuracy > 0:
        plot_model_summary_accuracy(eyes_accuracy, gills_accuracy, 'results')
        print(f"\n✓ FINAL RESULTS:")
        print(f"  Eyes Test Accuracy:  {eyes_accuracy*100:.2f}%")
        print(f"  Gills Test Accuracy: {gills_accuracy*100:.2f}%")
        print(f"  Average Accuracy:    {(eyes_accuracy + gills_accuracy)/2*100:.2f}%")

