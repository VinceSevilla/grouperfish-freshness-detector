print("VERY TOP OF FILE")

"""
Train fish freshness detection models using ResNet50 + GLCM features
Data: data/processed/{eyes,gills,eyes_and_gills}/{fresh,less_fresh,starting_to_rot,rotten}/
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import cv2
from datetime import datetime
import imgaug.augmenters as iaa
from tqdm import tqdm
import warnings

# Import GLCM extractor from backend
sys.path.insert(0, str(Path(__file__).parent / 'backend'))
from app.models_service.glcm_extractor import GLCMExtractor


class FishFreshnessTrainer:
    """Trains fish freshness models with ResNet50 + GLCM features"""
    
    FRESHNESS_CLASSES = ['fresh', 'less_fresh', 'starting_to_rot', 'rotten']
    CLASS_NAMES = ['Fresh', 'Less Fresh', 'Starting to Rot', 'Rotten']
    
    def __init__(self, data_dir='data/processed', output_dir='results'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        # Load ResNet50 for feature extraction (2048 features)
        self.resnet_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        self.resnet_model.trainable = False  # Freeze weights
        print("✓ Loaded ResNet50 (frozen weights)")
        # Load MobileNetV1 for feature extraction (1024 features)
        self.mobilenet_model = tf.keras.applications.MobileNet(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        self.mobilenet_model.trainable = False  # Initially frozen
        print("✓ Loaded MobileNetV1 (initially frozen)")
        # Data augmentation pipeline
        self.augmenter = iaa.Sequential([
            iaa.Fliplr(0.5),  # Horizontal flip 50%
            iaa.Flipud(0.2),  # Vertical flip 20%
            iaa.Affine(
                rotate=(-20, 20),  # Rotate -20 to +20 degrees
                scale=(0.9, 1.1)   # Scale 90% to 110%
            ),
            iaa.Multiply((0.8, 1.2)),  # Brightness adjustment
            iaa.GaussianBlur(sigma=(0, 1.0))  # Slight blur
        ], random_order=True)
        print("✓ Data augmentation pipeline ready")

    def apply_white_balance(self, img):
        """Simple white balance using Gray World Assumption"""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    
    def load_images_and_extract_features(self, folder_type):
        """
        Load images from processed folder and extract features
        
        Args:
            folder_type: 'eyes', 'gills', or 'eyes_and_gills'
        
        Returns:
            X_cnn: Combined CNN features (N, 3072) - ResNet50 (2048) + MobileNet (1024)
            X_glcm: GLCM features (N, 29)
            y: Class labels (N,) with values 0-3
        """
        print(f"\n[LOADER] Loading {folder_type} images...")
        
        resnet_features_list = []
        mobilenet_features_list = []
        glcm_features_list = []
        labels_list = []
        
        folder_path = self.data_dir / folder_type
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
                    # Load image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        print(f"⚠ Failed to load: {img_path}")
                        continue
                    # White balance correction for both eyes and gills
                    if folder_type in ['gills', 'eyes']:
                        image = self.apply_white_balance(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Apply data augmentation (50% chance)
                    if np.random.random() > 0.5:
                        image = self.augmenter(image=image)
                    # Extract ResNet50 features (2048-dim)
                    processed_resnet = preprocess_input(image.astype(np.float32))
                    batch = np.expand_dims(processed_resnet, axis=0)
                    resnet_feat = self.resnet_model.predict(batch, verbose=0)
                    resnet_features_list.append(resnet_feat[0])
                    # Extract MobileNetV1 features (1024-dim)
                    processed_mobile = mobilenet_preprocess(image.astype(np.float32))
                    batch_mobile = np.expand_dims(processed_mobile, axis=0)
                    mobilenet_feat = self.mobilenet_model.predict(batch_mobile, verbose=0)
                    mobilenet_features_list.append(mobilenet_feat[0])
                    # Extract GLCM features
                    glcm_dict = GLCMExtractor.compute_glcm_summary(image)
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

        # Concatenate ResNet50 (2048) + MobileNet (1024) = 3072 features
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
    
    def _flatten_glcm_features(self, glcm_dict):
        """Flatten GLCM dict into 29-feature vector"""
        features = []
        
        # 6 basic features
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
        
        # 18 multi-scale features (3 scales × 6 properties)
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
        
        # 5 directional variance features
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
        
        return np.array(features, dtype=np.float32)
    
    def enable_fine_tuning(self):
        """Unfreeze last layers of both models for fine-tuning"""
        print("\n[FINE-TUNE] Unfreezing last layers...")
        
        # Unfreeze last 30 layers of ResNet50 (has ~175 layers)
        for layer in self.resnet_model.layers[-30:]:
            layer.trainable = True
        trainable_resnet = sum([1 for layer in self.resnet_model.layers if layer.trainable])
        print(f"  ResNet50: {trainable_resnet} trainable layers")
        
        # Unfreeze last 15 layers of MobileNetV1 (has ~88 layers)
        for layer in self.mobilenet_model.layers[-15:]:
            layer.trainable = True
        trainable_mobile = sum([1 for layer in self.mobilenet_model.layers if layer.trainable])
        print(f"  MobileNetV1: {trainable_mobile} trainable layers")
    
    def build_model(self, input_cnn_dim=3072, input_glcm_dim=29, num_classes=4):
        """Build hybrid model with ResNet50+MobileNet CNN features + GLCM inputs"""
        
        # CNN input branch (ResNet50 2048 + MobileNetV1 1024 = 3072)
        cnn_input = layers.Input(shape=(input_cnn_dim,), name='cnn_input')
        x1 = layers.BatchNormalization()(cnn_input)
        x1 = layers.Dense(512, activation='relu')(x1)
        x1 = layers.Dropout(0.4)(x1)
        x1 = layers.Dense(256, activation='relu')(x1)
        x1 = layers.Dropout(0.3)(x1)
        x1 = layers.Dense(128, activation='relu')(x1)
        
        # GLCM input branch
        glcm_input = layers.Input(shape=(input_glcm_dim,), name='glcm_input')
        x2 = layers.BatchNormalization()(glcm_input)
        x2 = layers.Dense(64, activation='relu')(x2)
        x2 = layers.Dropout(0.3)(x2)
        x2 = layers.Dense(32, activation='relu')(x2)
        
        # Concatenate branches
        merged = layers.Concatenate()([x1, x2])
        merged = layers.BatchNormalization()(merged)
        merged = layers.Dense(128, activation='relu')(merged)
        merged = layers.Dropout(0.4)(merged)
        merged = layers.Dense(64, activation='relu')(merged)
        merged = layers.Dropout(0.3)(merged)
        
        # Output layer
        output = layers.Dense(num_classes, activation='softmax', name='output')(merged)
        
        model = models.Model(inputs=[cnn_input, glcm_input], outputs=output)
        
        # Don't compile here - will compile in train_model with proper learning rate
        return model
    
    def train_model(self, X_cnn, X_glcm, y, model_name, epochs=100, batch_size=16):
        """Train a model with two-stage training: frozen then fine-tuned"""
        print(f"\n[TRAIN] Training {model_name}...")
        
        # Split data
        X_cnn_train, X_cnn_test, X_g_train, X_g_test, y_train, y_test = train_test_split(
            X_cnn, X_glcm, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Train size: {len(X_cnn_train)}, Test size: {len(X_cnn_test)}")
        print(f"  Train class distribution: {np.bincount(y_train)}")
        print(f"  Test class distribution: {np.bincount(y_test)}")
        
        # Calculate class weights to handle imbalance
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"  Class weights: {class_weight_dict}")
        
        # Build model
        model = self.build_model()
        
        # STAGE 1: Train with frozen feature extractors (warmup)
        print("\n[STAGE 1] Training with frozen feature extractors...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for Stage 1
        model_path = self.output_dir / f'best_model_{model_name}.h5'
        checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        early_stop1 = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr1 = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train Stage 1
        history1 = model.fit(
            [X_cnn_train, X_g_train], y_train,
            validation_data=([X_cnn_test, X_g_test], y_test),
            epochs=30,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=[checkpoint1, early_stop1, reduce_lr1],
            verbose=1
        )
        
        # STAGE 2: Fine-tune with unfrozen layers
        print("\n[STAGE 2] Fine-tuning with unfrozen layers...")
        self.enable_fine_tuning()
        
        # Recompile with lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for Stage 2
        checkpoint2 = tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        early_stop2 = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr2 = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train Stage 2
        history2 = model.fit(
            [X_cnn_train, X_g_train], y_train,
            validation_data=([X_cnn_test, X_g_test], y_test),
            epochs=70,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=[checkpoint2, early_stop2, reduce_lr2],
            verbose=1
        )
        
        # Combine histories
        history = history1
        for key in history.history:
            history.history[key].extend(history2.history[key])
        
        # Evaluate
        loss, accuracy = model.evaluate([X_cnn_test, X_g_test], y_test, verbose=0)
        print(f"\n✓ {model_name} - Test Accuracy: {accuracy*100:.2f}%, Loss: {loss:.4f}")
        
        # Predictions
        y_pred = np.argmax(model.predict([X_cnn_test, X_g_test], verbose=0), axis=1)
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.CLASS_NAMES))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:\n{cm}")
        
        return model, history, accuracy
    
    def train_all(self):
        """Train all three models and object detectors"""
        results = {}

        # Train Eyes Model (commented out to only train gills)
        # print(f"\n{'='*60}")
        # print(f"Training EYES Model")
        # print(f"{'='*60}")
        # X_cnn, X_glcm, y = self.load_images_and_extract_features('eyes')
        # if X_cnn is not None:
        #     model, history, accuracy = self.train_model(
        #         X_cnn, X_glcm, y,
        #         model_name='eyes',
        #         epochs=100,
        #         batch_size=16
        #     )
        #     results['eyes'] = {'model': model, 'accuracy': accuracy, 'history': history}
        # else:
        #     print(f"✗ Skipping eyes - no data loaded")

        # Train Gills Model (with white balance correction)
        print(f"\n{'='*60}")
        print(f"Training GILLS Model (with white balance)")
        print(f"{'='*60}")
        X_cnn, X_glcm, y = self.load_images_and_extract_features('gills')
        if X_cnn is not None:
            model, history, accuracy = self.train_model(
                X_cnn, X_glcm, y,
                model_name='gills',
                epochs=100,
                batch_size=16
            )
            results['gills'] = {'model': model, 'accuracy': accuracy, 'history': history}
        else:
            print(f"✗ Skipping gills - no data loaded")

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        for folder_type, data in results.items():
            print(f"{folder_type}: Accuracy = {data['accuracy']*100:.2f}%")
        print(f"\n✓ Models saved to {self.output_dir}")

    def train_object_detector(self, part):
        """
        Train an object detection model for gills or eyes using TensorFlow Object Detection API.
        This function assumes you have annotated your dataset with bounding boxes for each freshness class:
        'fresh', 'less_fresh', 'starting_to_rot', 'rotten'.
        
        Steps:
        1. Prepare your images and annotations (in Pascal VOC or COCO format) for each class.
        2. Convert annotations to TFRecord format (TensorFlow OD API requirement).
        3. Create a label map file (e.g., part_label_map.pbtxt) with all freshness classes.
        4. Prepare a pipeline config file for your chosen model (e.g., SSD, Faster R-CNN).
        5. Run this function to start training.
        """
        import os
        import subprocess

        # Paths (update as needed)
        pipeline_config_path = f"object_detection/configs/{part}_pipeline.config"
        model_dir = f"object_detection/models/{part}_detector"
        train_record = f"object_detection/data/{part}_train.record"
        val_record = f"object_detection/data/{part}_val.record"
        label_map = f"object_detection/data/{part}_label_map.pbtxt"

        # Check required files
        for path in [pipeline_config_path, train_record, val_record, label_map]:
            if not os.path.exists(path):
                print(f"Missing required file: {path}")
                return

        # Run TensorFlow Object Detection training
        command = [
            "python", "object_detection/model_main_tf2.py",
            "--model_dir", model_dir,
            "--pipeline_config_path", pipeline_config_path,
            "--num_train_steps=10000",
            "--sample_1_of_n_eval_examples=1",
            "--alsologtostderr"
        ]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command)
        print(f"Training for {part} object detector complete.")

        # After training, export the model for inference as needed.
if __name__ == "__main__":
    print("[MAIN BLOCK] Starting Fish Freshness Training Script...")
    trainer = FishFreshnessTrainer()
    trainer.train_all()
