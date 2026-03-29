# Fish Freshness Detection

A hybrid deep learning system that detects fish freshness by analyzing eyes and gills using separate CNN models trained from scratch.

## Features

-  **Dual Detection System**: Separate models for eyes and gills analysis
-  **Hybrid Approach**: Combines predictions from both models for accurate freshness classification
-  **4-Class Classification**: Fresh, Less Fresh, Starting to Rot, Rotten
-  **Pre-trained Models**: Ready-to-use trained models included via Git LFS
-  **Image-Based**: Works with both camera and uploaded images

## Prerequisites

### Git LFS (Large File Storage)

Since this project uses Git LFS to store large model files (>100MB each), you need to install it:

**Windows:**
```bash
choco install git-lfs
```
or download from https://git-lfs.github.com

**Mac:**
```bash
brew install git-lfs
```

**Linux:**
```bash
sudo apt-get install git-lfs
```

Then initialize LFS:
```bash
git lfs install
```

### Python & Dependencies

- Python 3.8+
- pip/virtualenv

## Installation

1. **Clone the repository with Git LFS:**
```bash
git lfs install
git clone https://github.com/VinceSevilla/grouperfish-freshness-detector.git
cd grouperfish-freshness-detector
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
```

3. **Activate the environment:**
```bash
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

4. **Install dependencies:**
```bash
# Backend
pip install -r backend/requirements.txt

# Frontend (if running locally)
cd frontend
npm install
```

## Project Structure

```
.
├── backend/                    # Flask API & models
│   ├── app/
│   │   ├── detection/         # Eye and gill detectors
│   │   ├── models_service/    # Model loading & inference
│   │   └── main.py            # Flask app
│   └── results/               # Trained models (LFS)
│       ├── hybrid_eyes_model.h5
│       └── hybrid_gills_model.h5
├── frontend/                   # React/TypeScript UI
│   ├── src/
│   │   ├── pages/             # UI pages
│   │   └── components/        # React components
│   └── package.json
├── data/                       # Training data
│   ├── raw/                   # Original images
│   └── processed/             # Split datasets
└── train_from_scratch.py       # Training script
```

## Usage

### Backend API

```bash
cd backend
python -m app.main
```

API will run on `http://localhost:5000`

### Frontend (Development)

```bash
cd frontend
npm run dev
```

Frontend will run on `http://localhost:5173`

### Training New Models

```bash
python train_from_scratch.py
```

This will:
- Load and preprocess data from `data/processed/`
- Train separate models for eyes and gills
- Save trained models to `backend/results/`

## Model Information

### Eyes Model
- **Architecture**: CNN trained from scratch
- **Input**: Eye region images (224x224)
- **Output**: 4-class predictions (Fresh, Less Fresh, Starting to Rot, Rotten)
- **File**: `hybrid_eyes_model.h5` (~132 MB)

### Gills Model
- **Architecture**: CNN trained from scratch
- **Input**: Gill region images (224x224)
- **Output**: 4-class predictions
- **File**: `hybrid_gills_model.h5` (~132 MB)

### Hybrid System
- Combines predictions from both models
- Uses weighted averaging for final classification

## Dataset Structure

```
data/processed/
├── eyes_split/
│   ├── train/ ├── fresh, less_fresh, rotten, starting_to_rot
│   ├── val/   └── fresh, less_fresh, rotten, starting_to_rot
│   └── test/  └── fresh, less_fresh, rotten, starting_to_rot
└── gills_split/
    ├── train/ ├── fresh, less_fresh, rotten, starting_to_rot
    ├── val/   └── fresh, less_fresh, rotten, starting_to_rot
    └── test/  └── fresh, less_fresh, rotten, starting_to_rot
```

## Troubleshooting

### Models not downloading?
Make sure Git LFS is installed and initialized:
```bash
git lfs install
git lfs pull
```

### LFS files showing as pointers?
This means Git LFS didn't download the actual files. Try:
```bash
git lfs pull --include="*.h5"
```

## Contributing

Feel free to fork, modify, and improve! For major changes, please open an issue first.

## License

MIT License - feel free to use this project for your own purposes.

---

**Questions?** Open an issue on GitHub or contact the maintainers.
