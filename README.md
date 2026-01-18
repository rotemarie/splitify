# Splitify - AI Music Source Separation

Splitify is an end-to-end system that separates music into 4 stems: **vocals**, **bass**, **drums**, and **other** (melody/instruments) using deep learning.

## 🎵 Features

- **Multi-stem separation**: Separate any song into 4 distinct tracks
- **End-to-end pipeline**: From raw audio to separated stems
- **Flexible training**: Train on MUSDB18 or custom datasets
- **CPU/GPU support**: Works on both CPU and GPU
- **Multiple formats**: Supports MP3, WAV, FLAC, and more

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Splitify

# Install dependencies
pip install -r requirements.txt
```

### 2. Download a Pre-trained Model (Coming Soon)
```bash
# We'll provide pre-trained models in the future
# For now, you'll need to train your own model
```

### 3. Separate Your Music

```bash
python inference.py \
    --model_path checkpoints/your_model.pt \
    --input "path/to/your/song.mp3" \
    --output_dir "separated_stems/"
```

This will create 4 files:
- `song_vocals.wav` - Isolated vocals
- `song_bass.wav` - Bass line
- `song_drums.wav` - Drum track  
- `song_other.wav` - Other instruments/melody

## 🏋️ Training Your Own Model

### 1. Prepare MUSDB18 Dataset

```bash
# Download MUSDB18 dataset to musdb18/ folder
# Update config.json with your paths

# Convert to HDF5 format for fast training
python preprocessing/audio_to_hdf5.py --config config.json

# Create training indexes
python preprocessing/index.py \
    --workspace /path/to/Splitify \
    --config_yaml preprocessing/configs/sr=44100,vocals-bass-drums-other.yaml \
    --split train
```

### 2. Train the Model

```bash
cd learning/multistem

python train.py \
    --workspace ../../checkpoints \
    --index_pkl ../../indexes/musdb18/train/sr=44100,vocals-bass-drums-other.pkl \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3
```

### 3. Evaluate Your Model

```bash
python evaluate_multistem.py \
    --model_path ../../checkpoints/checkpoint_epoch50.pt \
    --input_audio "test_song.wav" \
    --output_dir "results/"
```

## 🖥️ CPU vs GPU Training

### GPU Training (Recommended)
- **Training time**: ~6-12 hours for 50 epochs on RTX 3080
- **Memory needed**: 8GB+ VRAM
- **Batch size**: 8-16

### CPU Training (Possible but Slow)
- **Training time**: ~3-7 days for 50 epochs
- **Memory needed**: 16GB+ RAM
- **Batch size**: 2-4 (reduce to avoid memory issues)
- **Recommendation**: Use smaller model (`base_channels=32` instead of 64)

To train on CPU:
```bash
python train.py \
    --workspace ../../checkpoints \
    --index_pkl ../../indexes/musdb18/train/sr=44100,vocals-bass-drums-other.pkl \
    --epochs 50 \
    --batch_size 2 \
    --lr 1e-3
```

## 📁 Project Structure

```
Splitify/
├── inference.py              # Main inference script
├── config.json              # Dataset configuration
├── requirements.txt         # Python dependencies
├── preprocessing/           # Data preparation
│   ├── audio_to_hdf5.py    # Convert MUSDB18 to HDF5
│   ├── index.py            # Create training indexes
│   └── configs/            # Configuration files
├── learning/
│   ├── multistem/          # Multi-stem separation
│   │   ├── model.py        # U-Net architecture
│   │   ├── train.py        # Training script
│   │   ├── dataset.py      # Data loading
│   │   └── stft_utils.py   # STFT/iSTFT utilities
│   └── singlestem/         # Single-stem separation
├── musdb18/                # MUSDB18 dataset (you provide)
├── hdf5s/                  # Processed HDF5 files
├── indexes/                # Training indexes
└── checkpoints/            # Saved models
```

## 🎯 Model Architecture

The system uses a **U-Net** architecture that:

1. **Input**: Takes STFT magnitude spectrogram of mixed audio
2. **Processing**: Uses encoder-decoder with skip connections
3. **Output**: Predicts 4 masks (one per stem) 
4. **Reconstruction**: Applies masks to original spectrogram and converts back to audio

## 🔧 Customization

### Change Stems
Edit the stems in your training script:
```python
stems = ["vocals", "piano", "guitar", "drums"]  # Custom stems
```

### Adjust Model Size
For faster training or CPU use:
```python
model = UNetMulti(in_channels=1, n_outputs=4, base_channels=32)  # Smaller model
```

### Different Audio Parameters
Modify STFT parameters in training:
```python
n_fft = 1024        # Smaller = faster, less frequency resolution
hop_length = 256    # Smaller = more time resolution, slower
```

## 📊 Expected Results

With proper training on MUSDB18:
- **Vocals**: Clean separation, minimal bleeding
- **Drums**: Good transient preservation
- **Bass**: Clear low-frequency separation  
- **Other**: Mixed results depending on complexity

## 🐛 Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size 2`
- Use smaller model: `base_channels=32`
- Process shorter segments

### Poor Separation Quality
- Train longer: `--epochs 100`
- Use larger model: `base_channels=128`
- Ensure good quality training data
- Check your STFT parameters

### Slow Training
- Use GPU if available
- Increase batch size if memory allows
- Use mixed precision training (advanced)

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines]

## 📚 References

- MUSDB18 Dataset
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- Open-Unmix: A Reference Implementation for Music Source Separation
