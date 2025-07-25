# 🎵 Splitify - Intelligent Music Clustering

Splitify is an advanced music analysis and clustering tool that automatically groups your songs based on their audio characteristics using machine learning. It extracts audio features from MP3 files and creates intelligent playlists by clustering similar songs together.

## ✨ Features

- **🎯 Intelligent Clustering**: Automatically determines the optimal number of clusters using elbow method and silhouette analysis
- **🔊 Audio Feature Extraction**: Analyzes rhythm, energy, and spectral characteristics using Essentia
- **📊 Visualization**: Creates beautiful scatter plots and analysis charts
- **📄 JSON Export**: Saves clustering results in structured JSON format
- **📈 Performance Analytics**: Detailed clustering metrics and optimization

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone or download the project**
```bash
git clone
cd splitify
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your music files**
   - Create a `songs/` folder in the project directory
   - Add your MP3 files to the `songs/` folder

### Usage

#### 🎵 Basic Song Clustering

Run the main clustering script:
```bash
python cluster_songs.py
```

This will:
- Extract audio features from all MP3 files in the `songs/` folder
- Automatically determine the optimal number of clusters
- Generate clustering results in `song_clusters.json`
- Create visualization plots (`song_clusters.png`, `optimal_k_analysis.png`)
```

#### 🔍 Analyze Individual Songs

Analyze a specific song's features:
```bash
python app.py
```

## 📊 Output Files

### `song_clusters.json`
Contains complete clustering results with metadata:
```json
{
  "metadata": {
    "total_songs": 70,
    "num_clusters": 3,
    "algorithm": "K-Means with optimal k determination",
    "features_used": ["rhythm.bpm", "rhythm.danceability", ...]
  },
  "clusters": {
    "cluster_1": {
      "songs": ["Song1.mp3", "Song2.mp3", ...]
    }
  }
}
```

### Visualization Files
- `song_clusters.png`: 2D visualization of clustered songs
- `optimal_k_analysis.png`: Elbow method and silhouette analysis charts
- `features_cache.pkl`: Cached audio features (auto-generated)

## 🎼 Audio Features Used

Splitify analyzes these audio characteristics:

- **🥁 Rhythm Features**
  - BPM (Beats Per Minute)
  - Danceability score

- **⚡ Energy Features**
  - Spectral energy


## 🛠️ Configuration

### Modify Features
Edit `extract_features_from_song()` in `cluster_songs.py` to include more features:

```python
# Add more features
if 'lowlevel.spectral_centroid.mean' in features.descriptorNames():
    feature_vector.append(features['lowlevel.spectral_centroid.mean'])
if 'tonal.key_edma.strength' in features.descriptorNames():
    feature_vector.append(features['tonal.key_edma.strength'])
```

### Force Specific Number of Clusters
```python
# In cluster_songs.py, change:
cluster_labels, song_files = cluster_songs(songs_folder, k=4)  # Force 4 clusters
```

## 📁 Project Structure

```
splitify/
├── songs/                         # Place your MP3 files here
├── cluster_songs.py               # Main clustering script
├── app.py                         # Individual song analysis
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── song_clusters.json             # Generated clustering results
├── song_clusters.png              # Generated visualization
├── optimal_k_analysis.png         # Generated k-analysis
```

## 📄 License

This project is open source. Feel free to use and modify for personal or educational purposes.

## 🙏 Acknowledgments

- **Essentia**: Excellent audio analysis library
- **scikit-learn**: Machine learning clustering algorithms
- **matplotlib**: Beautiful visualization capabilities
---

*Transform your music collection into intelligently organized playlists with Splitify.*
