import os
import numpy as np
import json
from essentia.standard import MusicExtractor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def extract_features_from_song(file_path):
    """Extract audio features from a single song."""
    try:
        extractor = MusicExtractor(lowlevelStats=['mean', 'stdev'],rhythmStats=['mean', 'stdev'])
        
        features, _ = extractor(file_path)
        
        # Select key features for clustering
        feature_vector = []
        
        # Rhythm features
        if 'rhythm.bpm' in features.descriptorNames():
            feature_vector.append(features['rhythm.bpm'])
        if 'rhythm.danceability' in features.descriptorNames():
            feature_vector.append(features['rhythm.danceability'])
        # Energy features
        if 'lowlevel.spectral_energy.mean' in features.descriptorNames():
            feature_vector.append(features['lowlevel.spectral_energy.mean'])

        return np.array(feature_vector)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def find_optimal_k(features_scaled, max_k=5):
    """Find optimal number of clusters using elbow method and silhouette analysis."""
    print("\n🔍 Determining optimal number of clusters...")
    
    # Limit max_k based on data size (rule of thumb: max k = sqrt(n/2))
    n_samples = len(features_scaled)
    max_k = min(max_k, max(2, int(np.sqrt(n_samples / 2))))
    
    if n_samples < 4:
        print("⚠️  Too few samples for clustering analysis, using k=2")
        return 2
    
    k_range = range(2, max_k + 1)
    inertias = []
    silhouette_scores = []
    
    print(f"📊 Testing k values from 2 to {max_k}...")
    
    for k in k_range:
        # Fit K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette_avg)
        
        print(f"  k={k}: Inertia={inertia:.2f}, Silhouette={silhouette_avg:.3f}")
    
    # Find elbow point using the "knee" method
    # Calculate the rate of decrease in inertia
    rate_of_decrease = []
    for i in range(1, len(inertias)):
        rate_of_decrease.append(inertias[i-1] - inertias[i])
    
    # Find the point where the rate of decrease drops significantly
    elbow_k = 2  # default
    if len(rate_of_decrease) > 1:
        # Find the biggest drop in the rate of decrease
        second_derivative = []
        for i in range(1, len(rate_of_decrease)):
            second_derivative.append(rate_of_decrease[i-1] - rate_of_decrease[i])
        
        if second_derivative:
            elbow_idx = np.argmax(second_derivative)
            elbow_k = k_range[elbow_idx + 1]
    
    # Find k with highest silhouette score
    best_silhouette_k = k_range[np.argmax(silhouette_scores)]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot elbow curve
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=elbow_k, color='r', linestyle='--', alpha=0.7, label=f'Elbow at k={elbow_k}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot silhouette scores
    ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=best_silhouette_k, color='r', linestyle='--', alpha=0.7, label=f'Best k={best_silhouette_k}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Average Silhouette Score')
    ax2.set_title('Silhouette Analysis for Optimal k')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('optimal_k_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Optimal k analysis saved as 'optimal_k_analysis.png'")
    plt.show()
    
    # Choose the final k value
    # Prefer elbow method, but consider silhouette if the difference is significant
    if abs(elbow_k - best_silhouette_k) <= 1:
        optimal_k = elbow_k
        method = "Elbow method"
    else:
        # If silhouette score for elbow_k is reasonable, use it; otherwise use silhouette best
        elbow_k_idx = list(k_range).index(elbow_k)
        silhouette_k_idx = list(k_range).index(best_silhouette_k)
        
        if silhouette_scores[elbow_k_idx] >= 0.3:  # Reasonable silhouette score threshold
            optimal_k = elbow_k
            method = "Elbow method"
        else:
            optimal_k = best_silhouette_k
            method = "Silhouette analysis"
    
    print(f"\n🎯 Optimal k = {optimal_k} (determined by {method})")
    print(f"   Elbow method suggests: k = {elbow_k}")
    print(f"   Best silhouette score: k = {best_silhouette_k} (score: {max(silhouette_scores):.3f})")
    
    return optimal_k

def cluster_songs(songs_folder, k=None):
    """Cluster songs based on their audio features."""
    print("🔍 Extracting features from songs...")
    song_files = []
    feature_matrix = []
    
    # Get all MP3 files in the folder
    mp3_files = [f for f in os.listdir(songs_folder) if f.endswith('.mp3')]
    print(f"Found {len(mp3_files)} MP3 files")
    
    # Process all audio files in the folder
    for i, filename in enumerate(mp3_files, 1):
        file_path = os.path.join(songs_folder, filename)
        print(f"[{i}/{len(mp3_files)}] Processing: {filename}")
        
        features = extract_features_from_song(file_path)
        if features is not None and len(features) > 0:
            song_files.append(filename)
            feature_matrix.append(features)
        else:
            print(f"  ❌ Skipping {filename} - could not extract features")
    
    if len(feature_matrix) == 0:
        print("❌ No valid features extracted!")
        return None, None
    
    # Convert to numpy array and save to cache
    feature_matrix = np.array(feature_matrix)
    
    print(f"\n📊 Using features for {len(song_files)} songs")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)
    
    # Determine optimal k if not provided
    if k is None:
        k = find_optimal_k(features_scaled)
    else:
        print(f"\n🎯 Using specified k = {k}")
    
    # Perform K-means clustering
    print(f"\n🎯 Clustering songs into {k} groups...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Print clustering results
    print("\n" + "="*60)
    print("CLUSTERING RESULTS")
    print("="*60)
    
    # Prepare data for JSON export
    clustering_results = {
        "metadata": {
            "total_songs": len(song_files),
            "num_clusters": k,
            "feature_vector_size": feature_matrix.shape[1],
            "algorithm": "K-Means with optimal k determination",
            "k_determination_method": "Elbow method + Silhouette analysis",
            "features_used": [
                "rhythm.bpm", "rhythm.danceability", "lowlevel.spectral_energy.mean"
            ]
        },
        "clusters": {}
    }
    
    for cluster_id in range(k):
        songs_in_cluster = [song_files[i] for i in range(len(song_files)) if cluster_labels[i] == cluster_id]
        
        # Store in results dictionary
        clustering_results["clusters"][f"cluster_{cluster_id + 1}"] = {
            "cluster_id": cluster_id + 1,
            "song_count": len(songs_in_cluster),
            "songs": songs_in_cluster
        }
        
        # Print to console
        print(f"\nCluster {cluster_id + 1} ({len(songs_in_cluster)} songs):")
        print("-" * 40)
        for song in songs_in_cluster:
            print(f"  • {song}")
    
    # Save clustering results to JSON file
    json_filename = "song_clusters.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(clustering_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Clustering results saved to '{json_filename}'")
    print(f"🎵 Successfully clustered {len(song_files)} songs into {k} groups")
    
    # Visualize clusters using PCA
    print("\n📊 Creating visualization...")
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)
    
    plt.figure(figsize=(12, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for cluster_id in range(k):
        cluster_points = features_2d[cluster_labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[cluster_id % len(colors)], 
                   label=f'Cluster {cluster_id + 1}', 
                   alpha=0.7, s=100)
    
    # Plot cluster centers
    cluster_centers_2d = pca.transform(kmeans.cluster_centers_)
    plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], 
               c='black', marker='x', s=200, linewidths=3, label='Centroids')
    
    plt.title('Song Clustering Visualization (PCA projection)', fontsize=16)
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('song_clusters.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Visualization saved as 'song_clusters.png'")
    plt.show()
    
    return cluster_labels, song_files

if __name__ == "__main__":
    songs_folder = "songs"
    # Let the algorithm determine optimal k automatically, if k is not specified
    cluster_labels, song_files = cluster_songs(songs_folder, k=None)
