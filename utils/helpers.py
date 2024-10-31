# utils/helpers.py
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

def perform_clustering_analysis(matrix, max_clusters=10):
    """Perform clustering analysis and return metrics"""
    distortions = []
    silhouette_scores = []
    K = range(2, max_clusters + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
        kmeans.fit(matrix)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(matrix, kmeans.labels_))

    return K, distortions, silhouette_scores

def get_recommended_clusters(distortions, silhouette_scores):
    """Determine recommended number of clusters"""
    deltas = np.diff(distortions)
    delta_changes = np.diff(deltas)
    elbow_point = np.argmin(delta_changes) + 2
    
    max_silhouette_idx = np.argmax(silhouette_scores)
    silhouette_point = max_silhouette_idx + 2
    
    return {
        'elbow_method': elbow_point,
        'silhouette_method': silhouette_point,
        'final_recommendation': round((elbow_point + silhouette_point) / 2)
    }

def plot_elbow_curve(K, distortions):
    """Create elbow curve plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(K), y=distortions, mode='lines+markers'))
    fig.update_layout(
        title='Elbow Method showing the optimal number of clusters',
        xaxis_title='Number of clusters',
        yaxis_title='Distortion',
        showlegend=False
    )
    return fig

def plot_silhouette_scores(K, silhouette_scores):
    """Create silhouette scores plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(K), y=silhouette_scores, mode='lines+markers'))
    fig.update_layout(
        title='Silhouette Scores for different number of clusters',
        xaxis_title='Number of clusters',
        yaxis_title='Silhouette Score',
        showlegend=False
    )
    return fig

def perform_final_clustering(matrix, n_clusters):
    """Perform final clustering"""
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    cluster_labels = kmeans.fit_predict(matrix)
    return cluster_labels

def create_tsne_plot(embeddings, cluster_labels, n_clusters):
    """Create TSNE visualization"""
    tsne = TSNE(
        n_components=2,
        perplexity=15,
        random_state=42,
        init="random",
        learning_rate=200
    )
    
    vis_dims = tsne.fit_transform(embeddings)
    
    df_vis = pd.DataFrame({
        'x': vis_dims[:, 0],
        'y': vis_dims[:, 1],
        'Cluster': cluster_labels
    })
    
    cluster_centers = df_vis.groupby('Cluster')[['x', 'y']].mean().reset_index()
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set3[:n_clusters]
    
    for cluster in range(n_clusters):
        cluster_data = df_vis[df_vis['Cluster'] == cluster]
        
        fig.add_trace(go.Scatter(
            x=cluster_data['x'],
            y=cluster_data['y'],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(
                size=8,
                color=colors[cluster],
                opacity=0.6
            ),
            hovertext=[f'Cluster {cluster}' for _ in range(len(cluster_data))]
        ))
        
        center = cluster_centers[cluster_centers['Cluster'] == cluster]
        fig.add_trace(go.Scatter(
            x=center['x'],
            y=center['y'],
            mode='markers',
            name=f'Center {cluster}',
            marker=dict(
                symbol='x',
                size=15,
                color=colors[cluster],
                line=dict(width=2)
            ),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Clusters Visualization (t-SNE 2D projection)",
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        hovermode='closest',
        width=800,
        height=600
    )
    
    return fig, df_vis

def display_cluster_summary(df, cluster_labels, n_clusters):
    """Display cluster summary statistics"""
    summary_data = []
    
    for cluster in range(n_clusters):
        cluster_data = df[cluster_labels == cluster]
        summary = {
            'Cluster': cluster,
            'Size': len(cluster_data),
            'Percentage': f"{(len(cluster_data) / len(df) * 100):.1f}%"
        }
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)