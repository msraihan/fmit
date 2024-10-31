# pages/visualize_report.py
import streamlit as st
from databricks import sql
import os
from utils.config import load_config
from utils.helpers import (
    perform_clustering_analysis, get_recommended_clusters,
    plot_elbow_curve, plot_silhouette_scores,
    perform_final_clustering, create_tsne_plot,
    display_cluster_summary
)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import plotly.express as px
from utils.llm_helper import process_cluster_themes
import json
from datetime import datetime
# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Initialize session states
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'clustering_performed' not in st.session_state:
    st.session_state.clustering_performed = False
if 'cluster_results' not in st.session_state:
    st.session_state.cluster_results = None

def fetch_data(engine_name, failcode, source_table):
    """Fetch data from Databricks SQL"""
    try:
        with sql.connect(
            server_hostname=st.secrets["DATABRICKS_SERVER_HOSTNAME"],
            http_path=st.secrets["DATABRICKS_HTTP_PATH"],
            access_token=st.secrets["DATABRICKS_TOKEN"],
            warehouse_id=DATABRICKS_WAREHOUSE_ID
        ) as connection:
            query = f"""
            SELECT *
            FROM {source_table}
            WHERE engine_name_desc = '{engine_name}'
            AND failcode = '{failcode}'
            """
            
            with connection.cursor() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                result = cursor.fetchall()
                
            df = pd.DataFrame(result, columns=columns)
            return df
            
    except Exception as e:
        st.error(f"Error connecting to Databricks: {str(e)}")
        return None

def create_tsne_plot(embeddings, cluster_labels, n_clusters):
    """Create TSNE visualization of clusters using Plotly"""
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
        
        #  cluster points
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
        
        # cluster centers
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

# create_tsne_plot
def display_cluster_themes(cluster_themes):
    """Display themes for each cluster"""
    st.subheader("Cluster Themes Analysis")
    
    for cluster, themes in cluster_themes.items():
        with st.expander(f"Cluster {cluster} Themes", expanded=True):
            st.markdown(themes)
            st.markdown("---")

def save_themes_to_file(cluster_themes, filename_prefix):
    """Save themes to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_themes_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(cluster_themes, f, indent=4)
    
    return filename

def display_theme_comparison(comparison_df):
    """Display theme comparison table"""
    st.subheader("Theme Comparison Across Clusters")
    
    # Style the dataframe
    styled_df = comparison_df.style.set_properties(**{
        'background-color': '#f0f2f6',
        'color': 'black',
        'border-color': 'white'
    })
    
    st.dataframe(styled_df)

def main():
    st.title("Failure Mode Analysis Dashboard")
    
    # Load configuration
    config = load_config()
    SOURCE_TABLE = st.secrets["SOURCE_TABLE"]
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("Analysis Parameters")
        
        selected_engine = st.selectbox(
            'Select Engine Name Description:',
            options=config['ENGINE_NAME_DESC']
        )
        
        selected_failcode = st.selectbox(
            'Select Fail Code:',
            options=config['FAIL_CODE']
        )
        
        # Add token limit selector
        token_limit = st.slider(
            "Token Limit for Analysis:",
            min_value=10000,
            max_value=128000,
            value=32000,
            step=1000,
            help="Maximum number of tokens to process for each cluster"
        )
        
        analyze_clicked = st.button('Analyze', type='primary')
    
    # Main content area
    if analyze_clicked:
        st.write(f"Analyzing data for Engine: {selected_engine} and Fail Code: {selected_failcode}")
        
        with st.spinner('Fetching data...'):
            df = fetch_data(selected_engine, selected_failcode, SOURCE_TABLE)
            
        if df is not None and not df.empty:
            st.success("Data retrieved successfully!")
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.clustering_performed = False
            
            # Display basic data info
            st.write(f"Total records: {len(df)}")
            
            # Display sample data without sensitive columns
            display_df = df.drop(columns=[
                'embedding_json', 'summary_paragraph', 'llm_cause', 
                'embedding_cause', 'embedding_paragraph'
            ], errors='ignore')
            st.write("Sample of the data:")
            st.dataframe(display_df.head())
        else:
            st.warning("No data found for the selected parameters.")
            st.session_state.data_loaded = False
    
    # Only show embedding selection if data is loaded
    if st.session_state.data_loaded:
        st.subheader("Clustering Analysis")
        
        embedding_type = st.radio(
            "Select embedding type for clustering:",
            ["embedding_cause", "embedding_paragraph"],
            key="embedding_selection"
        )
        
        start_clustering = st.button("Start Clustering Analysis")
        
        if start_clustering and embedding_type in st.session_state.df.columns:
            try:
                st.session_state.embeddings = np.array(st.session_state.df[embedding_type].apply(eval).tolist())
                
                with st.spinner('Performing clustering analysis...'):
                    K, distortions, silhouette_scores = perform_clustering_analysis(st.session_state.embeddings)
                    st.session_state.clustering_performed = True
                    st.session_state.K = K
                    st.session_state.distortions = distortions
                    st.session_state.silhouette_scores = silhouette_scores
                    
                    recommendations = get_recommended_clusters(distortions, silhouette_scores)
                    st.session_state.recommendations = recommendations
                
            except Exception as e:
                st.error(f"Error processing embeddings: {str(e)}")
        
        if st.session_state.clustering_performed:
            st.subheader("Cluster Recommendations")
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            with rec_col1:
                st.metric("Elbow Method", st.session_state.recommendations['elbow_method'])
            with rec_col2:
                st.metric("Silhouette Method", st.session_state.recommendations['silhouette_method'])
            with rec_col3:
                st.metric("Final Recommendation", st.session_state.recommendations['final_recommendation'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(plot_elbow_curve(st.session_state.K, st.session_state.distortions))
            
            with col2:
                st.plotly_chart(plot_silhouette_scores(st.session_state.K, st.session_state.silhouette_scores))
            
            n_clusters = st.slider(
                "Select number of clusters for analysis:",
                min_value=2,
                max_value=10,
                value=st.session_state.recommendations['final_recommendation'],
                key="n_clusters"
            )
            
# Modify the "Apply Clustering" button section in main():
            if st.button("Apply Clustering"):
                with st.spinner('Applying clustering...'):
                    cluster_labels = perform_final_clustering(st.session_state.embeddings, n_clusters)
                    
                    # Display cluster summary
                    st.subheader("Cluster Summary")
                    summary_df = display_cluster_summary(st.session_state.df, cluster_labels, n_clusters)
                    
                    st.dataframe(
                        summary_df.style.background_gradient(subset=['Size'], cmap='YlOrRd')
                        .format({'Percentage': '{}'})
                    )
                    
                    # Create DataFrame with clusters
                    df_with_clusters = st.session_state.df.copy()
                    df_with_clusters['Cluster'] = cluster_labels
                    
                    # Generate t-SNE visualization
                    st.subheader("Cluster Visualization")
                    with st.spinner('Generating t-SNE visualization...'):
                        tsne_fig, df_vis = create_tsne_plot(st.session_state.embeddings, cluster_labels, n_clusters)
                        st.plotly_chart(tsne_fig)
                    
                    # Process themes for each cluster
                    with st.spinner('Analyzing themes for each cluster...'):
                        selected_column = 'summary_paragraph' if embedding_type == 'embedding_paragraph' else 'llm_cause'
                        cluster_themes = process_cluster_themes(df_with_clusters, selected_column, token_limit)
                        display_cluster_themes(cluster_themes)
                    
                    # Prepare data for export
                    columns_to_exclude = [
                        'embedding_json',
                        'embedding_cause',
                        'embedding_paragraph'
                    ]
                    
                    export_df = df_with_clusters.drop(columns=[col for col in columns_to_exclude if col in df_with_clusters.columns])
                    
                    # Add download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="Download detailed results as CSV",
                            data=csv,
                            file_name="cluster_results.csv",
                            mime="text/csv",
                        )
                    
                    with col2:
                        vis_csv = df_vis.to_csv(index=False)
                        st.download_button(
                            label="Download visualization data as CSV",
                            data=vis_csv,
                            file_name="cluster_visualization.csv",
                            mime="text/csv",
                        )

if __name__ == "__main__":
    DATABRICKS_WAREHOUSE_ID = st.secrets["DATABRICKS_WAREHOUSE_ID"]
    main()