import os
import time
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt

app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_data(file_path):
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['CustomerID'])
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['TotalAmount'] = data['Quantity'] * data['UnitPrice']
    return data

def preprocess_data(data):
    customer_summary = data.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum',
        'InvoiceDate': ['max', 'min']
    }).reset_index()
    
    customer_summary.columns = ['CustomerID', 'PurchaseFrequency', 'TotalAmount', 'LastPurchaseDate', 'FirstPurchaseDate']
    current_date = dt.datetime(2011, 12, 10)
    customer_summary['Recency'] = (current_date - customer_summary['LastPurchaseDate']).dt.days
    customer_summary['AveragePurchaseAmount'] = customer_summary['TotalAmount'] / customer_summary['PurchaseFrequency']
    return customer_summary

def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def perform_kmeans(customer_summary, scaled_features, n_clusters=5):
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    customer_summary['Cluster_KMeans'] = kmeans.fit_predict(scaled_features)
    kmeans_score = silhouette_score(scaled_features, customer_summary['Cluster_KMeans'])
    kmeans_score_percentage = f"{kmeans_score * 100:.2f}%"
    runtime = time.time() - start_time
    print(f'KMeans Silhouette Score: {kmeans_score_percentage}, Runtime: {runtime:.2f} seconds')
    return customer_summary, kmeans_score_percentage, runtime

def plot_kmeans(customer_summary, n_clusters=5):
    fig_kmeans = go.Figure()
    colors = px.colors.qualitative.Bold
    
    for cluster in range(n_clusters):
        cluster_data = customer_summary[customer_summary['Cluster_KMeans'] == cluster]
        fig_kmeans.add_trace(go.Scatter(
            x=cluster_data['PurchaseFrequency'],
            y=cluster_data['TotalAmount'],
            mode='markers',
            marker=dict(color=colors[cluster]),
            name=f'Cluster {cluster}',
            hoverinfo='text',
            text=[
                f'CustomerID: {cid}<br>Recency: {recency}<br>Avg Purchase Amount: {avg_amt}'
                for cid, recency, avg_amt in zip(cluster_data['CustomerID'], cluster_data['Recency'], cluster_data['AveragePurchaseAmount'])
            ]
        ))
    
    fig_kmeans.update_layout(
        title='Customer Segmentation based on Purchase Frequency and Total Amount (KMeans)',
        xaxis_title='Purchase Frequency',
        yaxis_title='Total Amount Spent',
        updatemenus=[
            {
                'buttons': [
                    {
                        'method': 'update',
                        'label': f'Cluster {cluster}',
                        'args': [{'visible': [i == cluster for i in range(n_clusters)]}]
                    }
                    for cluster in range(n_clusters)
                ] + [
                    {
                        'method': 'update',
                        'label': 'All Clusters',
                        'args': [{'visible': [True] * n_clusters}]
                    }
                ],
                'direction': 'down',
                'showactive': True
            }
        ]
    )
    
    return fig_kmeans.to_html()

def perform_dbscan(customer_summary, pca_features):
    start_time = time.time()
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    customer_summary['Cluster_DBSCAN'] = dbscan.fit_predict(pca_features)
    
    if len(set(customer_summary['Cluster_DBSCAN'])) > 1:
        dbscan_score = silhouette_score(pca_features, customer_summary['Cluster_DBSCAN'])
        dbscan_score_percentage = f"{dbscan_score * 100:.2f}%"
    else:
        dbscan_score_percentage = 'Only one cluster found. Silhouette Score is not applicable.'
    
    runtime = time.time() - start_time
    print(f'DBSCAN Silhouette Score: {dbscan_score_percentage}, Runtime: {runtime:.2f} seconds')
    return customer_summary, dbscan_score_percentage, runtime

def plot_dbscan(customer_summary, pca_features):
    color_sequence = ['#636EFA', '#EF553B', '#00CC96', '#FFA15A']
    
    fig_dbscan = px.scatter(
        customer_summary,
        x=pca_features[:, 0],
        y=pca_features[:, 1],
        color='Cluster_DBSCAN',
        color_discrete_sequence=color_sequence,
        title='Customer Segmentation with DBSCAN after PCA',
        labels={'x': 'PCA Feature 1', 'y': 'PCA Feature 2'},
        hover_data={'CustomerID': True, 'Recency': True, 'AveragePurchaseAmount': True}
    )
    
    fig_dbscan.update_layout(title_text='Customer Segmentation with DBSCAN after PCA', title_x=0.5)
    return fig_dbscan.to_html()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            data = load_data(file_path)

            # Get column descriptions
            column_descriptions = data.dtypes.apply(lambda x: x.name).reset_index()
            column_descriptions.columns = ['Column Name', 'Data Type']
            column_descriptions_html = column_descriptions.to_html(index=False)

            customer_summary = preprocess_data(data)
            
            features = customer_summary[['PurchaseFrequency', 'TotalAmount', 'Recency', 'AveragePurchaseAmount']]
            scaled_features = normalize_features(features)
            
            # KMeans Clustering and Plotting
            customer_summary, kmeans_score, kmeans_runtime = perform_kmeans(customer_summary, scaled_features)
            kmeans_plot = plot_kmeans(customer_summary)
            
            # DBSCAN Clustering and Plotting
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(scaled_features)
            customer_summary, dbscan_score, dbscan_runtime = perform_dbscan(customer_summary, pca_features)
            dbscan_plot = plot_dbscan(customer_summary, pca_features)
            
            return render_template('index.html', column_descriptions=column_descriptions_html, kmeans_plot=kmeans_plot, kmeans_score=kmeans_score, kmeans_runtime=kmeans_runtime, dbscan_plot=dbscan_plot, dbscan_score=dbscan_score, dbscan_runtime=dbscan_runtime)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

