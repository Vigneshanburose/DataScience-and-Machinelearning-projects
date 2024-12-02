import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import io
import requests

@st.cache
def load_data_from_github():
    url = 'https://github.com/Vigneshanburose/DataScience-and-Machinelearning-projects/blob/main/Airbnb_Newyorkcity_DataAnalysis/listings.zip' 
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    
    # Extract the contents to a temporary folder and read the CSV
    zip_file.extractall("/tmp/dataset")  
    df = pd.read_csv('/tmp/dataset/airbnb_listings.csv')  
    return df

@st.cache_data
def load_data():
    """Load and preprocess the Airbnb dataset"""
    df = load_data_from_github()
    
    # Basic preprocessing
    df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Convert categorical columns
    categorical_cols = ['neighbourhood_group', 'room_type']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    return df

def main():
    st.title("New York City Airbnb Listings Analysis")
    
    # Add dataset statistics at the top
    df = load_data()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Listings", len(df))
    with col2:
        st.metric("Avg Price", f"${df['price'].mean():.2f}")
    with col3:
        st.metric("Median Price", f"${df['price'].median():.2f}")
    with col4:
        st.metric("Unique Neighborhoods", df['neighbourhood'].nunique())
    
    # Sidebar for filters
    st.sidebar.header("Visualization Controls")
    
    # Neighborhood Group Filter
    selected_neighborhoods = st.sidebar.multiselect(
        "Select Neighborhood Groups", 
        df['neighbourhood_group'].unique(),
        default=df['neighbourhood_group'].unique()
    )
    
    # Room Type Filter
    selected_room_types = st.sidebar.multiselect(
        "Select Room Types", 
        df['room_type'].unique(),
        default=df['room_type'].unique()
    )
    
    # Price Range Slider
    min_price = int(df['price'].min())
    max_price = int(df['price'].max())
    price_range = st.sidebar.slider(
        "Price Range", 
        min_value=min_price, 
        max_value=max_price, 
        value=(min_price, max_price)
    )
    
    # Filter dataframe
    filtered_df = df[
        (df['neighbourhood_group'].isin(selected_neighborhoods)) &
        (df['room_type'].isin(selected_room_types)) &
        (df['price'].between(price_range[0], price_range[1]))
    ]
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Price Distribution", 
        "Neighborhood Insights", 
        "Room Type Analysis", 
        "Geographical Visualization",
        "Availability Analysis"
    ])
    
    with tab1:
        # Price Distribution by Neighborhood Group
        st.header("Price Distribution")
        fig1 = px.box(
            filtered_df, 
            x='neighbourhood_group', 
            y='price', 
            title='Price Distribution Across Neighborhood Groups'
        )
        st.plotly_chart(fig1)
    
    with tab2:
        # Average Price and Number of Listings by Neighborhood
        st.header("Neighborhood Insights")
        neighborhood_summary = filtered_df.groupby('neighbourhood_group').agg({
            'price': 'mean',
            'id': 'count'
        }).rename(columns={'id': 'number_of_listings'}).reset_index()
        
        fig2 = px.bar(
            neighborhood_summary, 
            x='neighbourhood_group', 
            y='price', 
            color='number_of_listings',
            title='Average Price and Number of Listings by Neighborhood Group'
        )
        st.plotly_chart(fig2)
    
    with tab3:
        # Room Type Analysis
        st.header("Room Type Insights")
        room_summary = filtered_df.groupby('room_type').agg({
            'price': ['mean', 'count']
        }).reset_index()
        room_summary.columns = ['room_type', 'avg_price', 'number_of_listings']
        
        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.pie(
                room_summary, 
                values='number_of_listings', 
                names='room_type', 
                title='Distribution of Room Types'
            )
            st.plotly_chart(fig3)
        
        with col2:
            fig4 = px.bar(
                room_summary, 
                x='room_type', 
                y='avg_price', 
                title='Average Price by Room Type'
            )
            st.plotly_chart(fig4)
    
    with tab4:
        # Geographical Visualization
        st.header("Geographical Price Heatmap")
        fig5 = px.scatter_mapbox(
            filtered_df, 
            lat='latitude', 
            lon='longitude', 
            color='price',
            hover_name='neighbourhood',
            zoom=10, 
            height=600,
            title='NYC Airbnb Listings Price Heatmap'
        )
        fig5.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig5)
    
    with tab5:
        # Availability Analysis
        st.header("Listing Availability Insights")
        availability_summary = filtered_df.groupby('neighbourhood_group')[
            ['availability_365', 'availability_30', 'availability_60']
        ].mean().reset_index()
        
        fig6 = px.bar(
            availability_summary, 
            x='neighbourhood_group', 
            y=['availability_365', 'availability_30', 'availability_60'],
            title='Average Availability by Neighborhood Group'
        )
        st.plotly_chart(fig6)

if __name__ == "__main__":
    main()
