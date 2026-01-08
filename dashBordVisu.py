import streamlit as st
import pandas as pd
import plotly.express as px
import io

# Set page configuration
st.set_page_config(
    page_title="CSV Data Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("📊 CSV Data Visualizer")
st.markdown("Upload a CSV file to instantly generate interactive visualizations")

# Sidebar for file upload and controls
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("### Upload Your Data")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file", 
    type=["csv"],
    help="Upload a CSV file to visualize its data"
)
python -m venv venv
venv\Scripts\activate
# Function to create visualizations
def create_visualizations(df):
    # Create a container for visualizations
    viz_container = st.container()
    
    with viz_container:
        st.header("Data Visualizations")
        
        # Create tabs for different visualization types
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Data Preview", 
            "📊 Bar Chart", 
            "📈 Line Chart", 
            "🔍 Scatter Plot"
        ])
        
        with tab1:
            st.subheader("Raw Data Preview")
            st.dataframe(df)
            st.markdown(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Show data types
            with st.expander("View Data Types"):
                st.write(df.dtypes)
        
        with tab2:
            st.subheader("Bar Chart")
            
            # Let user select columns
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox(
                    "Select X-axis column",
                    df.columns,
                    help="Categorical column for the X-axis"
                )
            
            with col2:
                y_col = st.selectbox(
                    "Select Y-axis column",
                    df.select_dtypes(include=['number']).columns,
                    help="Numerical column for the Y-axis"
                )
            
            # Create bar chart
            if x_col and y_col:
                fig_bar = px.bar(
                    df, 
                    x=x_col, 
                    y=y_col,
                    title=f"{y_col} by {x_col}",
                    color=x_col,
                    template="plotly_white"
                )
                fig_bar.update_layout(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    height=500
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab3:
            st.subheader("Line Chart")
            
            # Let user select columns
            col1, col2 = st.columns(2)
            
            with col1:
                x_col_line = st.selectbox(
                    "Select X-axis column",
                    df.columns,
                    key="line_x",
                    help="Usually a time or sequential column"
                )
            
            with col2:
                y_col_line = st.selectbox(
                    "Select Y-axis column",
                    df.select_dtypes(include=['number']).columns,
                    key="line_y",
                    help="Numerical column for the Y-axis"
                )
            
            # Create line chart
            if x_col_line and y_col_line:
                fig_line = px.line(
                    df, 
                    x=x_col_line, 
                    y=y_col_line,
                    title=f"{y_col_line} over {x_col_line}",
                    markers=True,
                    template="plotly_white"
                )
                fig_line.update_layout(
                    xaxis_title=x_col_line,
                    yaxis_title=y_col_line,
                    height=500
                )
                st.plotly_chart(fig_line, use_container_width=True)
        
        with tab4:
            st.subheader("Scatter Plot")
            
            # Let user select columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_col_scatter = st.selectbox(
                    "Select X-axis column",
                    df.select_dtypes(include=['number']).columns,
                    key="scatter_x",
                    help="Numerical column for the X-axis"
                )
            
            with col2:
                y_col_scatter = st.selectbox(
                    "Select Y-axis column",
                    df.select_dtypes(include=['number']).columns,
                    key="scatter_y",
                    help="Numerical column for the Y-axis"
                )
            
            with col3:
                color_col = st.selectbox(
                    "Color by (optional)",
                    [None] + list(df.columns),
                    key="scatter_color",
                    help="Optional column to color points by"
                )
            
            # Create scatter plot
            if x_col_scatter and y_col_scatter:
                fig_scatter = px.scatter(
                    df, 
                    x=x_col_scatter, 
                    y=y_col_scatter,
                    color=color_col,
                    title=f"{y_col_scatter} vs {x_col_scatter}",
                    template="plotly_white",
                    height=500
                )
                fig_scatter.update_layout(
                    xaxis_title=x_col_scatter,
                    yaxis_title=y_col_scatter
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

# Main app logic
if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display success message
        st.sidebar.success("File uploaded successfully!")
        
        # Show dataset info
        st.sidebar.markdown("### Dataset Info")
        st.sidebar.write(f"**Rows:** {df.shape[0]}")
        st.sidebar.write(f"**Columns:** {df.shape[1]}")
        
        # Create visualizations
        create_visualizations(df)
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Please make sure you're uploading a valid CSV file.")
else:
    # Display instructions when no file is uploaded
    st.markdown("""
    ### How to use this dashboard:
    
    1. **Upload a CSV file** using the file uploader in the sidebar
    2. **Explore your data** through interactive visualizations
    3. **Customize charts** by selecting different columns
    
    ### Supported CSV formats:
    - Comma-separated values (.csv)
    - Files should have headers in the first row
    - Both numerical and categorical data are supported
    
    ### Sample CSV structure:
    ```
    Date,Category,Value,Location
    2023-01-01,Sales,150,New York
    2023-01-02,Marketing,75,Chicago
    2023-01-03,Sales,200,Los Angeles
    ```
    """)
    
    # Display sample visualization
    st.markdown("### Sample Visualization")
    st.info("Upload a CSV file to see your own data visualized like this:")
    
    # Create sample data
    sample_data = {
        'Date': pd.date_range(start='2023-01-01', periods=10),
        'Category': ['Sales', 'Marketing', 'Sales', 'Development', 'Marketing', 
                     'Sales', 'Development', 'Sales', 'Marketing', 'Development'],
        'Value': [150, 75, 200, 120, 90, 180, 110, 220, 85, 130],
        'Location': ['New York', 'Chicago', 'Los Angeles', 'New York', 'Chicago',
                     'Los Angeles', 'New York', 'Chicago', 'Los Angeles', 'New York']
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # Create sample charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_sample1 = px.bar(
            df_sample, 
            x='Category', 
            y='Value',
            color='Location',
            title="Sample Bar Chart",
            template="plotly_white"
        )
        st.plotly_chart(fig_sample1, use_container_width=True)
    
    with col2:
        fig_sample2 = px.line(
            df_sample, 
            x='Date', 
            y='Value',
            color='Category',
            markers=True,
            title="Sample Line Chart",
            template="plotly_white"
        )
        st.plotly_chart(fig_sample2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit and Plotly")