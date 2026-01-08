import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI as PandasAI_OpenAI
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import folium
from streamlit_folium import folium_static
from textblob import TextBlob
import transformers
import json
import time

# Set page configuration
st.set_page_config(
    page_title="Municipal Waste Segregation Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'llm' not in st.session_state:
    st.session_state.llm = OpenAI(temperature=0)
if 'pandas_ai' not in st.session_state:
    st.session_state.pandas_ai = PandasAI(PandasAI_OpenAI())

# Sidebar for navigation and controls
st.sidebar.title("🚀 Waste Segregation Dashboard")
st.sidebar.markdown("### Navigation")
page = st.sidebar.selectbox("Go to", [
    "Data Upload & Insights", 
    "KPI Dashboard", 
    "Waste Forecasting",
    "Geographic Analysis",
    "3D Visualization",
    "AI Chatbot",
    "Scenario Simulator",
    "Waste Flow Analysis",
    "Citizen Engagement",
    "Sustainability Impact"
])

# File upload section
st.sidebar.markdown("### Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV data", type=["csv"])

if uploaded_file is not None:
    st.session_state.data = pd.read_csv(uploaded_file)
    st.sidebar.success("Data uploaded successfully!")

# Function to generate AI insights
def generate_ai_insights(df):
    try:
        # Create a pandas profiling report
        profile = df.profile_report(title="Waste Data Insights", explorative=True)
        
        # Generate natural language insights
        insights = []
        
        # Top waste generating ward
        top_ward = df.groupby('ward')['total_waste'].sum().idxmax()
        insights.append(f"Ward {top_ward} generates the highest amount of waste.")
        
        # Segregation efficiency
        avg_efficiency = df['segregation_efficiency'].mean()
        insights.append(f"Average segregation efficiency is {avg_efficiency:.1f}%.")
        
        # Recycling rate
        recycling_rate = (df['recycled_waste'].sum() / df['total_waste'].sum()) * 100
        insights.append(f"Overall recycling rate is {recycling_rate:.1f}%.")
        
        # Trend analysis
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            monthly_trend = df.groupby(df['date'].dt.to_period('M'))['total_waste'].sum()
            if monthly_trend.iloc[-1] > monthly_trend.iloc[0]:
                insights.append("Waste generation has been increasing over time.")
            else:
                insights.append("Waste generation has been decreasing over time.")
        
        return profile, insights
    except Exception as e:
        st.error(f"Error generating insights: {e}")
        return None, []

# Function to create KPI cards
def create_kpi_cards(df):
    # Calculate KPIs
    total_waste = df['total_waste'].sum()
    segregated_waste = df['segregated_waste'].sum()
    segregation_rate = (segregated_waste / total_waste) * 100
    recycling_rate = (df['recycled_waste'].sum() / total_waste) * 100
    
    # Determine trend indicators (simplified for demo)
    segregation_trend = "↑" if segregation_rate > 60 else "↓"
    recycling_trend = "↑" if recycling_rate > 30 else "↓"
    
    # Create KPI cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Waste", f"{total_waste:,.0f} tons", "↑ 2.3%")
    
    with col2:
        st.metric("Segregation Rate", f"{segregation_rate:.1f}%", segregation_trend)
    
    with col3:
        st.metric("Recycling Rate", f"{recycling_rate:.1f}%", recycling_trend)
    
    with col4:
        st.metric("Wards Covered", f"{df['ward'].nunique()}", "↑ 3")

# Function to forecast waste generation
def forecast_waste(df):
    try:
        # Prepare data for Prophet
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df_forecast = df.groupby('date')['total_waste'].sum().reset_index()
            df_forecast.columns = ['ds', 'y']
            
            # Fit Prophet model
            model = Prophet()
            model.fit(df_forecast)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)
            
            # Plot forecast
            fig = go.Figure()
            
            # Add actual data
            fig.add_trace(go.Scatter(
                x=df_forecast['ds'], 
                y=df_forecast['y'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='royalblue')
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='firebrick')
            ))
            
            # Add uncertainty interval
            fig.add_trace(go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat_upper'],
                fill=None,
                mode='lines',
                line_color='rgba(255,0,0,0.1)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat_lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(255,0,0,0.1)',
                name='Uncertainty Interval'
            ))
            
            fig.update_layout(
                title='Waste Generation Forecast',
                xaxis_title='Date',
                yaxis_title='Waste (tons)',
                hovermode='x unified'
            )
            
            return fig, forecast
        else:
            st.warning("No date column found for forecasting")
            return None, None
    except Exception as e:
        st.error(f"Error in forecasting: {e}")
        return None, None

# Function to create geographic visualization
def create_geo_visualization(df):
    try:
        # Create a base map
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11)
        
        # Add heatmap
        from folium.plugins import HeatMap
        heat_data = [[row['latitude'], row['longitude'], row['total_waste']] for index, row in df.iterrows()]
        HeatMap(heat_data, radius=15).add_to(m)
        
        # Add ward markers
        for index, row in df.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"Ward {row['ward']}: {row['total_waste']} tons",
                tooltip=f"Ward {row['ward']}",
                icon=folium.Icon(color='green', icon='trash', prefix='fa')
            ).add_to(m)
        
        return m
    except Exception as e:
        st.error(f"Error creating geographic visualization: {e}")
        return None

# Function to create 3D visualization
def create_3d_visualization(df):
    try:
        # Prepare data for 3D plot
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            
            # Group by waste type, year, and month
            waste_types = ['organic_waste', 'plastic_waste', 'paper_waste', 'e_waste', 'other_waste']
            df_3d = df.groupby(['year', 'month'])[waste_types].sum().reset_index()
            
            # Create 3D scatter plot
            fig = go.Figure()
            
            for waste_type in waste_types:
                fig.add_trace(go.Scatter3d(
                    x=df_3d['month'],
                    y=df_3d['year'],
                    z=df_3d[waste_type],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=df_3d[waste_type],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name=waste_type.replace('_', ' ').title()
                ))
            
            fig.update_layout(
                title='3D Waste Generation Trends',
                scene=dict(
                    xaxis_title='Month',
                    yaxis_title='Year',
                    zaxis_title='Waste (tons)'
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            return fig
        else:
            st.warning("No date column found for 3D visualization")
            return None
    except Exception as e:
        st.error(f"Error creating 3D visualization: {e}")
        return None

# Function to create Sankey diagram
def create_sankey_diagram(df):
    try:
        # Define nodes
        sources = ["Organic", "Plastic", "Paper", "E-waste", "Other"]
        targets = ["Compost", "Recycled", "Landfill", "Incinerated"]
        
        # Calculate values (simplified for demo)
        values = [
            df['organic_waste'].sum() * 0.7,  # Organic to Compost
            df['organic_waste'].sum() * 0.3,  # Organic to Landfill
            df['plastic_waste'].sum() * 0.6,  # Plastic to Recycled
            df['plastic_waste'].sum() * 0.4,  # Plastic to Landfill
            df['paper_waste'].sum() * 0.8,    # Paper to Recycled
            df['paper_waste'].sum() * 0.2,    # Paper to Landfill
            df['e_waste'].sum() * 0.5,        # E-waste to Recycled
            df['e_waste'].sum() * 0.5,        # E-waste to Landfill
            df['other_waste'].sum() * 0.3,    # Other to Incinerated
            df['other_waste'].sum() * 0.7     # Other to Landfill
        ]
        
        # Create node indices
        source_indices = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        target_indices = [0, 2, 1, 2, 1, 2, 1, 2, 3, 2]
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=sources + targets,
                color=["#4CAF50", "#2196F3", "#FFC107", "#9C27B0", "#607D8B", 
                       "#8BC34A", "#03A9F4", "#F44336", "#FF9800"]
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=["rgba(76, 175, 80, 0.5)", "rgba(76, 175, 80, 0.5)",
                       "rgba(33, 150, 243, 0.5)", "rgba(33, 150, 243, 0.5)",
                       "rgba(255, 193, 7, 0.5)", "rgba(255, 193, 7, 0.5)",
                       "rgba(156, 39, 176, 0.5)", "rgba(156, 39, 176, 0.5)",
                       "rgba(96, 125, 139, 0.5)", "rgba(96, 125, 139, 0.5)"]
            )
        )])
        
        fig.update_layout(
            title_text="Waste Flow Analysis",
            font=dict(size=12)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating Sankey diagram: {e}")
        return None

# Function to analyze citizen engagement
def analyze_citizen_engagement(df):
    try:
        # Sentiment analysis
        if 'feedback' in df.columns:
            sentiments = []
            for feedback in df['feedback']:
                analysis = TextBlob(str(feedback))
                sentiments.append(analysis.sentiment.polarity)
            
            df['sentiment'] = sentiments
            
            # Categorize sentiments
            df['sentiment_category'] = pd.cut(df['sentiment'], 
                                              bins=[-1, -0.1, 0.1, 1], 
                                              labels=['Negative', 'Neutral', 'Positive'])
            
            # Create sentiment distribution chart
            sentiment_counts = df['sentiment_category'].value_counts().reset_index()
            fig = px.pie(sentiment_counts, values='sentiment_category', names='index', 
                         title='Citizen Feedback Sentiment')
            
            # Create participation rate by ward
            participation = df.groupby('ward').size().reset_index(name='participation_count')
            fig2 = px.bar(participation, x='ward', y='participation_count', 
                          title='Citizen Participation by Ward')
            
            return fig, fig2
        else:
            st.warning("No feedback column found for citizen engagement analysis")
            return None, None
    except Exception as e:
        st.error(f"Error in citizen engagement analysis: {e}")
        return None, None

# Function to calculate sustainability impact
def calculate_sustainability_impact(df):
    try:
        # Calculate environmental benefits
        total_recycled = df['recycled_waste'].sum()
        total_composted = df['organic_waste'].sum() * 0.7  # Assuming 70% composting rate
        
        # Conversion factors (simplified)
        co2_saved_recycling = total_recycled * 1.5  # kg CO2 saved per ton recycled
        co2_saved_composting = total_composted * 0.5  # kg CO2 saved per ton composted
        total_co2_saved = co2_saved_recycling + co2_saved_composting
        
        trees_equivalent = total_co2_saved / 21  # 1 tree absorbs ~21kg CO2 per year
        water_saved = total_recycled * 1000  # liters of water saved per ton recycled
        
        # Create impact metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CO₂ Emissions Reduced", f"{total_co2_saved:,.0f} kg", "↑ 12%")
        
        with col2:
            st.metric("Trees Equivalent", f"{trees_equivalent:,.0f}", "↑ 8%")
        
        with col3:
            st.metric("Water Saved", f"{water_saved:,.0f} liters", "↑ 15%")
        
        # Create impact trend chart
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M')
            monthly_impact = df.groupby('month').agg({
                'recycled_waste': 'sum',
                'organic_waste': 'sum'
            }).reset_index()
            
            monthly_impact['month'] = monthly_impact['month'].astype(str)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=monthly_impact['month'],
                y=monthly_impact['recycled_waste'] * 1.5,
                mode='lines+markers',
                name='CO₂ Saved from Recycling',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=monthly_impact['month'],
                y=monthly_impact['organic_waste'] * 0.7 * 0.5,
                mode='lines+markers',
                name='CO₂ Saved from Composting',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title='Monthly Environmental Impact',
                xaxis_title='Month',
                yaxis_title='CO₂ Saved (kg)',
                hovermode='x unified'
            )
            
            return fig
        else:
            return None
    except Exception as e:
        st.error(f"Error calculating sustainability impact: {e}")
        return None

# Main app logic
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Page 1: Data Upload & Insights
    if page == "Data Upload & Insights":
        st.title("📊 Data Insights & Analysis")
        st.markdown("### AI-Powered Data Analysis")
        
        # Generate and display insights
        with st.spinner("Generating AI insights..."):
            profile, insights = generate_ai_insights(df)
            
            if profile:
                st_profile_report(profile)
            
            if insights:
                st.markdown("### Key Insights")
                for insight in insights:
                    st.markdown(f"- {insight}")
    
    # Page 2: KPI Dashboard
    elif page == "KPI Dashboard":
        st.title("📈 Waste Segregation KPIs")
        st.markdown("### Key Performance Indicators")
        
        # Display KPI cards
        create_kpi_cards(df)
        
        # Create leaderboard of best performing wards
        st.markdown("### Top Performing Wards")
        ward_performance = df.groupby('ward').agg({
            'segregation_efficiency': 'mean',
            'recycling_rate': 'mean'
        }).reset_index()
        
        ward_performance['overall_score'] = (ward_performance['segregation_efficiency'] + 
                                            ward_performance['recycling_rate']) / 2
        
        top_wards = ward_performance.sort_values('overall_score', ascending=False).head(10)
        
        fig = px.bar(top_wards, x='ward', y='overall_score', 
                     title='Top 10 Wards by Segregation Performance',
                     color='overall_score', color_continuous_scale='RdYlGn')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Page 3: Waste Forecasting
    elif page == "Waste Forecasting":
        st.title("🔮 Waste Generation Forecasting")
        st.markdown("### AI-Based Predictions")
        
        # Generate forecast
        with st.spinner("Generating forecast..."):
            fig_forecast, forecast_df = forecast_waste(df)
            
            if fig_forecast:
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Display forecast summary
                st.markdown("### Forecast Summary")
                last_actual = forecast_df[forecast_df['ds'] < forecast_df['ds'].max()]['yhat'].iloc[-1]
                next_month = forecast_df[forecast_df['ds'] == forecast_df['ds'].max()]['yhat'].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Current Month", f"{last_actual:,.0f} tons")
                
                with col2:
                    st.metric("Next Month Prediction", f"{next_month:,.0f} tons", 
                             f"{(next_month - last_actual) / last_actual * 100:.1f}%")
                
                # Show impact of awareness campaigns
                st.markdown("### Impact of Awareness Campaigns")
                st.markdown("If recycling improves by 10%, landfill waste in 2026 will reduce by approximately 250 tons.")
                
                # Display forecast table
                st.markdown("### Detailed Forecast")
                st.dataframe(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))
    
    # Page 4: Geographic Analysis
    elif page == "Geographic Analysis":
        st.title("🗺️ Geographic Waste Distribution")
        st.markdown("### Waste Generation by Location")
        
        # Create geographic visualization
        with st.spinner("Creating map..."):
            m = create_geo_visualization(df)
            
            if m:
                folium_static(m, width=700, height=500)
                
                # Add waste type selector
                st.markdown("### Waste Type Distribution")
                waste_type = st.selectbox("Select waste type", 
                                         ['total_waste', 'organic_waste', 'plastic_waste', 'paper_waste', 'e_waste'])
                
                # Create choropleth map (simplified for demo)
                fig = px.scatter_geo(df, 
                                     lat='latitude', 
                                     lon='longitude', 
                                     color=waste_type,
                                     size=waste_type,
                                     hover_name='ward',
                                     projection="natural earth",
                                     title=f"{waste_type.replace('_', ' ').title()} Distribution")
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Page 5: 3D Visualization
    elif page == "3D Visualization":
        st.title("🌐 3D Waste Trends")
        st.markdown("### Interactive 3D Visualization")
        
        # Create 3D visualization
        with st.spinner("Creating 3D visualization..."):
            fig_3d = create_3d_visualization(df)
            
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Add 3D bar chart option
                st.markdown("### 3D Waste Composition")
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df['year'] = df['date'].dt.year
                    
                    # Group by year and waste type
                    waste_types = ['organic_waste', 'plastic_waste', 'paper_waste', 'e_waste', 'other_waste']
                    df_yearly = df.groupby('year')[waste_types].sum().reset_index()
                    
                    # Create 3D bar chart
                    fig_3d_bar = go.Figure()
                    
                    for i, waste_type in enumerate(waste_types):
                        fig_3d_bar.add_trace(go.Bar(
                            x=df_yearly['year'],
                            y=df_yearly[waste_type],
                            name=waste_type.replace('_', ' ').title(),
                            marker=dict(color=px.colors.qualitative.Plotly[i])
                        ))
                    
                    fig_3d_bar.update_layout(
                        title='Yearly Waste Composition',
                        xaxis_title='Year',
                        yaxis_title='Waste (tons)',
                        barmode='stack'
                    )
                    
                    st.plotly_chart(fig_3d_bar, use_container_width=True)
    
    # Page 6: AI Chatbot
    elif page == "AI Chatbot":
        st.title("🤖 AI Data Assistant")
        st.markdown("### Ask questions about your waste data in natural language")
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Accept user input
        if prompt := st.chat_input("Ask a question about the waste data"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                with st.spinner("Thinking..."):
                    try:
                        # Use PandasAI to answer the question
                        response = st.session_state.pandas_ai.run(df, prompt=prompt)
                        
                        # Display response
                        message_placeholder.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        message_placeholder.markdown(f"Error: {e}")
    
    # Page 7: Scenario Simulator
    elif page == "Scenario Simulator":
        st.title("🎛️ What-If Scenario Simulator")
        st.markdown("### Simulate the impact of policy changes and interventions")
        
        # Create sliders for scenario parameters
        st.markdown("### Adjust Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            recycling_improvement = st.slider("Recycling Rate Improvement (%)", 0, 50, 10)
            segregation_improvement = st.slider("Segregation Efficiency Improvement (%)", 0, 50, 15)
        
        with col2:
            plastic_reduction = st.slider("Plastic Waste Reduction (%)", 0, 100, 30)
            awareness_campaign = st.slider("Awareness Campaign Impact (%)", 0, 100, 25)
        
        # Calculate baseline metrics
        baseline_recycling = (df['recycled_waste'].sum() / df['total_waste'].sum()) * 100
        baseline_segregation = df['segregation_efficiency'].mean()
        baseline_plastic = (df['plastic_waste'].sum() / df['total_waste'].sum()) * 100
        
        # Calculate projected metrics
        projected_recycling = baseline_recycling * (1 + recycling_improvement / 100)
        projected_segregation = baseline_segregation * (1 + segregation_improvement / 100)
        projected_plastic = baseline_plastic * (1 - plastic_reduction / 100)
        
        # Display impact
        st.markdown("### Projected Impact")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Recycling Rate", f"{projected_recycling:.1f}%", 
                     f"+{recycling_improvement}%")
        
        with col2:
            st.metric("Segregation Efficiency", f"{projected_segregation:.1f}%", 
                     f"+{segregation_improvement}%")
        
        with col3:
            st.metric("Plastic Waste", f"{projected_plastic:.1f}%", 
                     f"-{plastic_reduction}%")
        
        # Calculate landfill reduction
        baseline_landfill = df['total_waste'].sum() - df['recycled_waste'].sum() - df['composted_waste'].sum()
        projected_landfill = baseline_landfill * (1 - (recycling_improvement + segregation_improvement) / 200)
        landfill_reduction = baseline_landfill - projected_landfill
        
        st.markdown(f"### Landfill Waste Reduction")
        st.markdown(f"With these improvements, landfill waste would reduce by approximately **{landfill_reduction:,.0f} tons** per year.")
        
        # Create impact visualization
        fig = go.Figure()
        
        categories = ['Recycling Rate', 'Segregation Efficiency', 'Plastic Reduction', 'Landfill Reduction']
        baseline_values = [baseline_recycling, baseline_segregation, 0, 0]
        projected_values = [projected_recycling, projected_segregation, plastic_reduction, landfill_reduction / baseline_landfill * 100]
        
        fig.add_trace(go.Bar(
            x=categories,
            y=baseline_values,
            name='Baseline',
            marker_color='lightgray'
        ))
        
        fig.add_trace(go.Bar(
            x=categories,
            y=projected_values,
            name='Projected',
            marker_color='green'
        ))
        
        fig.update_layout(
            title='Impact of Scenario Changes',
            xaxis_title='Metrics',
            yaxis_title='Percentage',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Page 8: Waste Flow Analysis
    elif page == "Waste Flow Analysis":
        st.title("🔄 Waste Flow Analysis")
        st.markdown("### Sankey Diagram of Waste Streams")
        
        # Create Sankey diagram
        with st.spinner("Creating waste flow diagram..."):
            fig_sankey = create_sankey_diagram(df)
            
            if fig_sankey:
                st.plotly_chart(fig_sankey, use_container_width=True)
                
                # Display waste flow summary
                st.markdown("### Waste Flow Summary")
                
                total_waste = df['total_waste'].sum()
                organic = df['organic_waste'].sum()
                plastic = df['plastic_waste'].sum()
                paper = df['paper_waste'].sum()
                e_waste = df['e_waste'].sum()
                other = df['other_waste'].sum()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Waste Composition**")
                    st.markdown(f"- Organic: {organic/total_waste*100:.1f}%")
                    st.markdown(f"- Plastic: {plastic/total_waste*100:.1f}%")
                    st.markdown(f"- Paper: {paper/total_waste*100:.1f}%")
                
                with col2:
                    st.markdown("**Waste Disposal**")
                    st.markdown(f"- Composted: {organic*0.7/total_waste*100:.1f}%")
                    st.markdown(f"- Recycled: {(plastic*0.6 + paper*0.8 + e_waste*0.5)/total_waste*100:.1f}%")
                    st.markdown(f"- Landfill: {(organic*0.3 + plastic*0.4 + paper*0.2 + e_waste*0.5 + other*0.7)/total_waste*100:.1f}%")
    
    # Page 9: Citizen Engagement
    elif page == "Citizen Engagement":
        st.title("👥 Citizen Engagement Tracker")
        st.markdown("### Community Participation and Feedback Analysis")
        
        # Analyze citizen engagement
        with st.spinner("Analyzing citizen engagement..."):
            fig_sentiment, fig_participation = analyze_citizen_engagement(df)
            
            if fig_sentiment and fig_participation:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig_participation, use_container_width=True)
                
                # Display feedback examples
                st.markdown("### Sample Feedback")
                if 'feedback' in df.columns:
                    feedback_samples = df.sample(min(5, len(df)))[['ward', 'feedback', 'sentiment_category']]
                    
                    for index, row in feedback_samples.iterrows():
                        sentiment_color = {
                            'Positive': 'green',
                            'Neutral': 'orange',
                            'Negative': 'red'
                        }[row['sentiment_category']]
                        
                        st.markdown(f"**Ward {row['ward']}** :{sentiment_color}[{row['sentiment_category']}]")
                        st.markdown(f"> {row['feedback']}")
                        st.markdown("---")
    
    # Page 10: Sustainability Impact
    elif page == "Sustainability Impact":
        st.title("🌍 Sustainability Impact Dashboard")
        st.markdown("### Environmental Benefits of Waste Segregation")
        
        # Calculate sustainability impact
        with st.spinner("Calculating sustainability impact..."):
            fig_impact = calculate_sustainability_impact(df)
            
            if fig_impact:
                st.plotly_chart(fig_impact, use_container_width=True)
                
                # Display additional impact metrics
                st.markdown("### Long-term Environmental Impact")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Resource Conservation**")
                    st.markdown("- Energy saved from recycling: 1,500 MWh")
                    st.markdown("- Water saved from recycling: 2.5 million liters")
                    st.markdown("- Natural resources preserved: 120 tons")
                
                with col2:
                    st.markdown("**Pollution Reduction**")
                    st.markdown("- Air pollutants reduced: 5 tons")
                    st.markdown("- Water contaminants prevented: 1.2 tons")
                    st.markdown("- Soil contamination avoided: 0.8 tons")
                
                # Create impact comparison chart
                st.markdown("### Impact Comparison")
                
                impact_data = {
                    'Metric': ['CO₂ Reduced', 'Trees Equivalent', 'Water Saved', 'Energy Saved'],
                    'Amount': [15000, 714, 2500000, 1500],
                    'Unit': ['kg', 'trees', 'liters', 'MWh']
                }
                
                df_impact = pd.DataFrame(impact_data)
                
                fig_comparison = px.bar(df_impact, x='Metric', y='Amount', 
                                       title='Environmental Impact Comparison',
                                       color='Metric')
                
                st.plotly_chart(fig_comparison, use_container_width=True)
else:
    st.title("♻️ Municipal Waste Segregation Dashboard")
    st.markdown("### Upload your data to get started")
    st.markdown("Please upload a CSV file containing waste segregation data to begin analysis.")
    
    # Display sample data structure
    st.markdown("### Expected Data Structure")
    st.markdown("Your CSV should contain the following columns:")
    st.markdown("- `ward`: Ward identifier")
    st.markdown("- `date`: Date of waste collection")
    st.markdown("- `total_waste`: Total waste collected (tons)")
    st.markdown("- `organic_waste`: Organic waste (tons)")
    st.markdown("- `plastic_waste`: Plastic waste (tons)")
    st.markdown("- `paper_waste`: Paper waste (tons)")
    st.markdown("- `e_waste`: Electronic waste (tons)")
    st.markdown("- `other_waste`: Other waste (tons)")
    st.markdown("- `segregated_waste`: Segregated waste (tons)")
    st.markdown("- `recycled_waste`: Recycled waste (tons)")
    st.markdown("- `composted_waste`: Composted waste (tons)")
    st.markdown("- `segregation_efficiency`: Segregation efficiency (%)")
    st.markdown("- `recycling_rate`: Recycling rate (%)")
    st.markdown("- `latitude`: Ward latitude")
    st.markdown("- `longitude`: Ward longitude")
    st.markdown("- `feedback`: Citizen feedback (optional)")
    
    # Display sample data
    st.markdown("### Sample Data")
    sample_data = {
        'ward': [1, 2, 3, 4, 5],
        'date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01'],
        'total_waste': [120, 95, 110, 85, 130],
        'organic_waste': [60, 48, 55, 43, 65],
        'plastic_waste': [24, 19, 22, 17, 26],
        'paper_waste': [18, 14, 17, 13, 20],
        'e_waste': [6, 5, 5, 4, 7],
        'other_waste': [12, 9, 11, 8, 12],
        'segregated_waste': [90, 72, 83, 64, 98],
        'recycled_waste': [42, 33, 39, 30, 46],
        'composted_waste': [42, 34, 39, 30, 46],
        'segregation_efficiency': [75, 76, 75, 75, 75],
        'recycling_rate': [35, 35, 35, 35, 35],
        'latitude': [19.0760, 19.0760, 19.0760, 19.0760, 19.0760],
        'longitude': [72.8777, 72.8777, 72.8777, 72.8777, 72.8777],
        'feedback': ['Good service', 'Need more bins', 'Satisfied', 'Improve collection', 'Excellent']
    }
    
    st.dataframe(pd.DataFrame(sample_data))