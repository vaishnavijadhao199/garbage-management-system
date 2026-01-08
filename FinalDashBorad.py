from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def dashboard():
    return render_template('DashBoard.html')

@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        df = None
        
        # Handle CSV files
        if file.filename.endswith('.csv'):
            # Try multiple encodings without chardet
            for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'ascii']:
                try:
                    df = pd.read_csv(filepath, encoding=enc, on_bad_lines='skip')
                    print(f"✓ Successfully read CSV with encoding: {enc}")
                    break
                except Exception as e:
                    print(f"✗ Failed with {enc}: {str(e)}")
                    continue
            
            if df is None:
                return jsonify({'error': 'Cannot read CSV file. Try saving as UTF-8 in Excel'}), 400
        
        # Handle Excel files
        elif file.filename.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(filepath, sheet_name=0)
                print("✓ Successfully read Excel file")
            except Exception as e:
                return jsonify({'error': f'Cannot read Excel file: {str(e)}'}), 400
        
        else:
            return jsonify({'error': 'Invalid format. Upload CSV or Excel file'}), 400
        
        if df is None or df.empty:
            return jsonify({'error': 'File is empty'}), 400
        
        print("\n📋 Original columns:", df.columns.tolist())
        
        # Clean column names - CRITICAL STEP
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '').str.replace('-', '')
        
        print("📋 Cleaned columns:", df.columns.tolist())
        
        # Column mapping - handles all variations
        column_mapping = {
            'wastetype': 'type',
            'waste_type': 'type',
            'type': 'type',
            'waste': 'type',
            'quantity': 'tons',
            'qty': 'tons',
            'wastons': 'tons',
            'tons': 'tons',
            'amount': 'tons',
            'date': 'wastedate',
            'waste_date': 'wastedate',
            'wastedate': 'wastedate',
            'collection_date': 'wastedate',
            'collectiondate': 'wastedate',
            'location': 'area',
            'area': 'area',
            'district': 'area',
            'zone': 'area',
            'recycling_percent': 'recyclepercent',
            'recyclingpercent': 'recyclepercent',
            'recyclepercent': 'recyclepercent',
            'recycle_percent': 'recyclepercent',
            'recyclepercent': 'recyclepercent',
            'recycling_rate': 'recyclepercent',
            'recyclerate': 'recyclepercent'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        print("📋 After mapping:", df.columns.tolist())
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Validate required columns
        required_cols = ['type', 'tons', 'wastedate', 'area', 'recyclepercent']
        available = df.columns.tolist()
        missing = [col for col in required_cols if col not in available]
        
        if missing:
            print(f"❌ Missing columns: {missing}")
            print(f"✓ Available: {available}")
            return jsonify({
                'error': f'Missing required columns: {missing}. Available: {available}',
                'available_columns': available,
                'required_columns': required_cols
            }), 400
        
        # Select only required columns
        df = df[required_cols].copy()
        
        print(f"✓ Selected columns: {df.columns.tolist()}")
        print(f"✓ Data shape: {df.shape}")
        
        # Convert data types with error handling
        try:
            df['tons'] = pd.to_numeric(df['tons'], errors='coerce')
            df['recyclepercent'] = pd.to_numeric(df['recyclepercent'], errors='coerce')
            df['wastedate'] = pd.to_datetime(df['wastedate'], errors='coerce')
            df['type'] = df['type'].astype(str).str.strip()
            df['area'] = df['area'].astype(str).str.strip()
        except Exception as e:
            print(f"❌ Error converting data types: {str(e)}")
            return jsonify({'error': f'Error converting data types: {str(e)}'}), 400
        
        # Remove rows with invalid/null data
        initial_rows = len(df)
        df = df.dropna()
        removed_rows = initial_rows - len(df)
        
        if removed_rows > 0:
            print(f"⚠️  Removed {removed_rows} rows with invalid data")
        
        if df.empty:
            return jsonify({'error': 'No valid data after cleaning. Check your CSV format'}), 400
        
        print(f"✓ Processing {len(df)} valid rows")
        
        charts = generate_charts(df)
        
        # Add data preview to response
        charts['data_preview'] = {
            'rows': len(df),
            'sample': df.head(3).to_dict('records')
        }
        
        return jsonify(charts)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Exception: {error_trace}")
        return jsonify({
            'error': f'Error: {str(e)}',
            'trace': error_trace
        }), 500

def generate_charts(df):
    try:
        # Pie chart
        waste_fig = px.pie(
            df, names='type', values='tons',
            title='Waste Type Distribution', hole=0.4
        )
        waste_fig.update_layout(height=500, showlegend=True)
        
        # Line chart
        trend_fig = px.line(
            df, x='wastedate', y='tons', color='area',
            title='Waste Collection Trends',
            labels={'tons': 'Tons Collected', 'wastedate': 'Date'}
        )
        trend_fig.update_layout(height=500)
        
        # Bar chart - Waste by Area
        area_waste = df.groupby('area')['tons'].sum().reset_index()
        bar_fig = px.bar(
            area_waste, x='area', y='tons',
            title='Total Waste by Area', 
            color='area',
            labels={'tons': 'Total Tons', 'area': 'Area'}
        )
        bar_fig.update_layout(height=500)
        
        # Histogram
        hist_fig = px.histogram(
            df, x='tons', nbins=20,
            title='Waste Distribution',
            labels={'tons': 'Waste Collected (Tons)'}
        )
        hist_fig.update_layout(height=500)
        
        # Recycling efficiency
        recycling_data = df.groupby('area')['recyclepercent'].mean().reset_index()
        recycling_fig = px.bar(
            recycling_data, x='area', y='recyclepercent',
            title='Recycling Efficiency by Area',
            labels={'recyclepercent': 'Recycling Rate (%)', 'area': 'Area'},
            color='recyclepercent'
        )
        recycling_fig.update_layout(height=500)
        
        return {
            'waste_chart': waste_fig.to_json(),
            'trend_chart': trend_fig.to_json(),
            'district_chart': bar_fig.to_json(),
            'recycling_chart': recycling_fig.to_json(),
            'bar_chart': bar_fig.to_json(),
            'histogram_chart': hist_fig.to_json(),
            'summary': generate_summary(df)
        }
    except Exception as e:
        print(f"❌ Chart generation error: {str(e)}")
        import traceback
        return {'error': f'Chart generation error: {str(e)}', 'trace': traceback.format_exc()}

def generate_summary(df):
    try:
        return {
            'total_waste': f"{df['tons'].sum():,.2f} tons",
            'avg_recycling': f"{df['recyclepercent'].mean():.1f}%",
            'top_area': str(df.groupby('area')['tons'].sum().idxmax()),
            'most_common_type': str(df['type'].value_counts().idxmax())
        }
    except Exception as e:
        print(f"❌ Summary generation error: {str(e)}")
        return {
            'total_waste': 'N/A',
            'avg_recycling': 'N/A',
            'top_area': 'N/A',
            'most_common_type': 'N/A'
        }

if __name__ == '__main__':
    print("🚀 Starting Flask app on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)