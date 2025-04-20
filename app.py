import dash
from flask import Flask
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import base64
import io
import joblib




# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "RecruitEase"

# Try to load the trained model, feature names and label encoders
try:
    model = joblib.load("attrition_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    # Load feature names exactly as they were during training
    feature_names = joblib.load("feature_names.pkl")
    model_loaded = True
    print("Model and encoders loaded successfully")
    print("Feature names from model:", feature_names)
except Exception as e:
    model_loaded = False
    feature_names = []
    print(f"Error loading model: {e}")

# Dropdown Options
DROPDOWN_OPTIONS = {
    'JobSatisfaction': [
        {'label': 'Low', 'value': '1'},
        {'label': 'Medium', 'value': '2'},
        {'label': 'High', 'value': '3'},
        {'label': 'Very High', 'value': '4'},
        {'label': 'Exceptional', 'value': '5'}
    ],
    'WorkLifeBalance': [
        {'label': 'Poor', 'value': '1'},
        {'label': 'Below Average', 'value': '2'},
        {'label': 'Good', 'value': '3'},
        {'label': 'Excellent', 'value': '4'}
    ],
    'PerformanceRating': [
        {'label': 'Below Expectations', 'value': '1'},
        {'label': 'Needs Improvement', 'value': '2'},
        {'label': 'Meets Expectations', 'value': '3'},
        {'label': 'Exceeds Expectations', 'value': '4'}
    ],
    'OverTime': [
        {'label': 'Works Overtime', 'value': 'Yes'},
        {'label': 'Standard Hours', 'value': 'No'}
    ],
    'RelationshipSatisfaction': [
        {'label': 'Low', 'value': '1'},
        {'label': 'Medium', 'value': '2'},
        {'label': 'High', 'value': '3'},
        {'label': 'Very High', 'value': '4'},
        {'label': 'Exceptional', 'value': '5'}
    ],
    'CareerGrowthOpportunity': [
        {'label': 'Low', 'value': '1'},
        {'label': 'Medium', 'value': '2'},
        {'label': 'High', 'value': '3'},
        {'label': 'Very High', 'value': '4'}
    ],
    'StockOptionLevel': [
        {'label': 'None', 'value': '0'},
        {'label': 'Basic', 'value': '1'},
        {'label': 'Moderate', 'value': '2'},
        {'label': 'High', 'value': '3'}
    ],
    'JobLevel': [
        {'label': 'Entry Level', 'value': '1'},
        {'label': 'Junior', 'value': '2'},
        {'label': 'Mid', 'value': '3'},
        {'label': 'Senior', 'value': '4'},
        {'label': 'Executive', 'value': '5'}
    ]
}

# Get categorical and numeric features from loaded feature names
CATEGORICAL_FEATURES = [
    'JobSatisfaction', 
    'WorkLifeBalance', 
    'PerformanceRating', 
    'OverTime',
    'RelationshipSatisfaction', 
    'CareerGrowthOpportunity', 
    'StockOptionLevel', 
    'JobLevel'
]

NUMERIC_FEATURES = [
    'Age', 
    'MonthlyIncome', 
    'YearsAtCompany', 
    'TotalWorkingYears', 
    'WorkHours', 
    'DistanceFromHome', 
    'TrainingHoursLastYear'
]

# Custom CSS for the navbar icons
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Navbar
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Img(src='/assets/logo.jpg', height="30px")),
            dbc.Col(dbc.NavbarBrand("RecruitEase", className="ml-2", style={"color": "white"}))
        ], align="center"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-home mr-1"), " Home"], href="/", style={"color": "#F8F9FA"})),
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-chart-bar mr-1"), " Dashboard"], href="/dashboard", style={"color": "#F8F9FA"})),
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-user-plus mr-1"), " Recruitment"], href="/recruitment", style={"color": "#F8F9FA"})),
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-file-alt mr-1"), " Resume Score"], href="/resume-score", style={"color": "#F8F9FA"})),
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-user-minus mr-1"), " Attrition"], href="/attrition", style={"color": "#F8F9FA", "font-weight": "bold"})),
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-comment-dots mr-1"), " ChatBot"], href="/chatbot", style={"color": "#F8F9FA"}))
        ], className="ml-auto")
    ]), color="#2563EB", dark=True
)

# App Layout
app.layout = dbc.Container([
    navbar,

    dbc.Card([
        dbc.CardHeader(html.H3("Employee Attrition Predictor", className="text-center")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Upload Dataset (CSV)"),
                    dcc.Upload(
                        id='upload-data',
                        children=dbc.Button("Select CSV File", color="primary", style={"background-color": "#2563EB"}),
                        multiple=False
                    )
                ], width=12)
            ], className="mb-3"),

            # Feature Inputs - Categorical
            dbc.Row([
                dbc.Col([
                    dbc.Label(feature),
                    dcc.Dropdown(
                        id=f"{feature}-dropdown",
                        options=DROPDOWN_OPTIONS.get(feature, []),
                        placeholder=f"Select {feature}"
                    )
                ], width=4)
                for feature in CATEGORICAL_FEATURES
            ]),

            # Numeric Inputs
            dbc.Row([
                dbc.Col([
                    dbc.Label(feature),
                    dbc.Input(id=f"{feature}-input", type="number", placeholder=f"Enter {feature}")
                ], width=4)
                for feature in NUMERIC_FEATURES
            ], className="mt-3"),

            # Prediction Button
            dbc.Row([
                dbc.Col([
                    dbc.Button("Predict Attrition Risk", id="predict-button", color="primary", 
                              className="mt-3 w-100", style={"background-color": "#2563EB"})
                ])
            ])
        ])
    ], className="mt-4"),

    html.Div(id='prediction-output')
], fluid=True)

# Callback for File Upload
@app.callback(
    Output('upload-data', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_file_upload(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return dbc.Alert(f"File {filename} uploaded successfully!", color="success")
        except Exception as e:
            return dbc.Alert(f"Error processing file: {str(e)}", color="danger")
    return dbc.Button("Select CSV File", color="primary", style={"background-color": "#2563EB"})

# Prediction Callback
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State(f"{feature}-dropdown", 'value') for feature in CATEGORICAL_FEATURES] +
    [State(f"{feature}-input", 'value') for feature in NUMERIC_FEATURES],
    prevent_initial_call=True
)
def predict_attrition(n_clicks, *args):
    if None in args or "" in args:
        return dbc.Alert("Please fill out all fields.", color="warning")
    
    # Check if model is loaded
    if not model_loaded:
        return dbc.Alert("Model not loaded. Please train the model first.", color="danger")
    
    try:
        print("Arguments received:", args)
        
        # Create input dataframe with features in the EXACT same order as during training
        input_data = {}
        
        # First, initialize with all the features that were used during training
        for feature in feature_names:
            input_data[feature] = [0]  # Placeholder value
            
        # Then fill in with actual values
        # Process categorical inputs
        for i, feature in enumerate(CATEGORICAL_FEATURES):
            value = args[i]
            if feature == 'OverTime':
                # OverTime is 'Yes'/'No'
                input_data[feature] = [value]
            else:
                # Other categorical features are numeric strings
                input_data[feature] = [int(value)]
                
        # Process numeric inputs
        for i, feature in enumerate(NUMERIC_FEATURES):
            value = args[i + len(CATEGORICAL_FEATURES)]
            input_data[feature] = [float(value)]
            
        # Create DataFrame with exact column order from training
        input_df = pd.DataFrame(input_data)
        input_df = input_df[feature_names]  # Ensure exact same order as training
        
        print("Input DataFrame columns:", input_df.columns.tolist())
        
        # Apply label encoders for categorical features
        for feature, encoder in label_encoders.items():
            if feature in input_df.columns:
                if feature == 'OverTime':
                    # Handle OverTime which is Yes/No
                    input_df[feature] = encoder.transform(input_df[feature])
                else:
                    # For other categorical features, convert to string first (matches training)
                    input_df[feature] = encoder.transform(input_df[feature].astype(str))
        
        # Double-check that columns match exactly what the model expects
        print("Final DataFrame columns for prediction:", input_df.columns.tolist())
        
               # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # Determine attrition probability
        attrition_probability = prediction_proba[1]  # Assuming '1' corresponds to 'Yes' for attrition

        if prediction == 1:
            result_text = f"⚠️ The employee is likely to leave. Attrition Risk: {attrition_probability * 100:.2f}%"
            color = "danger"
        else:
            result_text = f"✅ The employee is likely to stay. Attrition Risk: {attrition_probability * 100:.2f}%"
            color = "success"

        return dbc.Alert(result_text, color=color)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return dbc.Alert(f"An error occurred during prediction: {e}", color="danger")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
    
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=10000)

app = Flask(__name__)
