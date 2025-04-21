import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import base64
import io
import joblib

# Create Flask App first
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "✅ Employee Attrition Predictor is Live!"

# Create Dash app with the Flask server
app = dash.Dash(
    __name__, 
    server=flask_app, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Set app title
app.title = "RecruitEase"

# Try loading model and assets
try:
    model = joblib.load("attrition_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    model_loaded = True
    print("Model and encoders loaded successfully.")
except Exception as e:
    print(f"Model loading error: {e}")
    model_loaded = False
    feature_names = []

# Dropdown Options
DROPDOWN_OPTIONS = {
    'JobSatisfaction': [{'label': l, 'value': str(i+1)} for i, l in enumerate(['Low', 'Medium', 'High', 'Very High', 'Exceptional'])],
    'WorkLifeBalance': [{'label': l, 'value': str(i+1)} for i, l in enumerate(['Poor', 'Below Average', 'Good', 'Excellent'])],
    'PerformanceRating': [{'label': l, 'value': str(i+1)} for i, l in enumerate(['Below Expectations', 'Needs Improvement', 'Meets Expectations', 'Exceeds Expectations'])],
    'OverTime': [{'label': 'Works Overtime', 'value': 'Yes'}, {'label': 'Standard Hours', 'value': 'No'}],
    'RelationshipSatisfaction': [{'label': l, 'value': str(i+1)} for i, l in enumerate(['Low', 'Medium', 'High', 'Very High', 'Exceptional'])],
    'CareerGrowthOpportunity': [{'label': l, 'value': str(i+1)} for i, l in enumerate(['Low', 'Medium', 'High', 'Very High'])],
    'StockOptionLevel': [{'label': l, 'value': str(i)} for i, l in enumerate(['None', 'Basic', 'Moderate', 'High'])],
    'JobLevel': [{'label': l, 'value': str(i+1)} for i, l in enumerate(['Entry Level', 'Junior', 'Mid', 'Senior', 'Executive'])]
}

CATEGORICAL_FEATURES = list(DROPDOWN_OPTIONS.keys())
NUMERIC_FEATURES = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears', 'WorkHours', 'DistanceFromHome', 'TrainingHoursLastYear']

# HTML Template for font-awesome
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
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-home mr-1"), " Home"], href="/")),
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-user-minus mr-1"), " Attrition"], href="#", style={"font-weight": "bold"}))
        ], className="ml-auto")
    ]), color="#2563EB", dark=True
)

# Layout
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
                ])
            ], className="mb-3"),

            # Dropdowns for categorical
            dbc.Row([
                dbc.Col([
                    dbc.Label(feature),
                    dcc.Dropdown(id=f"{feature}-dropdown", options=DROPDOWN_OPTIONS[feature], placeholder=f"Select {feature}")
                ], width=4)
                for feature in CATEGORICAL_FEATURES
            ]),

            # Inputs for numeric
            dbc.Row([
                dbc.Col([
                    dbc.Label(feature),
                    dbc.Input(id=f"{feature}-input", type="number", placeholder=f"Enter {feature}")
                ], width=4)
                for feature in NUMERIC_FEATURES
            ], className="mt-3"),

            # Button
            dbc.Row([
                dbc.Col([
                    dbc.Button("Predict Attrition Risk", id="predict-button", color="primary", className="mt-3 w-100", style={"background-color": "#2563EB"})
                ])
            ])
        ])
    ], className="mt-4"),

    html.Div(id='prediction-output')
], fluid=True)

# File upload
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
            pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return dbc.Alert(f"File {filename} uploaded successfully!", color="success")
        except Exception as e:
            return dbc.Alert(f"Error processing file: {str(e)}", color="danger")
    return dbc.Button("Select CSV File", color="primary", style={"background-color": "#2563EB"})

# Prediction logic
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

    if not model_loaded:
        return dbc.Alert("Model not loaded. Please check deployment logs.", color="danger")

    try:
        input_data = {feature: [0] for feature in feature_names}

        # Fill categorical values
        for i, feature in enumerate(CATEGORICAL_FEATURES):
            val = args[i]
            input_data[feature] = [val if feature == 'OverTime' else int(val)]

        # Fill numeric values
        for i, feature in enumerate(NUMERIC_FEATURES):
            input_data[feature] = [float(args[len(CATEGORICAL_FEATURES) + i])]

        df = pd.DataFrame(input_data)[feature_names]

        # Label encoding
        for feature, encoder in label_encoders.items():
            if feature == 'OverTime':
                df[feature] = encoder.transform(df[feature])
            else:
                df[feature] = encoder.transform(df[feature].astype(str))

        # Predict
        pred = model.predict(df)[0]
        label = "⚠️ Likely to Leave" if pred == 1 else "✅ Likely to Stay"
        color = "danger" if pred == 1 else "success"

        return dbc.Alert(f"Prediction: {label}", color=color)

    except Exception as e:
        return dbc.Alert(f"Prediction error: {str(e)}", color="danger")

# Run Flask server (for deployment via gunicorn or locally)
if __name__ == "__main__":
    flask_app.run(debug=True)
