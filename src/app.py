import gradio as gr
import pandas as pd
import pickle
import os

# Load the saved components
def load_pickle(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)

loaded_components = load_pickle(r'src\assets\gradio_toolkit.pkl')

model = loaded_components['model']
encoder = loaded_components['cat_preprocessor']

# Load the Gradio toolkit
# Load the trained model
with open(r'src\assets\optimized_gb_classifier.pkl', 'rb') as model_file:
    optimized_gb_classifier = pickle.load(model_file)

# Load the transformers (cat_preprocessor and num_transformer)
with open(r'src\assets\cat_preprocessor.pkl', 'rb') as cat_preprocessor_file:
    cat_preprocessor = pickle.load(cat_preprocessor_file)

with open(r'src\assets\num_transformer.pkl', 'rb') as num_transformer_file:
    num_transformer = pickle.load(num_transformer_file)

with open(r'src\assets\cat_transformer.pkl', 'rb') as cat_transformer_file:
    cat_transformer = pickle.load(cat_transformer_file)

# Create a dictionary to hold all the components

with open(r'src\assets\gradio_toolkit.pkl', 'rb') as toolkit_file:
    gradio_toolkit = pickle.load(toolkit_file)

# Define choices
yes_or_no = ["Yes", "No"]
internet_service_choices = ["Yes", "No", "No internet service"]

# Extract the model and transformers
model = gradio_toolkit['model']
encode = gradio_toolkit['cat_preprocessor']
scaler = gradio_toolkit['num_transformer']
cat_encoder = gradio_toolkit['cat_transformer']


inputs = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines', 'DeviceProtection', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport']

categoricals = ['tenure', 'MonthlyCharges', 'TotalCharges','SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines', 'DeviceProtection', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport']


# Define Gradio components
text1 = gr.Markdown("<div style='text-align: center;'><span style='font-size: 20px; font-weight: bold; color: blue;'>CUSTOMER CHURN PREDICTION APPLICATION</span></div>")
text2 = gr.Markdown("<div style='text-align: center;'><span style='font-size: 20px; font-weight: bold; color: black;'>Welcome! Enter your customer's attributes to predict whether the customer will churn or not.</span></div>")

# Input components for Gradio interface
input_components = [
    gr.Slider(label="Tenure (months)", minimum=1, maximum=12, step=1),
    gr.Slider(label="Monthly Charges", step=0.05, maximum=7000),
    gr.Slider(label="Total Charges", step=0.05, maximum=10000),
    gr.Radio(label="Senior Citizen", choices=yes_or_no),
    gr.Radio(label="Partner", choices=yes_or_no),
    gr.Radio(label="Dependents", choices=yes_or_no),
    gr.Radio(label="Device Protection", choices=yes_or_no),
    gr.Dropdown(label="Multiple Lines", choices=["Yes", "No", "No phone service"]),
    gr.Dropdown(label="TV Streaming", choices=internet_service_choices),
    gr.Dropdown(label="Movie Streaming", choices=internet_service_choices),
    gr.Dropdown(label="Contract", choices=["Month-to-month", "One year", "Two year"]),
    gr.Radio(label="Paperless Billing", choices=yes_or_no),
    gr.Dropdown(label="Payment Method", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
    gr.Dropdown(label="Internet Service", choices=["DSL", "Fiber optic", "No"]),
    gr.Dropdown(label="Online Security", choices=internet_service_choices),
    gr.Dropdown(label="Online Backup", choices=internet_service_choices),
    gr.Dropdown(label="Tech Support", choices=internet_service_choices)
]

# Prediction function
def predict(*args):
    input_data = pd.DataFrame([args], columns=inputs)
    preprocessed_data = encoder.transform(input_data)
    prediction = model.predict(preprocessed_data)
    return "Your customer will churn." if prediction[0] == 1 else "Your customer will not churn."

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=input_components,
    outputs="textbox",
    live=False,
    title="CUSTOMER CHURN PREDICTION APPLICATION",
    description="Enter your customer's attributes to predict whether the customer will churn or not."
)

# Launch the Gradio interface
iface.launch(share=True)
