
import gradio as gr
import pandas as pd
import pickle
# Define choices
yes_or_no = ["Yes", "No"]
# def  loaded_object(filepath='Gradio_toolkit'):
#     "Function to load saved objects"

#     with open(filepath, 'rb') as file:
#         loaded_object = pickle.load(file)
    
#     return loaded_object

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

# Extract the model and transformers
model = gradio_toolkit['model']
encode = gradio_toolkit['cat_preprocessor']
scaler = gradio_toolkit['num_transformer']
cat_encoder = gradio_toolkit['cat_transformer']


inputs = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines', 'DeviceProtection', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport']

categoricals = ['tenure', 'MonthlyCharges', 'TotalCharges','SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines', 'DeviceProtection', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport']

# Define your prediction function
def predict(*args, model=model, encoder=encode):
    # Creating a dataframe of inputs
    input_data = pd.DataFrame([args], columns=inputs)
    
    all_inputs = input_data[inputs]
    # numerical_inputs = input_data.drop(columns=categoricals)

    # Encode the categorical columns
    preprocessed_data = encoder.transform(all_inputs)

    # Define a manual list of feature names based on your model and Selected_Features
    # Ensure that this list matches your model's expectations
    preprocessed_columns = [
       'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen_No',
       'SeniorCitizen_Yes', 'Partner_No', 'Partner_Yes', 'Dependents_No',
       'Dependents_Yes', 'MultipleLines_No', 'MultipleLines_No phone service',
       'MultipleLines_Yes', 'InternetService_DSL',
       'InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'OnlineBackup_No',
       'OnlineBackup_No internet service', 'OnlineBackup_Yes',
       'DeviceProtection_No', 'DeviceProtection_No internet service',
       'DeviceProtection_Yes', 'TechSupport_No',
       'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
       'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No internet service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
       'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]

    Selected_Features = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'InternetService_Fiber optic', 'OnlineSecurity_No', 'OnlineSecurity_Yes',
        'OnlineBackup_No', 'DeviceProtection_No', 'TechSupport_No',
        'Contract_Month-to-month', 'PaperlessBilling_Yes', 'PaymentMethod_Electronic check'
    ]

    # Ensure that Selected Features are valid
    for feature in Selected_Features:
        if feature not in preprocessed_columns:
            raise ValueError(f"Selected feature '{feature}' is not valid.")

    # Create a DataFrame with the selected features
    selected_features_df = pd.DataFrame(preprocessed_data, columns=preprocessed_columns)
    selected_features_df = selected_features_df[Selected_Features]
    
    # Make predictions
    model_output = model.predict(selected_features_df)
    output_str = "Your customer will churn." if model_output[0] == 1 else "Your customer will not churn."
    return output_str




    # Make predictions
    # model_output = model.predict(encoded_categoricals[Selected_Features])
    # output_str = "Your customer will churn." if model_output[0] == 1 else "Your customer will not churn."
    # return output_str




# Define the interface
yes_or_no = ["Yes", "No"]
internet_service_choices = ["Yes", "No", "No internet service"]
# Define the custom theme with a light blue background


# Define Gradio components
with gr.Blocks(theme=gr.themes.Base(primary_hue="stone",neutral_hue="stone",)) as block:
    text = gr.Markdown("<div style='text-align: center;'><span style='font-size: 20px; font-weight: bold; color: blue;'>CUSTOMER CHURN PREDICTION APPLICATION</span></div>")

    text = gr.Markdown( "<div style='text-align: center;'><span style='font-size: 20px; font-weight: bold; color: black;'>Welcome!  Enter your customer's attributes to predict wheither the customer will churn or not.</span></div>")
    with gr.Row():
        with gr.Column():
            tenure = gr.components.Slider(label="Tenure (months)", minimum=1, maximum=12, step=1)
            MonthlyCharges = gr.components.Slider(label="Monthly Charges", step=0.05, maximum=7000)  # Set a default value
            TotalCharges = gr.components.Slider(label="Total Charges", step=0.05, maximum=10000)     # Set a default value

            SeniorCitizen = gr.components.Radio(label="Senior Citizen", choices=yes_or_no)
            Partner = gr.components.Radio(label="Partner", choices=yes_or_no)
            Dependents = gr.components.Radio(label="Dependents", choices=yes_or_no)
            DeviceProtection = gr.components.Radio(label="Device Protection", choices=yes_or_no)
            
            
        with gr.Column():
            MultipleLines = gr.components.Dropdown(label="Multiple Lines", choices=["Yes", "No", "No phone service"])
            StreamingTV = gr.components.Dropdown(label="TV Streaming", choices=internet_service_choices)
            StreamingMovies = gr.components.Dropdown(label="Movie Streaming", choices=internet_service_choices)
            Contract = gr.components.Dropdown(label="Contract", choices=["Month-to-month", "One year", "Two year"])
            PaperlessBilling = gr.components.Radio(label="Paperless Billing", choices=yes_or_no)
            
            
            
        with gr.Column():
           PaymentMethod = gr.components.Dropdown(label="Payment Method", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
           InternetService = gr.components.Dropdown(label="Internet Service", choices=["DSL", "Fiber optic", "No"])
           OnlineSecurity = gr.components.Dropdown(label="Online Security", choices=internet_service_choices)
           OnlineBackup = gr.components.Dropdown(label="Online Backup", choices=internet_service_choices)
           TechSupport = gr.components.Dropdown(label="Tech Support", choices=internet_service_choices)

                   

            #create a variable that clear button will clear
    input_components = [tenure, MonthlyCharges, TotalCharges, SeniorCitizen, Partner, Dependents, MultipleLines, DeviceProtection, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, InternetService, OnlineSecurity, OnlineBackup, TechSupport]
            

   
    output = gr.components.Textbox(label="Prediction")
                 
    #create markdown for ouput
    
    
    # Define Gradio outputs
    # output = gr.HTML("Awaiting Prediction")

    # Create a button
    button = gr.Button("Predict")

    # Create Gradio interface
    button.click(fn=predict,inputs=input_components, outputs=output)
  

    # Define the "Clear" button function
    def clear_inputs():
        for input_component in input_components:
            if isinstance(input_component, gr.components.Slider):
                input_component.value = input_component.minimum  # Reset sliders to their minimum value
            elif isinstance(input_component, gr.components.Radio):
                input_component.value = input_component.choices[0]  # Reset radios to their first choice

        # Clear the output
        output.value = ""

    # Assign the click function to the "Clear" button
    # clear_button.click(clear_inputs)
#start gradio app
block.launch(share=True)
    