SWELL Stress Detection API with XAI
🚀 Project Overview
This project presents a robust system designed to detect stress levels using Heart Rate Variability (HRV) features derived from the SWELL dataset. At its core, it leverages a sophisticated deep learning model: a Convolutional Neural Network (CNN) combined with a Long Short-Term Memory (LSTM) network, engineered for accurate classification of stress states.

A pivotal aspect of this system is its integration of Explainable AI (XAI), specifically employing SHAP (SHapley Additive exPlanations). This allows for unparalleled transparency by providing clear insights into why a particular stress prediction was made, highlighting the specific HRV features that influenced the model's decision.

The entire solution is built around a high-performance FastAPI backend, providing a scalable and efficient API for handling real-time prediction requests. Complementing this, an intuitive Streamlit application serves as the user-friendly frontend, empowering users to easily input HRV data, receive instant stress predictions, and visualize the detailed SHAP explanations.

💻 Prerequisites and System Requirements
To set up and run this project, ensure your system meets the following requirements:

Operating System: Windows, macOS, or Linux.
Python Version: Python 3.9 or higher is recommended.
Memory (RAM): 8 GB RAM or more is recommended for smooth operation, especially during model loading and SHAP explanation generation.
Storage: Sufficient disk space for the project files, datasets (train.csv, test.csv), and the pre-trained model (best_model.pth).
Internet Connection: Required for initial package installation.
🧠 System Architecture
The system follows a client-server architecture, divided into a backend API and a frontend user interface. Data flows from the user input through the frontend to the backend for processing and then back to the frontend for display.

Workflow Breakdown:

User Interaction: A user interacts with the Streamlit Frontend via a web browser, inputting raw HRV features.
API Request: The Streamlit app sends an HTTP POST request containing the HRV features to the FastAPI Backend.
Backend Processing:
FastAPI receives and validates the input data.
The raw features are passed through the same data preprocessing pipeline (scaling, reshaping) used during model training.
The preprocessed features are fed into the loaded CNN-LSTM model for stress prediction.
Simultaneously, the SHAP utility (using a subset of the training data as background) analyzes the model's prediction for the given input, calculating feature importance scores.
Response Generation: FastAPI compiles the predicted stress class and the human-readable SHAP explanations into a JSON response.
Result Display: The Streamlit frontend receives the JSON response and elegantly displays the predicted stress level along with the detailed feature explanations to the user.
🚀 Tech Stack
This project is built using a modern and efficient set of technologies:

Backend Framework:
FastAPI: A cutting-edge, high-performance web framework for building robust and scalable APIs. It's chosen for its speed, automatic interactive API documentation (/docs), and strong type hints.
Uvicorn: An ASGI server that serves the FastAPI application, providing asynchronous capabilities for handling concurrent requests.
Deep Learning:
PyTorch: The primary deep learning framework. It's used for defining, training, and running the custom CNN-LSTM model, offering flexibility and control over neural network architectures.
torchsummary: A utility library to visualize PyTorch model architecture and parameters, aiding in debugging and understanding.
Explainable AI (XAI):
SHAP: A powerful library for interpreting machine learning models. It provides model-agnostic explanations by calculating Shapley values, highlighting the contribution of each feature to a prediction.
Frontend Development:
Streamlit: An open-source Python library that turns data scripts into shareable web apps in minutes. It's ideal for quickly creating interactive dashboards and demos without needing extensive web development knowledge.
Data Handling & Utilities:
NumPy: The fundamental package for numerical computing in Python, used extensively for array operations and mathematical functions.
Pandas: A powerful library for data manipulation and analysis, primarily used here for handling tabular data like train.csv and test.csv.
scikit-learn: Provides essential tools for machine learning, including data preprocessing utilities like StandardScaler for feature normalization and LabelEncoder for converting categorical labels to numerical formats.
🙏 Acknowledgements
We extend our gratitude to the following:

The SWELL Knowledge Work Dataset: For providing the comprehensive dataset on stress and physiological data, which is fundamental to this project.
FastAPI, PyTorch, SHAP, Streamlit, Uvicorn, NumPy, Pandas, scikit-learn, and torchsummary communities: For developing and maintaining these incredible open-source libraries that make projects like this possible.
Researchers and developers whose work in deep learning, HRV analysis, and Explainable AI contributed to the concepts implemented here.
