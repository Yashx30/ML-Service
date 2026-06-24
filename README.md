Dental Image ML Inference API

A machine learning inference API for dental image analysis. This project demonstrates how a computer vision model can be served through an API to analyze dental images and return structured prediction results.

Overview

This project was built as a portfolio/demo version of an ML service used for dental image classification. The goal is to support automated dental image screening by accepting an image input, running model inference, and returning predicted dental conditions with confidence scores.

The service is designed to be lightweight, API-driven, and ready for future cloud deployment.

Key Features

* REST API for dental image inference
* Computer vision model integration
* Structured JSON prediction response
* Model versioning support
* Basic image validation and preprocessing
* Scalable project structure for future deployment
* Prepared for AWS/SageMaker-based model hosting

Use Case

The system can be used to support dental image screening workflows by identifying possible visual conditions such as:

* Caries
* Calculus
* Gingivitis
* Mouth ulcers
* Tooth discoloration
* Hypodontia

This project is intended as a technical demonstration and not as a medical diagnosis tool.

Tech Stack

* Python
* FastAPI
* Machine Learning / Computer Vision
* PyTorch / TensorFlow compatible structure
* AWS SageMaker-ready deployment structure
* AWS S3-ready image storage flow
* GitHub for version control

Project Structure

ML-service/
│
├── app/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   ├── models/
│   └── utils/
│
├── scripts/
│   └── helper scripts
│
├── requirements-ml.txt
├── README.md
├── .gitignore
└── .python-version

How It Works

1. User uploads or sends a dental image to the API.
2. The API validates and preprocesses the image.
3. The image is passed to the ML model for inference.
4. The model returns predicted dental conditions with confidence scores.
5. The API formats the output into a clean JSON response.

Example API Response

{
  "model_version": "v1.0",
  "overall_status": "ATTENTION",
  "predictions": [
    {
      "condition": "Caries",
      "confidence": 0.87
    },
    {
      "condition": "Gingivitis",
      "confidence": 0.72
    }
  ],
  "recommendation": "Possible dental issue detected. Please consult a dental professional."
}

Setup Instructions

Clone the repository:

git clone https://github.com/YOUR_USERNAME/dental-image-ml-inference-api.git
cd dental-image-ml-inference-api

Create a virtual environment:

python -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements-ml.txt

Run the API:

uvicorn app.main:app --reload

Open the API documentation:

http://127.0.0.1:8000/docs

My Contribution

* Built and maintained the ML inference API structure
* Worked on dental image preprocessing and prediction response design
* Supported model deployment planning using AWS/SageMaker
* Helped structure the service for future ML modules
* Worked with computer vision model outputs and confidence-based responses
* Focused on production-readiness, model serving, and API integration

Future Improvements

* Add model monitoring and drift detection
* Add image quality scoring before inference
* Add Docker support
* Add CI/CD pipeline for deployment
* Add cloud deployment using AWS Lambda, SageMaker, or ECS
* Add logging and model performance tracking
* Add unit tests for API routes and preprocessing functions

Disclaimer

This project is for educational and portfolio purposes only. It does not provide medical advice or diagnosis. Any dental findings should be reviewed by a licensed dental professional.
