# Plant Disease Detection API

A FastAPI-based service for detecting plant diseases from images using deep learning.

## Features

- Plant disease detection from images
- Background removal
- Image preprocessing
- REST API endpoint for predictions
- Support for multiple plant types and diseases

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
```

2. Create and activate a conda environment:
```bash
conda create -n plant python=3.11
conda activate plant
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the API server:
```bash
python api/main.py
```

2. The API will be available at `http://localhost:8000`

3. Available endpoints:
   - GET `/ping` - Check if the service is running
   - POST `/predict` - Get plant disease prediction

## API Documentation

### Predict Endpoint

Send a POST request to `/predict` with a JSON body containing a base64-encoded image:

```json
{
    "image_base64": "base64_encoded_image_string"
}
```

Response:
```json
{
    "plantname_disease_name": "PlantName__DiseaseName",
    "confidence": 0.95
}
```

## Model Information

The model is trained to detect diseases in the following plants:
- Aloe Vera
- Money Plant
- Potato
- Tomato

## License

[Your chosen license]

## Author

[Your Name] 