# FlowerVision 🌸
### Flower Classification System — Machine Learning Operations (MLOps)

FlowerVision is an image classification application that uses deep learning to identify five types of flowers from uploaded images. It combines a trained machine learning model, a FastAPI backend, an ONNX inference model, a Streamlit frontend, and Docker containerization.

The application allows users to upload a flower image and receive a predicted flower category with its confidence score.

## Live Application

- **Streamlit Web App:** https://flowervision-alu.streamlit.app/ 
- **FastAPI Backend:** https://flower-vision-api-sm13.onrender.com
- **API Documentation:** https://flower-vision-api-sm13.onrender.com/docs

## Supported Flower Categories

The model classifies images into five categories:

- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

**Note:** The model is trained to recognize these five categories. Images outside these categories may still receive a prediction from one of the supported classes.

## Model Development and Evaluation

The classification model was developed using transfer learning with MobileNetV2.

The development process included:

- Dataset exploration and visualization.
- Image preprocessing and stratified dataset splitting.
- Data augmentation.
- Transfer learning using MobileNetV2.
- Fine-tuning of the pretrained model.
- Model evaluation and comparison.
- Conversion of the selected model to ONNX format.

### Dataset

The dataset contains 3,670 flower images across five classes.

| Dataset split | Number of images |
|---|---:|
| Training | 2,569 |
| Validation | 550 |
| Testing | 551 |
| **Total** | **3,670** |

### Model Performance

The selected fine-tuned model achieved the following results on the test dataset:

| Evaluation metric | Score |
|---|---:|
| Accuracy | 88.93% |
| Macro Precision | 88.69% |
| Macro Recall | 88.78% |
| Macro F1-score | 88.71% |

### Model Optimization

The trained model was converted from Keras to ONNX to support lightweight inference through ONNX Runtime.

The ONNX model was approximately 8.52 MB, compared with approximately 20.79 MB for the Keras model. Predictions were verified against the original model, with a maximum probability difference of approximately 0.00000012.

## System Architecture

FlowerVision uses the following components:

1. **Machine Learning Model:** MobileNetV2 transfer learning and fine-tuning.
2. **Model Format:** ONNX.
3. **Inference Backend:** FastAPI and ONNX Runtime.
4. **Frontend:** Streamlit.
5. **Containerization:** Docker.
6. **Backend Hosting:** Render.
7. **Frontend Hosting:** Streamlit Community Cloud.
8. **Load Testing:** Locust.

### Prediction Workflow

1. The user uploads a flower image through the Streamlit interface.
2. The frontend sends the image to the FastAPI prediction endpoint.
3. The backend preprocesses the image.
4. ONNX Runtime performs inference using the trained model.
5. The API returns the predicted class and class probabilities.
6. The frontend displays the prediction and confidence information.

## Project Structure

```text
MLOP/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── locustfile.py
├── README.md
├── FlowerVision_Image_Classification.ipynb
├── test_flower.jpg
│
├── models/
│   ├── class_names.json
│   ├── flower_vision_final.keras
│   └── flower_vision_final.onnx
│
├── frontend/
│   ├── app.py
│   └── requirements.txt
│
└── results/
    └── load_test/
        ├── locust_requests.csv
        ├── locust_failures.csv
        ├── locust_exceptions.csv
        └── locust_report.html
```

## Running the Backend Locally

### Prerequisites

- Python 3.11
- pip
- Docker Desktop (for containerized execution)

### 1. Clone the repository

```bash
git clone https://github.com/Esther446/MLOP.git
cd MLOP
```

### 2. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

## Running the Frontend Locally

From the project root:

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

The frontend will be available at the local address displayed in the terminal.

## Running with Docker

The backend can also be built and run inside a Docker container.

### Build the Docker image

From the project root:

```bash
docker build -t flowervision-api .
```

### Run the container

```bash
docker run -d -p 8000:8000 --name flowervision-container flowervision-api
```

The API can then be accessed at:

```text
http://localhost:8000
```

To stop the container:

```bash
docker stop flowervision-container
```

## Load Testing

Locust was used to evaluate the performance of the containerized FastAPI backend.

### Test Configuration

| Setting | Value |
|---|---:|
| Concurrent users | 5 |
| Ramp-up rate | 1 user/second |
| Test duration | 1 minute |
| Total requests | 193 |

### Results

| Metric | Result |
|---|---:|
| Successful requests | 193 |
| Failed requests | 0 |
| Average response time | 14.93 ms |
| Median response time | 10 ms |
| 95th percentile | 53 ms |
| 99th percentile | 58 ms |
| Maximum response time | 288 ms |
| Current RPS at observation | 3.2 |

The API successfully handled all requests during this test, with no recorded failures.

The test was conducted locally against the Dockerized API. These results represent the tested workload and are not a benchmark of the publicly hosted service under heavy traffic.

Detailed test exports are available in `results/load_test/`.


## Future Improvements

- Implement a complete data upload and storage workflow for retraining.
- Add a retraining pipeline using the developed model as a pretrained model.
- Expand the dataset to support additional flower categories.
- Evaluate performance under larger concurrent workloads.
- Improve model generalization to images outside the supported categories.
