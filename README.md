# FlowerVision 🌸
### Flower Classification System — Machine Learning Operations (MLOps)

FlowerVision is an image classification application that uses deep learning to identify five types of flowers from uploaded images. It combines a trained machine learning model, a FastAPI backend, an ONNX inference model, a Streamlit frontend, and Docker containerization.

The application allows users to upload a flower image and receive a predicted flower category with its confidence score. It also includes a data contribution and retraining workflow that allows users to upload labelled images for future model training.

## Live Application

- **Streamlit Web App:** [https://flowervision-alu.streamlit.app/](https://flowervision-alu.streamlit.app/)
- **FastAPI Backend:** [https://flower-vision-api-sm13.onrender.com](https://flower-vision-api-sm13.onrender.com)
- **API Documentation:** [https://flower-vision-api-sm13.onrender.com/docs](https://flower-vision-api-sm13.onrender.com/docs)
- **Drive Folder:** [https://drive.google.com/drive/folders/1bjaca_6I-y1WYcFnH3yggFCXRDTda3_5?usp=sharing](https://drive.google.com/drive/folders/1bjaca_6I-y1WYcFnH3yggFCXRDTda3_5?usp=sharing)
- **Demo Video:** [Watch the FlowerVision Demonstration](https://www.loom.com/share/81c371562bea461e9d3e2973801732c3)

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

The original dataset contains 3,670 flower images across five classes.

| Dataset split | Number of images |
|---|---:|
| Training | 2,569 |
| Validation | 550 |
| Testing | 551 |
| **Total** | **3,670** |

### Initial Model Performance

The selected fine-tuned model achieved the following results on the original test evaluation:

| Evaluation metric | Score |
|---|---:|
| Accuracy | 88.93% |
| Macro Precision | 88.69% |
| Macro Recall | 88.78% |
| Macro F1-score | 88.71% |
| Test Loss | 0.3040 |

### Model Optimization

The trained model was converted from Keras to ONNX to support lightweight inference through ONNX Runtime.

The ONNX model was approximately 8.52 MB, compared with approximately 20.79 MB for the Keras model. Predictions were verified against the original model, with a maximum probability difference of approximately 0.00000012.

## System Architecture

FlowerVision uses the following components:

1. **Machine Learning Model:** MobileNetV2 transfer learning and fine-tuning.
2. **Model Format:** ONNX.
3. **Inference Backend:** FastAPI and ONNX Runtime.
4. **Frontend:** Streamlit.
5. **Database and Image Storage:** Supabase.
6. **Model Development and Retraining:** TensorFlow and Google Colab.
7. **Containerization:** Docker.
8. **Backend Hosting:** Render.
9. **Frontend Hosting:** Streamlit Community Cloud.
10. **Load Testing:** Locust.

### Prediction Workflow

1. The user uploads a flower image through the Streamlit interface.
2. The frontend sends the image to the FastAPI prediction endpoint.
3. The backend preprocesses the image.
4. ONNX Runtime performs inference using the trained model.
5. The API returns the predicted class and class probabilities.
6. The frontend displays the prediction and confidence information.

## Data Contribution and Retraining Pipeline

FlowerVision includes a data contribution workflow that allows users to upload labelled flower images for future model training.

The workflow uses Supabase for image storage and database management, and Google Colab for data preprocessing and model retraining.

### Data Upload and Storage

1. Users upload a flower image and select its corresponding flower category through the Streamlit interface.
2. The image is sent to the FastAPI backend for validation and processing.
3. The processed image is stored in a private Supabase Storage bucket.
4. Image metadata, including the filename, storage path, flower label, upload date, and processing status, is saved in the `uploaded_images` database table.

### Data Preprocessing

The Google Colab notebook connects to Supabase and retrieves uploaded images with a pending status.

The preprocessing steps include:

- Downloading images from Supabase Storage.
- Converting images to RGB format.
- Resizing images to 180 × 180 pixels.
- Normalizing pixel values.
- Converting flower labels into numerical class labels.

The processed images are then combined with the original training dataset. The original validation and test datasets remain unchanged.

### Model Retraining

The existing FlowerVision MobileNetV2 model is used as the pretrained model for fine-tuning.

The retraining process includes:

1. Loading the existing fine-tuned model.
2. Combining the original training dataset with newly contributed images.
3. Compiling the model with a low learning rate.
4. Fine-tuning using the expanded training dataset.
5. Monitoring validation loss and accuracy.
6. Applying early stopping and model checkpointing.
7. Saving the best retrained candidate model separately from the production model.

After the retraining process, the uploaded image records are updated in Supabase to indicate that they have been used for training.

### Retraining Dataset

Five newly contributed images, one from each flower category, were retrieved from Supabase and added to the original training dataset.

| Dataset | Number of images |
|---|---:|
| Original training dataset | 2,569 |
| Newly contributed images | 5 |
| **Expanded training dataset** | **2,574** |

The expanded training dataset contains:

| Flower category | Original images | Uploaded images | Combined images |
|---|---:|---:|---:|
| Daisy | 443 | 1 | 444 |
| Dandelion | 629 | 1 | 630 |
| Roses | 449 | 1 | 450 |
| Sunflowers | 489 | 1 | 490 |
| Tulips | 559 | 1 | 560 |
| **Total** | **2,569** | **5** | **2,574** |

### Retraining Evaluation and Model Selection

After fine-tuning, the original model and retrained candidate were evaluated using the same untouched test dataset.

The following metrics were used for comparison:

| Evaluation metric | Original Model | Retrained Model |
|---|---:|---:|
| Accuracy | 90.20% | 90.02% |
| Macro Precision | 89.91% | 89.79% |
| Macro Recall | 90.14% | 89.97% |
| Macro F1-score | 90.01% | 89.84% |
| Test Loss | 0.2938 | 0.2955 |

The original model performed slightly better across the evaluated metrics. Therefore, it was retained as the production model, while the retrained candidate was saved separately for comparison.

This ensures that a newly retrained model is evaluated before being selected for production.

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

The retraining notebook, retrained candidate model, and model comparison results are available in the linked Google Drive folder.

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
