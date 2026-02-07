# RT-DETR Vertex AI Training Guide

This guide explains how to run RT-DETR training on Google Cloud Vertex AI using KaggleHub for dataset management.

## Prerequisites

1.  **Google Cloud Project**: You need a GCP project with billing enabled.
2.  **Vertex AI API**: Enable the Vertex AI API.
3.  **Google Cloud SDK**: Install and authenticate `gcloud`.
    ```bash
    gcloud auth login
    gcloud auth configure-docker
    ```
4.  **GCS Bucket**: Create a bucket for storing logs and staging artifacts (e.g., `gs://my-rtdetr-bucket`).
5.  **Artifact Registry or GCR**: Used to store the Docker image.

## 1. Build and Push Docker Image

Set your variables:
```bash
export PROJECT_ID="your-project-id"
export REPO_NAME="rtdetr-repo" # If using Artifact Registry
export IMAGE_NAME="rtdetr-train"
export TAG="latest"
# Check if you are using GCR (gcr.io) or Artifact Registry (us-docker.pkg.dev)
export IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
```

Build the image:
```bash
docker build -t $IMAGE_URI -f Dockerfile .
```

Push the image:
```bash
docker push $IMAGE_URI
```

## 2. Submit Training Job

The training job will automatically download the dataset `duwipurnamasidik/visdrone-2019-coco-format` using `kagglehub`.

First, install the python client locally if you haven't:
```bash
pip install google-cloud-aiplatform
```

Run the submission script:
```bash
python submit_vertex_job.py \
  --project_id "your-project-id" \
  --staging_bucket "gs://my-rtdetr-bucket" \
  --container_uri "gcr.io/your-project-id/rtdetr-train:latest" \
  --config_path "configs/rtdetr/rtdetr_r18vd_p2_gnconv.yml"
```

You can optionally specify a different dataset handle:
```bash
  --dataset_handle "another/dataset-handle"
```

## 3. Monitor Job

Go to the [Vertex AI Training page](https://console.cloud.google.com/vertex-ai/training/training-pipelines) to see your job status and logs.
