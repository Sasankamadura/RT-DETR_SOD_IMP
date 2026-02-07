
import argparse
from google.cloud import aiplatform

def submit_job(
    project_id: str,
    location: str,
    staging_bucket: str,
    display_name: str,
    container_uri: str,
    config_path: str,
    dataset_handle: str = None,
    dataset_path: str = None,
    machine_type: str = "n1-standard-8",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
):
    aiplatform.init(project=project_id, location=location, staging_bucket=staging_bucket)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_uri,
        # command=["python", "entrypoint.py"], # Implied by ENTRYPOINT
    )

    # Arguments to pass to the entrypoint script
    args = [
        "--config", config_path,
    ]
    
    if dataset_handle:
        args.extend(["--dataset_handle", dataset_handle])
        
    if dataset_path:
        # If user still wants to pass a GCS path or similar (though entrypoint.py logic currently favors kagglehub if handle is present)
        # We can add custom logic here if needed. For now, entrypoint.py primarily uses kagglehub.
        # This might be used if entrypoint logic was expanded to handle GCS paths too.
        pass
    
    print(f"Submitting job {display_name} to Vertex AI...")
    model = job.run(
        args=args,
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        sync=False, # Return immediately
    )
    
    print(f"Job submitted. Resource name: {model.resource_name}")
    print(f"View job in Cloud Console: https://console.cloud.google.com/vertex-ai/training/training-pipelines?project={project_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit RT-DETR training job to Vertex AI")
    parser.add_argument("--project_id", type=str, required=True, help="GCP Project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="GCP Region")
    parser.add_argument("--staging_bucket", type=str, required=True, help="GCS Bucket for staging artifacts (gs://...)")
    parser.add_argument("--display_name", type=str, default="rtdetr-training", help="Display name for the job")
    parser.add_argument("--container_uri", type=str, required=True, help="URI of the Docker image")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file relative to project root")
    parser.add_argument("--dataset_handle", type=str, default="duwipurnamasidik/visdrone-2019-coco-format", help="Kaggle dataset handle")
    parser.add_argument("--dataset_path", type=str, help="Optional: specific dataset path if not using KaggleHub")
    
    args = parser.parse_args()

    submit_job(
        project_id=args.project_id,
        location=args.location,
        staging_bucket=args.staging_bucket,
        display_name=args.display_name,
        container_uri=args.container_uri,
        config_path=args.config_path,
        dataset_handle=args.dataset_handle,
        dataset_path=args.dataset_path,
    )
