param(
    [string]$Service = "ai-audit-assistant",
    [string]$Region = "europe-west1"
)

# Keeps one warm instance and gives Streamlit/PDF processing enough headroom.
# Run after deploying a new revision, or copy the same settings in the Cloud Run console.
gcloud run services update $Service `
    --region $Region `
    --min-instances 1 `
    --max-instances 10 `
    --concurrency 10 `
    --cpu 2 `
    --memory 4Gi `
    --timeout 3600 `
    --cpu-boost
