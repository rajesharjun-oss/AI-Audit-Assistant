from pathlib import Path


def test_cloudrun_service_config_keeps_warm_scaling_headroom():
    text = Path("cloudrun-service.yaml").read_text(encoding="utf-8")

    assert 'autoscaling.knative.dev/minScale: "1"' in text
    assert 'autoscaling.knative.dev/maxScale: "10"' in text
    assert "containerConcurrency: 10" in text
    assert "timeoutSeconds: 3600" in text
    assert 'run.googleapis.com/startup-cpu-boost: "true"' in text


def test_streamlit_config_is_cloudrun_friendly():
    text = Path(".streamlit/config.toml").read_text(encoding="utf-8")

    assert "[server]" in text
    assert "headless=true" in text
    assert 'fileWatcherType="none"' in text
    assert "[browser]" in text
    assert "gatherUsageStats=false" in text


def test_deploy_helper_documents_capacity_update_command():
    text = Path("deploy-cloudrun.ps1").read_text(encoding="utf-8")

    assert "gcloud run services update" in text
    assert "--min-instances 1" in text
    assert "--max-instances 10" in text
    assert "--concurrency 10" in text
    assert "--timeout 3600" in text
