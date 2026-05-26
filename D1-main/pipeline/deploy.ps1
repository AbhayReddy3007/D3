<#
.SYNOPSIS
    Deploy the LOE pipeline to Cloud Run Jobs from Windows.
    Run from the pipeline folder (where Dockerfile is).

.NOTES
    Prerequisites:
      1. gcloud CLI installed  (https://cloud.google.com/sdk/docs/install)
      2. service-account.json placed in this folder
      3. Edit the CONFIG section below with your values

.EXAMPLE
    cd C:\pipeline
    .\deploy.ps1
#>

$ErrorActionPreference = "Stop"

# ═══════════════════════════════════════════════════════════════
# CONFIG — edit these values
# ═══════════════════════════════════════════════════════════════
$GCP_PROJECT   = "cognito-prod-394707"
$GCP_REGION    = "asia-south1"
$REPO_NAME     = "pipeline-repo"
$VPC_CONNECTOR = "pipeline-connector"     # your VPC connector for AlloyDB access

# Environment variables for the Cloud Run Jobs
# Fill in ALL values below
$ENV_VARS = @(
    "PROJECT_ID=$GCP_PROJECT",
    "BQ_PROJECT_ID=$GCP_PROJECT",
    "BQ_UPLOAD_PROJECT=$GCP_PROJECT",
    "BQ_UPLOAD_DATASET=cognito_prod_datamart",
    "BQ_DATASET_ID=cognito_prod_datamart",
    "BQ_TABLE_NAME=clinical_efficacy",
    "BQ_UPLOAD_LOCATION=$GCP_REGION",
    "GCS_BUCKET_NAME=YOUR_BUCKET_NAME_HERE",
    "GCS_PATENTS_PREFIX=patents",
    "ALLOYDB_HOST=YOUR_ALLOYDB_PRIVATE_IP",
    "ALLOYDB_PASSWORD=YOUR_ALLOYDB_PASSWORD",
    "ALLOYDB_USER=postgres",
    "ALLOYDB_DB=postgres",
    "GOOGLE_GENAI_API_KEY=YOUR_GEMINI_KEY"
) -join ","
# ═══════════════════════════════════════════════════════════════

$IMAGE = "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$REPO_NAME/loe-pipeline"

# ── Preflight check ──────────────────────────────────────────
if (-not (Test-Path "service-account.json")) {
    Write-Host "ERROR: service-account.json not found in current directory." -ForegroundColor Red
    Write-Host "Copy your GCP service account JSON here and name it service-account.json"
    exit 1
}

if (-not (Test-Path "Dockerfile")) {
    Write-Host "ERROR: Dockerfile not found. Run this from the pipeline folder." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  LOE Pipeline Deploy"
Write-Host "  Project:  $GCP_PROJECT"
Write-Host "  Region:   $GCP_REGION"
Write-Host "  Image:    $IMAGE"
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan

# ── 1. Enable APIs ───────────────────────────────────────────
Write-Host "`n[1/6] Enabling APIs..." -ForegroundColor Yellow
gcloud services enable `
    artifactregistry.googleapis.com `
    run.googleapis.com `
    cloudbuild.googleapis.com `
    bigquery.googleapis.com `
    aiplatform.googleapis.com `
    storage.googleapis.com `
    vpcaccess.googleapis.com `
    --project=$GCP_PROJECT --quiet

# ── 2. Create Artifact Registry ─────────────────────────────
Write-Host "`n[2/6] Ensuring Artifact Registry repo..." -ForegroundColor Yellow
$null = gcloud artifacts repositories describe $REPO_NAME `
    --location=$GCP_REGION --project=$GCP_PROJECT 2>$null
if ($LASTEXITCODE -ne 0) {
    gcloud artifacts repositories create $REPO_NAME `
        --repository-format=docker `
        --location=$GCP_REGION `
        --project=$GCP_PROJECT
}

# ── 3. Build image via Cloud Build ──────────────────────────
Write-Host "`n[3/6] Building container via Cloud Build..." -ForegroundColor Yellow
Write-Host "       (your code + service-account.json are uploaded to GCP for building)" -ForegroundColor Gray
gcloud builds submit `
    --tag "${IMAGE}:latest" `
    --project=$GCP_PROJECT `
    --region=$GCP_REGION `
    --timeout=1800

# ── 4. Ensure VPC connector ─────────────────────────────────
Write-Host "`n[4/6] Ensuring VPC connector..." -ForegroundColor Yellow
$null = gcloud compute networks vpc-access connectors describe $VPC_CONNECTOR `
    --region=$GCP_REGION --project=$GCP_PROJECT 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Creating VPC connector '$VPC_CONNECTOR'..."
    gcloud compute networks vpc-access connectors create $VPC_CONNECTOR `
        --region=$GCP_REGION `
        --network=default `
        --range="10.8.0.0/28" `
        --project=$GCP_PROJECT
}

# ── 5. Deploy Job 1: loe-patents (10 parallel tasks) ────────
Write-Host "`n[5/6] Deploying Job 1: loe-patents (10 parallel tasks)..." -ForegroundColor Yellow
$null = gcloud run jobs describe loe-patents --region=$GCP_REGION --project=$GCP_PROJECT 2>$null
$verb = if ($LASTEXITCODE -eq 0) { "update" } else { "create" }

gcloud run jobs $verb loe-patents `
    --image="${IMAGE}:latest" `
    --region=$GCP_REGION `
    --project=$GCP_PROJECT `
    --tasks=10 `
    --parallelism=10 `
    --memory=4Gi `
    --cpu=2 `
    --task-timeout=3600 `
    --max-retries=1 `
    --vpc-connector=$VPC_CONNECTOR `
    --set-env-vars="$ENV_VARS" `
    --args="--mode,patents"

# ── 6. Deploy Job 2: loe-forecast (1 task) ──────────────────
Write-Host "`n[6/6] Deploying Job 2: loe-forecast (single task)..." -ForegroundColor Yellow
$null = gcloud run jobs describe loe-forecast --region=$GCP_REGION --project=$GCP_PROJECT 2>$null
$verb = if ($LASTEXITCODE -eq 0) { "update" } else { "create" }

gcloud run jobs $verb loe-forecast `
    --image="${IMAGE}:latest" `
    --region=$GCP_REGION `
    --project=$GCP_PROJECT `
    --tasks=1 `
    --memory=4Gi `
    --cpu=2 `
    --task-timeout=7200 `
    --max-retries=1 `
    --vpc-connector=$VPC_CONNECTOR `
    --set-env-vars="$ENV_VARS" `
    --args="--mode,forecast"

# ── Done ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  DEPLOYED SUCCESSFULLY" -ForegroundColor Green
Write-Host ""
Write-Host "  STEP A — Run patent processing (10 parallel tasks):"
Write-Host "    gcloud run jobs execute loe-patents --region=$GCP_REGION --project=$GCP_PROJECT" -ForegroundColor White
Write-Host ""
Write-Host "  STEP B — After Job 1 finishes, run forecast + merge:"
Write-Host "    gcloud run jobs execute loe-forecast --region=$GCP_REGION --project=$GCP_PROJECT" -ForegroundColor White
Write-Host ""
Write-Host "  VIEW LOGS:"
Write-Host "    gcloud run jobs executions list --job=loe-patents --region=$GCP_REGION"
Write-Host "    gcloud run jobs executions list --job=loe-forecast --region=$GCP_REGION"
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Green
