# Script para registrar documentação de governança no MLflow
# Uso: .\register_governance.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Registrando Governança no MLflow" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar se Docker está rodando
Write-Host "1. Verificando Docker..." -ForegroundColor Yellow
$dockerRunning = docker ps 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ❌ Docker não está rodando!" -ForegroundColor Red
    Write-Host "   Inicie o Docker Desktop e tente novamente." -ForegroundColor Red
    exit 1
}
Write-Host "   ✅ Docker rodando" -ForegroundColor Green

# Verificar se MLflow está rodando
Write-Host "2. Verificando MLflow..." -ForegroundColor Yellow
$mlflowContainer = docker ps --filter "name=autopredict-mlflow" --format "{{.Names}}"
if ($mlflowContainer -ne "autopredict-mlflow") {
    Write-Host "   ❌ MLflow não está rodando!" -ForegroundColor Red
    Write-Host "   Execute: docker compose up -d mlflow" -ForegroundColor Red
    exit 1
}
Write-Host "   ✅ MLflow está rodando" -ForegroundColor Green

# Verificar se API está rodando
Write-Host "3. Verificando API..." -ForegroundColor Yellow
$apiContainer = docker ps --filter "name=autopredict-api" --format "{{.Names}}"
if ($apiContainer -ne "autopredict-api") {
    Write-Host "   ❌ API não está rodando!" -ForegroundColor Red
    Write-Host "   Execute: docker compose up -d api" -ForegroundColor Red
    exit 1
}
Write-Host "   ✅ API está rodando" -ForegroundColor Green

# Executar script de registro (MLflow)
Write-Host ""
Write-Host "4. Registrando documentação no MLflow..." -ForegroundColor Yellow
Write-Host ""

docker exec -w /app autopredict-api python -m src.data_pipeline.register_governance_docs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Erro ao registrar documentação no MLflow" -ForegroundColor Red
    exit 1
}

# Upload para MinIO (bucket governance)
Write-Host ""
Write-Host "5. Enviando governança para o MinIO..." -ForegroundColor Yellow
Write-Host ""

docker exec -w /app autopredict-api python -m src.data_pipeline.upload_governance_to_minio

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  ✅ CONCLUÍDO COM SUCESSO!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 MLflow UI:" -ForegroundColor Cyan
    Write-Host "   http://localhost:5001" -ForegroundColor White
    Write-Host "   Experimento: AutoPredict-Governance" -ForegroundColor White
    Write-Host ""
    Write-Host "🗄️  MinIO Console:" -ForegroundColor Cyan
    Write-Host "   http://localhost:9001" -ForegroundColor White
    Write-Host "   Bucket: governance → pasta governance/" -ForegroundColor White
    Write-Host "   Login: minioadmin / minioadmin123" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Erro ao registrar documentação" -ForegroundColor Red
    Write-Host "Verifique os logs acima para mais detalhes" -ForegroundColor Red
    exit 1
}
