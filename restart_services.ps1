# Script de Restart Rápido - AutoPredict AI
# Execute este script quando Docker voltar

Write-Host "🔄 Reiniciando Serviços AutoPredict AI..." -ForegroundColor Cyan

# 1. Verifica se Docker está rodando
Write-Host "`n[1/4] Verificando Docker..." -ForegroundColor Yellow
docker version > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker não está rodando! Inicie o Docker Desktop primeiro." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Docker operacional" -ForegroundColor Green

# 2. Para containers que possam estar travados
Write-Host "`n[2/4] Parando containers antigos..." -ForegroundColor Yellow
docker compose down 2>&1 | Out-Null
Start-Sleep -Seconds 2
Write-Host "✅ Containers parados" -ForegroundColor Green

# 3. Sobe todos os serviços novamente
Write-Host "`n[3/4] Iniciando todos os serviços..." -ForegroundColor Yellow
docker compose up -d
Start-Sleep -Seconds 10
Write-Host "✅ Containers iniciados" -ForegroundColor Green

# 4. Verifica status
Write-Host "`n[4/4] Verificando status..." -ForegroundColor Yellow
docker compose ps

# Testa endpoints
Write-Host "`n🧪 Testando endpoints..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

Write-Host "  • API:       " -NoNewline
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
    Write-Host "✅ http://localhost:8000" -ForegroundColor Green
} catch {
    Write-Host "❌ Não responde" -ForegroundColor Red
}

Write-Host "  • Frontend:  " -NoNewline
try {
    $response = Invoke-WebRequest -Uri "http://localhost:7860" -UseBasicParsing -TimeoutSec 5
    Write-Host "✅ http://localhost:7860" -ForegroundColor Green
} catch {
    Write-Host "❌ Não responde" -ForegroundColor Red
}

Write-Host "  • MLflow:    " -NoNewline
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5001" -UseBasicParsing -TimeoutSec 5
    Write-Host "✅ http://localhost:5001" -ForegroundColor Green
} catch {
    Write-Host "❌ Não responde" -ForegroundColor Red
}

Write-Host "`n🎉 Pronto! Acesse o frontend em: http://localhost:7860" -ForegroundColor Cyan
Write-Host "📊 Swagger API: http://localhost:8000/docs" -ForegroundColor Cyan
