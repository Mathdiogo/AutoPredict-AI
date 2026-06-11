# ============================================================
# Script para Indexar Informações do Sistema no Milvus
# ============================================================
# Executa o script Python que indexa o SYSTEM_INFO.md
# para que o sistema possa responder perguntas sobre si mesmo
# ============================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Indexando System Info no Milvus" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1] Verificando se API está rodando..." -ForegroundColor Yellow
$apiRunning = $false
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 5
    if ($response.status -eq "healthy") {
        Write-Host "  ✅ API está saudável" -ForegroundColor Green
        $apiRunning = $true
    }
} catch {
    Write-Host "  ❌ API não está respondendo!" -ForegroundColor Red
    Write-Host "     Execute: docker compose up -d" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "[2] Indexando documentação de governança..." -ForegroundColor Yellow

try {
    docker exec autopredict-api python /app/src/data_pipeline/index_system_info.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  ✅ System Info indexado com sucesso!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Agora o sistema pode responder perguntas como:" -ForegroundColor Cyan
        Write-Host "  - Quem é você?" -ForegroundColor White
        Write-Host "  - Quais modelos foram treinados?" -ForegroundColor White
        Write-Host "  - Quais estratégias de governança você usa?" -ForegroundColor White
        Write-Host "  - Mostre as métricas dos modelos de ML" -ForegroundColor White
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "❌ Erro ao indexar System Info" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host ""
    Write-Host "❌ Erro ao executar script: $_" -ForegroundColor Red
    exit 1
}
