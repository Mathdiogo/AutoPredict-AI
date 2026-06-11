# ============================================================
# Script para indexar informações do sistema no Milvus
# ============================================================
# Este script adiciona o documento SYSTEM_INFO.md ao Milvus
# para que o sistema possa responder perguntas sobre si mesmo.
# ============================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Indexando Informações do Sistema" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Ativa o ambiente virtual se existir
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "[INFO] Ativando ambiente virtual..." -ForegroundColor Yellow
    & .venv\Scripts\Activate.ps1
}

# Verifica se os serviços estão rodando
Write-Host "[INFO] Verificando serviços..." -ForegroundColor Yellow
$milvusStatus = docker compose ps milvus -q
$minioStatus = docker compose ps minio -q

if (-not $milvusStatus) {
    Write-Host "[ERRO] Milvus não está rodando!" -ForegroundColor Red
    Write-Host "       Execute: docker compose up -d milvus" -ForegroundColor Red
    exit 1
}

if (-not $minioStatus) {
    Write-Host "[AVISO] MinIO não está rodando. Pode ser necessário." -ForegroundColor Yellow
}

Write-Host "[OK] Serviços verificados!" -ForegroundColor Green
Write-Host ""

# Executa o script de indexação
Write-Host "[INFO] Indexando documento SYSTEM_INFO.md..." -ForegroundColor Yellow
python src\data_pipeline\index_system_info.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  ✅ Indexação concluída com sucesso!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Agora o sistema pode responder perguntas como:" -ForegroundColor Cyan
    Write-Host "  • Quem é você?" -ForegroundColor White
    Write-Host "  • Quais modelos foram treinados?" -ForegroundColor White
    Write-Host "  • Quais estratégias de governança você usa?" -ForegroundColor White
    Write-Host "  • Quais datasets você utiliza?" -ForegroundColor White
    Write-Host "  • Como funciona o pooling de modelos?" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  ❌ Erro na indexação!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    exit 1
}
