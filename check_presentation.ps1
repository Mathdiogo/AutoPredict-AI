# ✅ Checklist Pré-Apresentação - AutoPredict AI
# Execute este script 5 minutos antes de apresentar

Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  🎯 CHECKLIST PRÉ-APRESENTAÇÃO - AutoPredict AI" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════`n" -ForegroundColor Cyan

$allOk = $true

# ══════════════════════════════════════════════════════════
# 1. DOCKER CONTAINERS
# ══════════════════════════════════════════════════════════
Write-Host "[1/7] 🐳 Verificando Docker Containers..." -ForegroundColor Yellow

$containers = @("autopredict-api", "autopredict-frontend", "autopredict-milvus", "autopredict-postgres", "autopredict-minio", "autopredict-ollama", "autopredict-mlflow")
$running = 0

foreach ($container in $containers) {
    $status = docker inspect -f '{{.State.Running}}' $container 2>$null
    if ($status -eq "true") {
        Write-Host "  ✅ $container" -ForegroundColor Green
        $running++
    } else {
        Write-Host "  ❌ $container (não está rodando!)" -ForegroundColor Red
        $allOk = $false
    }
}

Write-Host "  📊 Status: $running/$($containers.Count) containers rodando`n" -ForegroundColor Cyan

# ══════════════════════════════════════════════════════════
# 2. API REST
# ══════════════════════════════════════════════════════════
Write-Host "[2/7] 🌐 Testando API REST (http://localhost:8000)..." -ForegroundColor Yellow

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✅ API está respondendo" -ForegroundColor Green
        $health = $response.Content | ConvertFrom-Json
        Write-Host "    • Milvus: $($health.milvus)" -ForegroundColor Gray
        Write-Host "    • MinIO: $($health.minio)" -ForegroundColor Gray
        Write-Host "    • Postgres: $($health.postgres)" -ForegroundColor Gray
        Write-Host "    • Ollama: $($health.ollama)" -ForegroundColor Gray
    }
} catch {
    Write-Host "  ❌ API não responde!" -ForegroundColor Red
    Write-Host "    Execute: docker logs autopredict-api --tail 20" -ForegroundColor Yellow
    $allOk = $false
}

Write-Host ""

# ══════════════════════════════════════════════════════════
# 3. FRONTEND GRADIO
# ══════════════════════════════════════════════════════════
Write-Host "[3/7] 🖥️ Testando Frontend (http://localhost:7860)..." -ForegroundColor Yellow

try {
    $response = Invoke-WebRequest -Uri "http://localhost:7860" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✅ Frontend está acessível" -ForegroundColor Green
    }
} catch {
    Write-Host "  ❌ Frontend não responde!" -ForegroundColor Red
    Write-Host "    Execute: docker logs autopredict-frontend --tail 20" -ForegroundColor Yellow
    $allOk = $false
}

Write-Host ""

# ══════════════════════════════════════════════════════════
# 4. OLLAMA MODELS
# ══════════════════════════════════════════════════════════
Write-Host "[4/7] 🤖 Verificando Modelos Ollama..." -ForegroundColor Yellow

try {
    $models = docker exec autopredict-ollama ollama list 2>&1
    if ($models -match "llama") {
        Write-Host "  ✅ Modelos instalados:" -ForegroundColor Green
        $models | Select-String -Pattern "llama|qwen|mistral|phi" | ForEach-Object {
            Write-Host "    • $_" -ForegroundColor Gray
        }
    } else {
        Write-Host "  ⚠️ Nenhum modelo Ollama encontrado!" -ForegroundColor Yellow
        Write-Host "    Execute: docker exec autopredict-ollama ollama pull llama3.2:1b" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ❌ Ollama não responde!" -ForegroundColor Red
    $allOk = $false
}

Write-Host ""

# ══════════════════════════════════════════════════════════
# 5. MILVUS COLLECTIONS
# ══════════════════════════════════════════════════════════
Write-Host "[5/7] 🗄️ Verificando Coleções Milvus..." -ForegroundColor Yellow

try {
    $result = docker exec autopredict-api python -c "from src.database.milvus_client import get_milvus_client; client = get_milvus_client(); print(', '.join(client.list_collections()))" 2>&1
    if ($result -match "vehicle|car|engine") {
        Write-Host "  ✅ Coleções encontradas: $result" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ Coleções não encontradas!" -ForegroundColor Yellow
        Write-Host "    Execute: docker exec autopredict-api python -m src.data_pipeline.run_pipeline" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ❌ Erro ao verificar Milvus!" -ForegroundColor Red
}

Write-Host ""

# ══════════════════════════════════════════════════════════
# 6. MLFLOW
# ══════════════════════════════════════════════════════════
Write-Host "[6/7] 📊 Testando MLflow (http://localhost:5001)..." -ForegroundColor Yellow

try {
    $response = Invoke-WebRequest -Uri "http://localhost:5001" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✅ MLflow está acessível" -ForegroundColor Green
    }
} catch {
    Write-Host "  ⚠️ MLflow não responde (não crítico)" -ForegroundColor Yellow
}

Write-Host ""

# ══════════════════════════════════════════════════════════
# 7. TESTE DE QUERY RAG
# ══════════════════════════════════════════════════════════
Write-Host "[7/7] 🧪 Testando Query RAG..." -ForegroundColor Yellow

try {
    $testQuery = @{
        question = "Quando trocar o óleo?"
        min_score = 0.25
    } | ConvertTo-Json
    
    $response = Invoke-WebRequest -Uri "http://localhost:8000/chat" -Method POST -Body $testQuery -ContentType "application/json" -UseBasicParsing -TimeoutSec 30
    
    if ($response.StatusCode -eq 200) {
        $result = $response.Content | ConvertFrom-Json
        Write-Host "  ✅ RAG está funcionando!" -ForegroundColor Green
        Write-Host "    Resposta (preview): $($result.answer.Substring(0, [Math]::Min(80, $result.answer.Length)))..." -ForegroundColor Gray
        Write-Host "    Fontes retornadas: $($result.sources.Count)" -ForegroundColor Gray
    }
} catch {
    Write-Host "  ❌ RAG não está funcionando!" -ForegroundColor Red
    Write-Host "    Erro: $($_.Exception.Message)" -ForegroundColor Yellow
    $allOk = $false
}

Write-Host ""

# ══════════════════════════════════════════════════════════
# RESULTADO FINAL
# ══════════════════════════════════════════════════════════
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan

if ($allOk) {
    Write-Host "  ✅ TUDO PRONTO PARA APRESENTAÇÃO! 🎉" -ForegroundColor Green
    Write-Host "" 
    Write-Host "  📋 URLs importantes:" -ForegroundColor Cyan
    Write-Host "    • Frontend:  http://localhost:7860" -ForegroundColor White
    Write-Host "    • API Docs:  http://localhost:8000/docs" -ForegroundColor White
    Write-Host "    • MLflow:    http://localhost:5001" -ForegroundColor White
    Write-Host ""
    Write-Host "  💡 Dica: Abra o frontend AGORA e teste uma pergunta!" -ForegroundColor Yellow
} else {
    Write-Host "  ⚠️ ATENÇÃO: Alguns problemas detectados!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  🔧 Solução rápida:" -ForegroundColor Cyan
    Write-Host "    1. Execute: docker compose restart" -ForegroundColor White
    Write-Host "    2. Aguarde 30 segundos" -ForegroundColor White
    Write-Host "    3. Execute este script novamente" -ForegroundColor White
}

Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
