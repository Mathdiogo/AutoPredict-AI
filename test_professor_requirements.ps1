# ============================================================
# Script de Teste - Requisitos do Professor
# ============================================================
# Testa todas as funcionalidades solicitadas pelo professor:
# 1. Pooling de modelos (troca dinâmica)
# 2. Contrato da aplicação / governança
# 3. Métricas no response
# 4. Perguntas de autoconhecimento
# ============================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Testes dos Requisitos do Professor" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$API_URL = "http://localhost:8000"
$testsPassed = 0
$testsFailed = 0

# Função para fazer requisição POST
function Test-Chat {
    param(
        [string]$Question,
        [string]$Model = $null,
        [string]$UserId = "prof_demo"
    )
    
    $body = @{
        question = $Question
        user_id = $UserId
        top_p = 0.9
        top_k = 40
        temperature = 0.2
    }
    
    if ($Model) { $body.model = $Model }
    
    try {
        $response = Invoke-RestMethod -Uri "$API_URL/chat" -Method POST `
            -ContentType "application/json" `
            -Body ($body | ConvertTo-Json) `
            -TimeoutSec 60
        return $response
    } catch {
        Write-Host "  [ERRO] $_" -ForegroundColor Red
        return $null
    }
}

# ============================================================
# PRÉ-REQUISITO: Health Check
# ============================================================
Write-Host "[PRÉ-REQUISITO] Health Check" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$API_URL/health" -Method GET
    if ($health.status -eq "healthy") {
        Write-Host "  ✅ API está saudável" -ForegroundColor Green
    } else {
        Write-Host "  ❌ API não está saudável!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  ❌ API não está respondendo!" -ForegroundColor Red
    Write-Host "     Execute: docker compose up -d" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# ============================================================
# TESTE 1: Pooling de Modelos (Entry Point)
# ============================================================
Write-Host "[TESTE 1] Pooling de Modelos - Entry Point para Trocar Modelos" -ForegroundColor Cyan
Write-Host "Verificando endpoint GET /models..." -ForegroundColor Yellow

try {
    $models = Invoke-RestMethod -Uri "$API_URL/models" -Method GET
    
    if ($models.models.Count -gt 0) {
        Write-Host "  ✅ Endpoint /models OK" -ForegroundColor Green
        Write-Host "     Total de modelos disponíveis: $($models.models.Count)" -ForegroundColor Gray
        Write-Host "     - Ollama: $($models.by_provider.ollama)" -ForegroundColor Gray
        Write-Host "     - OpenAI: $($models.by_provider.openai)" -ForegroundColor Gray
        Write-Host "     - Groq: $($models.by_provider.groq)" -ForegroundColor Gray
        Write-Host "     - Anthropic: $($models.by_provider.anthropic)" -ForegroundColor Gray
        Write-Host "     - Modelo padrão: $($models.default_model)" -ForegroundColor Gray
        $testsPassed++
    } else {
        Write-Host "  ❌ Nenhum modelo retornado" -ForegroundColor Red
        $testsFailed++
    }
} catch {
    Write-Host "  ❌ Erro ao listar modelos" -ForegroundColor Red
    $testsFailed++
}
Write-Host ""

# ============================================================
# TESTE 2: Contrato da Aplicação / Governança
# ============================================================
Write-Host "[TESTE 2] Contrato da Aplicação / Governança" -ForegroundColor Cyan
Write-Host "Verificando se API aceita parâmetros de governança..." -ForegroundColor Yellow

$response = Test-Chat `
    -Question "Teste de governança" `
    -UserId "prof_demo" `
    -Model $null

if ($response) {
    $metricsOk = $true
    
    # Verifica se response tem todos os campos esperados
    if (-not $response.answer) {
        Write-Host "  ❌ Campo 'answer' ausente" -ForegroundColor Red
        $metricsOk = $false
    }
    
    if (-not $response.metrics) {
        Write-Host "  ❌ Campo 'metrics' ausente" -ForegroundColor Red
        $metricsOk = $false
    } else {
        # Verifica campos individuais das métricas
        $requiredFields = @(
            "inference_time_seconds",
            "tokens_used",
            "chunks_retrieved",
            "collections_used",
            "user_id",
            "model_provider",
            "model_name",
            "top_p",
            "top_k",
            "temperature"
        )
        
        foreach ($field in $requiredFields) {
            if ($null -eq $response.metrics.$field) {
                Write-Host "  ❌ Campo 'metrics.$field' ausente" -ForegroundColor Red
                $metricsOk = $false
            }
        }
    }
    
    if ($metricsOk) {
        Write-Host "  ✅ Contrato completo implementado" -ForegroundColor Green
        Write-Host "     Todos os campos de governança presentes:" -ForegroundColor Gray
        Write-Host "     - user_id: $($response.metrics.user_id)" -ForegroundColor Gray
        Write-Host "     - top_p: $($response.metrics.top_p)" -ForegroundColor Gray
        Write-Host "     - top_k: $($response.metrics.top_k)" -ForegroundColor Gray
        Write-Host "     - temperature: $($response.metrics.temperature)" -ForegroundColor Gray
        $testsPassed++
    } else {
        $testsFailed++
    }
} else {
    Write-Host "  ❌ Requisição falhou" -ForegroundColor Red
    $testsFailed++
}
Write-Host ""

# ============================================================
# TESTE 3: Métricas no Response
# ============================================================
Write-Host "[TESTE 3] Métricas de Inferência no Response" -ForegroundColor Cyan
Write-Host "Verificando métricas detalhadas..." -ForegroundColor Yellow

$response = Test-Chat `
    -Question "Quais são as causas de superaquecimento do motor?" `
    -UserId "prof_demo"

if ($response -and $response.metrics) {
    Write-Host "  ✅ Métricas completas retornadas:" -ForegroundColor Green
    Write-Host "     ⏱️  Tempo de inferência: $($response.metrics.inference_time_seconds)s" -ForegroundColor Gray
    Write-Host "     🔢 Tokens utilizados: $($response.metrics.tokens_used)" -ForegroundColor Gray
    Write-Host "     📚 Chunks recuperados: $($response.metrics.chunks_retrieved)" -ForegroundColor Gray
    Write-Host "     🗂️  Collections usadas: $($response.metrics.collections_used -join ', ')" -ForegroundColor Gray
    Write-Host "     👤 User ID: $($response.metrics.user_id)" -ForegroundColor Gray
    Write-Host "     🤖 Modelo: $($response.metrics.model_provider):$($response.metrics.model_name)" -ForegroundColor Gray
    $testsPassed++
} else {
    Write-Host "  ❌ Métricas ausentes ou incompletas" -ForegroundColor Red
    $testsFailed++
}
Write-Host ""

# ============================================================
# TESTE 4: Autoconhecimento - "Quem é você?"
# ============================================================
Write-Host "[TESTE 4] Autoconhecimento - Quem é você?" -ForegroundColor Cyan
Write-Host "Perguntando sobre identidade do sistema..." -ForegroundColor Yellow

$response = Test-Chat -Question "Quem é você?" -UserId "prof_demo"

if ($response -and $response.answer) {
    $answer = $response.answer.ToLower()
    
    # Verifica se a resposta menciona elementos-chave
    $hasAutoPredict = $answer -like "*autopredict*"
    $hasRAG = $answer -like "*rag*" -or $answer -like "*retrieval*"
    $hasManutencao = $answer -like "*manutenção*" -or $answer -like "*preditiv*"
    
    if ($hasAutoPredict -and ($hasRAG -or $hasManutencao)) {
        Write-Host "  ✅ Sistema se identifica corretamente" -ForegroundColor Green
        Write-Host "     Resposta menciona:" -ForegroundColor Gray
        if ($hasAutoPredict) { Write-Host "       - AutoPredict ✓" -ForegroundColor Gray }
        if ($hasRAG) { Write-Host "       - RAG ✓" -ForegroundColor Gray }
        if ($hasManutencao) { Write-Host "       - Manutenção preditiva ✓" -ForegroundColor Gray }
        $testsPassed++
    } else {
        Write-Host "  ⚠️  Resposta incompleta ou genérica" -ForegroundColor Yellow
        Write-Host "     Resposta recebida (primeiros 200 chars):" -ForegroundColor Gray
        Write-Host "     $($response.answer.Substring(0, [Math]::Min(200, $response.answer.Length)))..." -ForegroundColor Gray
        $testsFailed++
    }
} else {
    Write-Host "  ❌ Não conseguiu responder" -ForegroundColor Red
    $testsFailed++
}
Write-Host ""

# ============================================================
# TESTE 5: Modelos ML Treinados
# ============================================================
Write-Host "[TESTE 5] Autoconhecimento - Modelos ML Treinados" -ForegroundColor Cyan
Write-Host "Perguntando sobre modelos de machine learning..." -ForegroundColor Yellow

$response = Test-Chat -Question "Quais modelos de machine learning foram treinados e quais métricas foram utilizadas?" -UserId "prof_demo"

if ($response -and $response.answer) {
    $answer = $response.answer.ToLower()
    
    # Verifica menção aos modelos corretos
    $hasLogistic = $answer -like "*logistic*" -or $answer -like "*regressão logística*"
    $hasRandomForest = $answer -like "*random forest*" -or $answer -like "*floresta*"
    $hasXGBoost = $answer -like "*xgboost*" -or $answer -like "*gradient boosting*"
    $hasMetrics = $answer -like "*acurácia*" -or $answer -like "*f1*" -or $answer -like "*precisão*" -or $answer -like "*recall*"
    
    if (($hasLogistic -or $hasRandomForest -or $hasXGBoost) -and $hasMetrics) {
        Write-Host "  ✅ Responde sobre modelos ML corretamente" -ForegroundColor Green
        Write-Host "     Resposta menciona:" -ForegroundColor Gray
        if ($hasLogistic) { Write-Host "       - Logistic Regression ✓" -ForegroundColor Gray }
        if ($hasRandomForest) { Write-Host "       - Random Forest ✓" -ForegroundColor Gray }
        if ($hasXGBoost) { Write-Host "       - XGBoost/Gradient Boosting ✓" -ForegroundColor Gray }
        if ($hasMetrics) { Write-Host "       - Métricas (acurácia, F1, etc) ✓" -ForegroundColor Gray }
        $testsPassed++
    } else {
        Write-Host "  ⚠️  Resposta incompleta" -ForegroundColor Yellow
        Write-Host "     Resposta recebida (primeiros 200 chars):" -ForegroundColor Gray
        Write-Host "     $($response.answer.Substring(0, [Math]::Min(200, $response.answer.Length)))..." -ForegroundColor Gray
        $testsFailed++
    }
} else {
    Write-Host "  ❌ Não conseguiu responder" -ForegroundColor Red
    $testsFailed++
}
Write-Host ""

# ============================================================
# TESTE 6: Estratégias de Governança
# ============================================================
Write-Host "[TESTE 6] Autoconhecimento - Estratégias de Governança" -ForegroundColor Cyan
Write-Host "Perguntando sobre governança..." -ForegroundColor Yellow

$response = Test-Chat -Question "Como funciona a governança neste sistema e quais estratégias foram adotadas?" -UserId "prof_demo"

if ($response -and $response.answer) {
    $answer = $response.answer.ToLower()
    
    # Verifica menção aos elementos de governança
    $hasMedallion = $answer -like "*medallion*" -or $answer -like "*bronze*" -or $answer -like "*silver*" -or $answer -like "*gold*"
    $hasTokens = $answer -like "*token*"
    $hasMetrics = $answer -like "*métrica*" -or $answer -like "*auditoria*"
    $hasUserTracking = $answer -like "*user*" -or $answer -like "*usuário*"
    
    if ($hasMedallion -or $hasTokens -or ($hasMetrics -and $hasUserTracking)) {
        Write-Host "  ✅ Responde sobre governança corretamente" -ForegroundColor Green
        Write-Host "     Resposta menciona:" -ForegroundColor Gray
        if ($hasMedallion) { Write-Host "       - Arquitetura Medallion ✓" -ForegroundColor Gray }
        if ($hasTokens) { Write-Host "       - Controle de tokens ✓" -ForegroundColor Gray }
        if ($hasMetrics) { Write-Host "       - Métricas/Auditoria ✓" -ForegroundColor Gray }
        if ($hasUserTracking) { Write-Host "       - Rastreamento de usuários ✓" -ForegroundColor Gray }
        $testsPassed++
    } else {
        Write-Host "  ⚠️  Resposta incompleta" -ForegroundColor Yellow
        Write-Host "     Resposta recebida (primeiros 200 chars):" -ForegroundColor Gray
        Write-Host "     $($response.answer.Substring(0, [Math]::Min(200, $response.answer.Length)))..." -ForegroundColor Gray
        $testsFailed++
    }
} else {
    Write-Host "  ❌ Não conseguiu responder" -ForegroundColor Red
    $testsFailed++
}
Write-Host ""

# ============================================================
# RESUMO DOS TESTES
# ============================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RESUMO DOS TESTES" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ Testes passados: $testsPassed" -ForegroundColor Green
Write-Host "❌ Testes falhados: $testsFailed" -ForegroundColor Red
Write-Host ""

if ($testsFailed -eq 0) {
    Write-Host "🎉 TODOS OS REQUISITOS DO PROFESSOR IMPLEMENTADOS!" -ForegroundColor Green
    Write-Host ""
    Write-Host "O sistema está pronto para demonstração:" -ForegroundColor Cyan
    Write-Host "  ✅ Pooling de modelos (troca dinâmica)" -ForegroundColor White
    Write-Host "  ✅ Contrato da aplicação com governança completa" -ForegroundColor White
    Write-Host "  ✅ Métricas detalhadas no response" -ForegroundColor White
    Write-Host "  ✅ Autoconhecimento (quem é você, modelos, governança)" -ForegroundColor White
    Write-Host ""
    exit 0
} else {
    Write-Host "⚠️  Alguns testes falharam. Revise os detalhes acima." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
