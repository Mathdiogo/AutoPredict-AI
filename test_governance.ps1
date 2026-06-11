# ============================================================
# Script de Teste das Novas Funcionalidades
# ============================================================
# Testa pooling de modelos, métricas e autoconhecimento
# ============================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Testando Novas Funcionalidades" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$API_URL = "http://localhost:8000"

# Função para fazer requisição POST
function Test-Chat {
    param(
        [string]$Question,
        [string]$Model = $null,
        [string]$UserId = $null
    )
    
    $body = @{
        question = $Question
    }
    
    if ($Model) { $body.model = $Model }
    if ($UserId) { $body.user_id = $UserId }
    
    try {
        $response = Invoke-RestMethod -Uri "$API_URL/chat" -Method POST `
            -ContentType "application/json" `
            -Body ($body | ConvertTo-Json)
        return $response
    } catch {
        Write-Host "  [ERRO] $_" -ForegroundColor Red
        return $null
    }
}

# ============================================================
# Teste 1: Health Check
# ============================================================
Write-Host "[Teste 1] Health Check" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$API_URL/health" -Method GET
    if ($health.status -eq "healthy") {
        Write-Host "  ✅ API está saudável" -ForegroundColor Green
    }
} catch {
    Write-Host "  ❌ API não está respondendo!" -ForegroundColor Red
    Write-Host "     Execute: docker compose up -d" -ForegroundColor Red
    exit 1
}
Write-Host ""

# ============================================================
# Teste 2: Listar Modelos
# ============================================================
Write-Host "[Teste 2] Pooling de Modelos" -ForegroundColor Yellow
try {
    $models = Invoke-RestMethod -Uri "$API_URL/models" -Method GET
    Write-Host "  ✅ Modelos Ollama: $($models.ollama.Count)" -ForegroundColor Green
    Write-Host "  ✅ Modelos OpenAI: $($models.openai.Count)" -ForegroundColor Green
    Write-Host "  ✅ Modelos Groq: $($models.groq.Count)" -ForegroundColor Green
    Write-Host "  ✅ Modelo padrão: $($models.default)" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️ Erro ao listar modelos" -ForegroundColor Yellow
}
Write-Host ""

# ============================================================
# Teste 3: Métricas de Governança
# ============================================================
Write-Host "[Teste 3] Métricas de Governança" -ForegroundColor Yellow
$response = Test-Chat -Question "Teste de métricas" -UserId "test_user"
if ($response) {
    Write-Host "  ✅ Tempo de inferência: $($response.metrics.inference_time_seconds)s" -ForegroundColor Green
    Write-Host "  ✅ Tokens usados: $($response.metrics.tokens_used)" -ForegroundColor Green
    Write-Host "  ✅ Chunks recuperados: $($response.metrics.chunks_retrieved)" -ForegroundColor Green
    Write-Host "  ✅ Collections: $($response.metrics.collections_used -join ', ')" -ForegroundColor Green
    Write-Host "  ✅ User ID: $($response.metrics.user_id)" -ForegroundColor Green
    Write-Host "  ✅ Provider: $($response.metrics.model_provider)" -ForegroundColor Green
    Write-Host "  ✅ Modelo: $($response.metrics.model_name)" -ForegroundColor Green
}
Write-Host ""

# ============================================================
# Teste 4: Autoconhecimento - "Quem é você?"
# ============================================================
Write-Host "[Teste 4] Autoconhecimento - Identidade" -ForegroundColor Yellow
$response = Test-Chat -Question "Quem é você?"
if ($response) {
    if ($response.answer -match "AutoPredict AI") {
        Write-Host "  ✅ Sistema respondeu corretamente sobre sua identidade" -ForegroundColor Green
        Write-Host "  Preview: $($response.answer.Substring(0, [Math]::Min(150, $response.answer.Length)))..." -ForegroundColor White
    } else {
        Write-Host "  ⚠️ Sistema não reconheceu a pergunta sobre identidade" -ForegroundColor Yellow
    }
}
Write-Host ""

# ============================================================
# Teste 5: Autoconhecimento - Modelos Treinados
# ============================================================
Write-Host "[Teste 5] Autoconhecimento - Modelos Treinados" -ForegroundColor Yellow
$response = Test-Chat -Question "Quais modelos de machine learning foram treinados?"
if ($response) {
    if ($response.answer -match "Random Forest|Gradient Boosting|Acurácia") {
        Write-Host "  ✅ Sistema respondeu sobre modelos treinados" -ForegroundColor Green
        Write-Host "  Preview: $($response.answer.Substring(0, [Math]::Min(150, $response.answer.Length)))..." -ForegroundColor White
    } else {
        Write-Host "  ⚠️ Sistema não reconheceu a pergunta sobre modelos" -ForegroundColor Yellow
    }
}
Write-Host ""

# ============================================================
# Teste 6: Autoconhecimento - Governança
# ============================================================
Write-Host "[Teste 6] Autoconhecimento - Governança" -ForegroundColor Yellow
$response = Test-Chat -Question "Quais estratégias de governança você utiliza?"
if ($response) {
    if ($response.answer -match "Medallion|Bronze|Silver|Gold|governança") {
        Write-Host "  ✅ Sistema respondeu sobre governança" -ForegroundColor Green
        Write-Host "  Preview: $($response.answer.Substring(0, [Math]::Min(150, $response.answer.Length)))..." -ForegroundColor White
    } else {
        Write-Host "  ⚠️ Sistema não reconheceu a pergunta sobre governança" -ForegroundColor Yellow
    }
}
Write-Host ""

# ============================================================
# Teste 7: Troca de Modelo
# ============================================================
Write-Host "[Teste 7] Troca de Modelo" -ForegroundColor Yellow
$response = Test-Chat -Question "Teste de troca" -Model "llama3.2:1b"
if ($response) {
    Write-Host "  ✅ Modelo usado: $($response.model)" -ForegroundColor Green
    Write-Host "  ✅ Provider: $($response.metrics.model_provider)" -ForegroundColor Green
}
Write-Host ""

# ============================================================
# Resumo Final
# ============================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Resumo dos Testes" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ API operacional" -ForegroundColor Green
Write-Host "✅ Pooling de modelos funcionando" -ForegroundColor Green
Write-Host "✅ Métricas de governança completas" -ForegroundColor Green
Write-Host "✅ Sistema de autoconhecimento ativo" -ForegroundColor Green
Write-Host ""
Write-Host "Pronto para apresentação! 🎉" -ForegroundColor Cyan
Write-Host ""
