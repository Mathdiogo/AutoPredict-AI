# ============================================================
# Script para Testar Modelos Disponíveis
# ============================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Testando Modelos Disponíveis" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1] Aguardando API iniciar..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

try {
    Write-Host "[2] Buscando modelos..." -ForegroundColor Yellow
    $response = Invoke-RestMethod -Uri "http://localhost:8000/models" -TimeoutSec 10
    
    Write-Host ""
    Write-Host "✅ Total de modelos: $($response.total_available)" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "📋 Modelos Ollama (Local): $($response.by_provider.ollama)" -ForegroundColor Cyan
    Write-Host "☁️  Modelos Groq (Cloud): $($response.by_provider.groq)" -ForegroundColor Cyan
    Write-Host "🤖 Modelos OpenAI: $($response.by_provider.openai)" -ForegroundColor Cyan
    Write-Host "🧠 Modelos Anthropic: $($response.by_provider.anthropic)" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "📝 Lista Completa de Modelos:" -ForegroundColor Yellow
    Write-Host ""
    
    foreach ($model in $response.models) {
        $icon = if ($model.local) { "🖥️ " } else { "☁️  " }
        Write-Host "$icon $($model.display_name)" -ForegroundColor White
    }
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ Tudo OK! Modelos carregados!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Agora:" -ForegroundColor Cyan
    Write-Host "1. Abra: http://localhost:7860" -ForegroundColor White
    Write-Host "2. Pressione F5 para recarregar a página" -ForegroundColor White
    Write-Host "3. Click no dropdown 'Modelo LLM'" -ForegroundColor White
    Write-Host "4. Veja todos os $($response.total_available) modelos!" -ForegroundColor White
    Write-Host ""
    
} catch {
    Write-Host ""
    Write-Host "❌ Erro ao buscar modelos" -ForegroundColor Red
    Write-Host ""
    Write-Host "Tente:" -ForegroundColor Yellow
    Write-Host "1. Aguardar mais 10 segundos" -ForegroundColor White
    Write-Host "2. Rodar este script novamente" -ForegroundColor White
    Write-Host ""
    Write-Host "Ou verifique os containers:" -ForegroundColor Yellow
    Write-Host "docker compose ps" -ForegroundColor White
    Write-Host ""
}
