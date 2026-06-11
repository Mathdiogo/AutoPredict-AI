# ============================================================
# Teste Rápido - Groq e Novos Modelos
# ============================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Testando Groq + Novos Modelos" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1] Aguardando API..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

try {
    Write-Host "[2] Consultando modelos..." -ForegroundColor Yellow
    $response = Invoke-RestMethod -Uri "http://localhost:8000/models" -TimeoutSec 10
    
    Write-Host ""
    Write-Host "✅ Total: $($response.total_available) modelos" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "🖥️  MODELOS LOCAIS:" -ForegroundColor Cyan
    $response.models | Where-Object {$_.local -eq $true} | ForEach-Object {
        Write-Host "   ✅ $($_.display_name)" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "☁️   MODELOS GROQ:" -ForegroundColor Cyan
    $response.models | Where-Object {$_.provider -eq "groq"} | ForEach-Object {
        $icon = if ($_.requires_key) { "⚠️ " } else { "✅" }
        Write-Host "   $icon $($_.display_name)" -ForegroundColor $(if ($_.requires_key) {"Yellow"} else {"Green"})
    }
    
    Write-Host ""
    if ($response.models | Where-Object {$_.provider -eq "groq" -and $_.requires_key -eq $false}) {
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "🎉 GROQ CONFIGURADO COM SUCESSO!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
    } else {
        Write-Host "========================================" -ForegroundColor Yellow
        Write-Host "⚠️  Groq ainda não configurado" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "Agora:" -ForegroundColor Cyan
    Write-Host "1. Acesse: http://localhost:7860" -ForegroundColor White
    Write-Host "2. Pressione Ctrl+Shift+R (force reload)" -ForegroundColor White
    Write-Host "3. Veja os modelos no dropdown!" -ForegroundColor White
    Write-Host ""
    
} catch {
    Write-Host ""
    Write-Host "❌ API ainda não respondeu" -ForegroundColor Red
    Write-Host "Aguarde mais 10s e rode novamente" -ForegroundColor Yellow
    Write-Host ""
}
