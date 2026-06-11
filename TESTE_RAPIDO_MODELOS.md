# 🚀 TESTE RÁPIDO - Novos Modelos e Groq

## ✅ Checklist Rápido

1. **API construída?** ⏳ Aguarde o Docker terminar
2. **Teste automático**:

```powershell
.\testar_groq.ps1
```

3. **O que você DEVE ver**:

```
✅ Total: 13 modelos

🖥️  MODELOS LOCAIS:
   ✅ qwen2.5:0.5b (Ollama - Local)      ⚡⚡⚡ SUPER RÁPIDO!
   ✅ tinyllama:latest (Ollama - Local)   ⚡⚡⚡ SUPER RÁPIDO!
   ✅ llama3.2:1b (Ollama - Local)       ⚡⚡  RÁPIDO
   ✅ phi3:mini (Ollama - Local)         ⚡⚡  RÁPIDO

☁️  MODELOS GROQ:
   ✅ Llama 3.1 70B (Groq - Grátis) ✨    ⚡⚡⚡⚡ ULTRA RÁPIDO!
   ✅ Mixtral 8x7B (Groq - Grátis) ✨
   ✅ Gemma 7B (Groq - Grátis) ✨

🎉 GROQ CONFIGURADO COM SUCESSO!
```

4. **Se Groq ainda mostrar ⚠️**:
   - Espere mais 1 minuto para API inicializar
   - Rode `.\testar_groq.ps1` novamente

## 📱 Testando no Frontend

1. Acesse: http://localhost:7860
2. Pressione `Ctrl + Shift + R` (force reload)
3. Clique no dropdown de modelos
4. Você DEVE ver **13 modelos**!

## 🎯 Modelos Recomendados (do + rápido para + poderoso)

### Para Máquina Fraca (SUA SITUAÇÃO):
1. **qwen2.5:0.5b** 🏆 CAMPEÃO de velocidade!
2. **tinyllama** 🥈 Segundo mais rápido
3. **Llama 3.1 70B (Groq)** 🥇 Se tiver internet boa!

### Para Qualidade Máxima (mais lento):
1. **Llama 3.1 70B (Groq)** - Melhor geral (grátis!)
2. **Mixtral 8x7B (Groq)** - Bom equilíbrio
3. **llama3.2:1b** - Melhor local

## 🔧 Se algo der errado:

```powershell
# 1. Ver logs da API
docker compose logs api --tail=20

# 2. Reiniciar tudo
docker compose restart api frontend

# 3. Aguardar 15s e testar
Start-Sleep -Seconds 15
.\testar_groq.ps1
```

## 📝 Próximos Passos

Quando estiver funcionando:
1. ✅ Teste os modelos no chat
2. ✅ Compare velocidades
3. ✅ Escolha seu favorito
4. 🎥 Pronto para gravar o pitch!
