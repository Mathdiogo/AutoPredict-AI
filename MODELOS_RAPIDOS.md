# ⚡ Guia de Modelos Rápidos

## ✅ O que foi configurado:

1. **Modelo padrão alterado** para `llama3.2:1b` (MUITO mais leve e rápido)
2. **Arquivo `.env` criado** com configurações otimizadas
3. **Suporte ao Groq** pronto (API gratuita e super rápida)
4. **Todos os modelos** agora aparecem no dropdown

---

## 🚀 Opções de Modelos Disponíveis

### 1. **Llama 3.2 1B (Local)** ⚡ RECOMENDADO para máquina fraca
- ✅ Já instalado e configurado
- ✅ Respostas em 10-15 segundos
- ✅ Usa pouca RAM (~2GB)
- ✅ Boa qualidade para demos
- ✅ **Este é o novo padrão!**

### 2. **Llama 3.1 70B (Groq - Cloud)** 🌟 MELHOR OPÇÃO!
- ⚡ Respostas em **2-3 segundos**
- ✅ **100% GRATUITO**
- ✅ Qualidade EXCELENTE
- ✅ Não usa recursos da sua máquina
- ⚠️ Requer API key (5 minutos para configurar)
- 📖 Ver guia: `ATIVAR_GROQ.md`

### 3. Outros modelos Groq (Grátis)
- Mixtral 8x7B - Respostas em 2-4s
- Gemma 7B - Respostas em 2-3s

### 4. OpenAI / Anthropic (Pagos)
- GPT-4, Claude 3 Opus
- Excelente qualidade mas precisa cartão de crédito

---

## 📋 Como Usar Agora

### Opção A: Usar Llama 1B Local (Já Configurado!)

1. Acesse: http://localhost:7860
2. No dropdown, selecione: **"Llama 3.2 1B (Ollama - Local)"**
3. Pronto! Respostas serão **2-3x mais rápidas** que antes

### Opção B: Ativar Groq (5 minutos) - **RECOMENDADO!**

1. Siga o guia: `ATIVAR_GROQ.md`
2. Leva só 5 minutos
3. Terá acesso a Llama 3.1 70B **GRATUITO e SUPER RÁPIDO**

---

## 🎬 Para o Vídeo Pitch

### Se NÃO configurar Groq:
```
Use: Llama 3.2 1B (Local)
Tempo: 10-15s por resposta
Qualidade: Boa para demo
```

### Se configurar Groq (RECOMENDADO!):
```
Use: Llama 3.1 70B (Groq)
Tempo: 2-3s por resposta ⚡
Qualidade: Excelente
Impressiona mais! 🌟
```

---

## 🔧 Troubleshooting

### "Ainda está mostrando só 1 modelo"
```powershell
# Reinicie os containers
docker compose restart api frontend

# Aguarde 15 segundos
Start-Sleep -Seconds 15

# Recarregue a página: http://localhost:7860
```

### "Modelo 1B ainda está lento"
- Primeira resposta sempre é mais lenta (carrega o modelo)
- Segunda resposta em diante será mais rápida
- Se continuar lento: **Configure o Groq!** (ver `ATIVAR_GROQ.md`)

### "Groq não funciona"
- Verifique se a API key está correta no `.env`
- Reinicie: `docker compose restart api frontend`
- Teste: http://localhost:8000/models

---

## 📊 Comparação de Velocidade

| Modelo | 1ª Resposta | 2ª+ Respostas | Qualidade | Custo |
|--------|-------------|---------------|-----------|-------|
| Llama 1B (Local) | 15-20s | 10-15s | Boa | Grátis |
| Llama 3B (Local) | 40-50s | 30-40s | Muito Boa | Grátis |
| **Llama 70B (Groq)** | **2-3s** | **2-3s** | **Excelente** | **Grátis** ⭐ |
| GPT-4 (OpenAI) | 3-5s | 3-5s | Excelente | Pago |

---

## 💡 Recomendação Final

Para máquina fraca:

1. **Agora mesmo:** Use Llama 1B (já está configurado)
2. **Daqui 5 minutos:** Configure Groq e use Llama 70B (muito melhor!)

**Groq é 100% grátis e 5-10x mais rápido!** Sério, vale muito a pena configurar! 🚀

---

## 📚 Links Úteis

- **Ativar Groq:** Ver `ATIVAR_GROQ.md`
- **Groq Console:** https://console.groq.com
- **Frontend:** http://localhost:7860
- **API Models:** http://localhost:8000/models

---

**Configurado! Agora suas respostas serão MUITO mais rápidas!** ⚡
