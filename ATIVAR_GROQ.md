# 🚀 Como Ativar Groq (GRATUITO e SUPER Rápido!)

O Groq é uma API de IA **100% gratuita** que roda modelos como **Llama 3.1 70B** (muito melhor que o llama 3.2 1b local) em servidores super potentes. **É MUITO mais rápido** que rodar local na sua máquina!

---

## ⚡ Por que usar Groq?

- ✅ **100% GRATUITO** (sem cartão de crédito)
- ✅ **EXTREMAMENTE RÁPIDO** (hardware dedicado)
- ✅ **Llama 3.1 70B** (modelo ENORME, muito melhor que o 1b local)
- ✅ **Mixtral 8x7B, Gemma 7B** (outros modelos disponíveis)
- ✅ Respostas em **2-3 segundos** vs 20-30 segundos local

---

## 📝 Passo a Passo (5 minutos)

### 1. Criar Conta Gratuita

1. Acesse: **https://console.groq.com**
2. Click em "Sign Up" (ou "Get Started")
3. Use seu email ou login com Google/GitHub
4. Confirme o email

### 2. Criar API Key

1. No dashboard do Groq, click em **"API Keys"** (menu lateral)
2. Click em **"Create API Key"**
3. Dê um nome: `AutoPredict-AI`
4. Click em **"Create"**
5. **COPIE A CHAVE** (começa com `gsk_...`)
   - ⚠️ **IMPORTANTE:** Copie AGORA, não vai poder ver depois!

### 3. Adicionar ao Projeto

1. Abra o arquivo `.env` na raiz do projeto
2. Encontre a linha:
   ```
   GROQ_API_KEY=
   ```
3. Cole sua chave:
   ```
   GROQ_API_KEY=gsk_sua_chave_aqui
   ```
4. Salve o arquivo

### 4. Reiniciar Serviços

```powershell
docker compose restart api frontend
```

Aguarde ~10 segundos.

### 5. Testar!

1. Abra o frontend: http://localhost:7860
2. No dropdown "Modelo LLM", você verá:
   - Llama 3.2 1B (Ollama - Local)
   - **Llama 3.1 70B (Groq - Grátis) ✨**
   - **Mixtral 8x7B (Groq - Grátis) ✨**
   - **Gemma 7B (Groq - Grátis) ✨**
   - GPT-4, Claude (se configurar - pagos)

3. Selecione **"Llama 3.1 70B (Groq)"**
4. Faça uma pergunta
5. Veja a **VELOCIDADE**! 🚀

---

## 🎯 Comparação de Velocidade

| Modelo | Tempo de Resposta | Qualidade |
|--------|-------------------|-----------|
| Llama 3.2 1B (Local) | 20-30s | Básica |
| Llama 3.2 3B (Local) | 40-60s | Boa |
| **Llama 3.1 70B (Groq)** | **2-3s** | **Excelente** ✨ |
| Mixtral 8x7B (Groq) | 2-4s | Muito Boa |

---

## 🔒 Segurança

- Suas perguntas são enviadas para os servidores do Groq
- Eles **NÃO armazenam** suas conversas (política de privacidade)
- É seguro para demos e projetos acadêmicos
- Para dados sensíveis, use modelos locais

---

## 💰 Limites Gratuitos

O Groq tem limites generosos:
- **6.000 requisições por minuto** (RPM)
- **600.000 tokens por minuto** (TPM)

**Para o seu projeto:** Ilimitado para fins práticos! 🎉

---

## 🎬 Para o Vídeo Pitch

**RECOMENDAÇÃO:** Use o Groq na demonstração!

Vantagens:
- ✅ Respostas **instantâneas** (impressiona!)
- ✅ Qualidade de resposta **muito superior**
- ✅ Não trava durante a gravação
- ✅ Mostra integração com cloud

Fale no vídeo:
> "O sistema suporta múltiplos providers: modelos locais via Ollama e também integração com APIs cloud como Groq, OpenAI e Anthropic. Para esta demo, vou usar o Llama 3.1 70B via Groq, que é gratuito e extremamente rápido."

---

## 🆘 Problemas?

### "Groq API key inválida"
- Verifique se copiou a chave completa (começa com `gsk_`)
- Verifique se não tem espaços extras no `.env`
- Reinicie os containers: `docker compose restart api frontend`

### "Não aparece no dropdown"
- Reinicie: `docker compose restart api frontend`
- Verifique o `.env` (deve ter `GROQ_API_KEY=gsk_...`)
- Teste a API: http://localhost:8000/models

### "Rate limit exceeded"
- Improvável (limite é MUITO alto)
- Se acontecer, use modelo local temporariamente

---

## 📚 Mais Info

- Documentação: https://console.groq.com/docs
- Models: https://console.groq.com/docs/models
- Playground: https://console.groq.com/playground

---

**Configure em 5 minutos e tenha respostas MUITO mais rápidas!** 🚀
