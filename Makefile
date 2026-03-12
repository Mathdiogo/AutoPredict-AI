# ============================================================
# AutoPredict AI - Makefile
# ============================================================
# Uso (requer make instalado via Git Bash, WSL ou Chocolatey):
#
#   make setup      -> Primeira configuração (cria .env, sobe tudo, baixa modelo)
#   make up         -> Sobe todos os serviços
#   make down       -> Para todos os serviços
#   make logs       -> Ver logs em tempo real
#   make ingest     -> Roda o pipeline de ingestão dos datasets
#   make clean      -> Remove tudo (APAGA OS DADOS!)
# ============================================================

.PHONY: setup up down logs pull-model ingest clean rebuild status

# --- Primeira configuração ---
setup:
	@if [ ! -f .env ]; then cp .env.example .env; echo ".env criado a partir do .env.example"; fi
	docker compose up -d --build
	@echo "Aguardando serviços iniciarem (30s)..."
	sleep 30
	$(MAKE) pull-model
	@echo ""
	@echo "====================================="
	@echo " AutoPredict AI está pronto!"
	@echo "====================================="
	@echo " API:      http://localhost:8000/docs"
	@echo " Chat:     http://localhost:7860"
	@echo " MinIO:    http://localhost:9001"
	@echo "====================================="

# --- Controle dos containers ---
up:
	docker compose up -d

down:
	docker compose down

rebuild:
	docker compose down
	docker compose up -d --build

status:
	docker compose ps

# --- Logs ---
logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

logs-frontend:
	docker compose logs -f frontend

logs-ollama:
	docker compose logs -f ollama

# --- LLM: Baixar modelo no Ollama ---
# Necessário APENAS na primeira vez (o modelo fica salvo no volume)
pull-model:
	@echo "Baixando modelo $(shell grep OLLAMA_MODEL .env | cut -d'=' -f2)..."
	docker exec autopredict-ollama ollama pull $(shell grep OLLAMA_MODEL .env | cut -d'=' -f2)

# --- Pipeline de dados ---
# Baixe os datasets do Kaggle, coloque em ./data/ e rode este comando
ingest:
	@echo "Rodando pipeline de ingestão..."
	docker exec autopredict-api python /app/src/data_pipeline/run_pipeline.py

# --- Utilitários ---
# Abre um shell dentro do container da API (útil para debug)
shell-api:
	docker exec -it autopredict-api bash

shell-postgres:
	docker exec -it autopredict-postgres psql -U autopredict -d autopredict

# --- Limpeza (CUIDADO: apaga todos os dados!) ---
clean:
	@echo "ATENÇÃO: Isso vai apagar TODOS os dados dos volumes!"
	@read -p "Tem certeza? (s/N): " confirm && [ "$$confirm" = "s" ]
	docker compose down -v
	docker system prune -f
