"""
Script para registrar documentação de governança de dados no MLflow.

Este script cria um experimento dedicado à governança e registra
os documentos das camadas Bronze, Silver e Gold como artefatos,
permitindo rastreabilidade e versionamento da documentação.
"""

import mlflow
import os
from datetime import datetime
from pathlib import Path


def register_governance_docs():
    """
    Registra documentos de governança das camadas Medallion no MLflow.
    
    Cria uma run no experimento "AutoPredict-Governance" e anexa
    os arquivos markdown como artefatos, além de registrar metadados
    relevantes como tags e métricas de documentação.
    """
    
    # Configuração do MLflow
    # Importar settings para usar a URL correta (mlflow:5000 dentro do container)
    from src.config import get_settings
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
    # Criar ou selecionar experimento de governança
    experiment_name = "AutoPredict-Governance"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                tags={
                    "purpose": "Data Governance Documentation",
                    "framework": "Medallion Architecture",
                    "project": "AutoPredict AI"
                }
            )
            print(f"✅ Experimento '{experiment_name}' criado (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Experimento '{experiment_name}' encontrado (ID: {experiment_id})")
    except Exception as e:
        print(f"❌ Erro ao configurar experimento: {e}")
        return
    
    # Caminhos dos documentos de governança
    docs_dir = Path(__file__).parent.parent.parent / "docs" / "governance"
    
    docs = {
        "bronze": docs_dir / "bronze_layer.md",
        "silver": docs_dir / "silver_layer.md",
        "gold": docs_dir / "gold_layer.md"
    }
    
    # Verificar se os documentos existem
    missing_docs = [layer for layer, path in docs.items() if not path.exists()]
    if missing_docs:
        print(f"❌ Documentos faltando: {', '.join(missing_docs)}")
        print(f"   Esperado em: {docs_dir}")
        return
    
    print(f"✅ Todos os 3 documentos de governança encontrados")
    
    # Iniciar run do MLflow
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"governance_docs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Tags descritivas
        mlflow.set_tag("mlflow.runName", f"Governance Docs - {datetime.now().strftime('%Y-%m-%d')}")
        mlflow.set_tag("governance.version", "1.0")
        mlflow.set_tag("governance.author", "AutoPredict Team")
        mlflow.set_tag("governance.framework", "Medallion (Bronze/Silver/Gold)")
        mlflow.set_tag("governance.last_updated", datetime.now().isoformat())
        
        # Parâmetros de configuração de governança
        mlflow.log_param("architecture", "Medallion")
        mlflow.log_param("layers", "Bronze, Silver, Gold")
        mlflow.log_param("storage_backend", "MinIO (S3-compatible)")
        mlflow.log_param("vector_db", "Milvus")
        mlflow.log_param("audit_db", "PostgreSQL")
        mlflow.log_param("total_layers", 3)
        
        # Métricas de documentação (contagem de linhas/palavras)
        for layer, doc_path in docs.items():
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.count('\n')
                words = len(content.split())
                
                mlflow.log_metric(f"{layer}_doc_lines", lines)
                mlflow.log_metric(f"{layer}_doc_words", words)
                
                print(f"   📄 {layer.capitalize()}: {lines} linhas, {words} palavras")
        
        # Registrar documentos como artefatos
        print("\n📦 Registrando documentos como artefatos no MLflow...")
        
        for layer, doc_path in docs.items():
            try:
                mlflow.log_artifact(str(doc_path), artifact_path="governance")
                print(f"   ✅ {layer}_layer.md registrado")
            except Exception as e:
                print(f"   ❌ Erro ao registrar {layer}_layer.md: {e}")
        
        # Criar um arquivo de índice
        index_content = f"""# Governança de Dados — AutoPredict AI

**Data de criação:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Versão:** 1.0  
**Framework:** Medallion Architecture

---

## 📚 Documentos Disponíveis

### [Bronze Layer](bronze_layer.md)
Camada de **ingestão** de dados brutos.
- Princípio: Imutabilidade
- Storage: MinIO bucket `bronze/`
- Formato: CSV original sem modificações

### [Silver Layer](silver_layer.md)
Camada de **limpeza e normalização**.
- Princípio: Qualidade de dados
- Storage: MinIO bucket `silver/`
- Transformações: Dropna, normalização de colunas, validação de tipos

### [Gold Layer](gold_layer.md)
Camada de **produção** otimizada para IA.
- Princípio: Pronto para consumo
- Storage: MinIO bucket `gold/` + Milvus (11.100 docs)
- Transformações: Chunking, embedding, indexação vetorial

---

## 🎯 Métricas de Qualidade

| Camada | KPI Principal | Meta |
|---|---|---|
| Bronze | Taxa de ingestão completa | 100% |
| Silver | Taxa de transformação | ≥ 95% |
| Gold | Taxa de indexação | 100% |

---

## 🔐 Políticas de Acesso

| Camada | Escrita | Leitura |
|---|---|---|
| Bronze | Pipeline Bronze apenas | Pipelines Silver/Gold |
| Silver | Pipeline Silver apenas | Pipeline Gold |
| Gold | Pipeline Gold apenas | RAG Retriever |

---

## 📞 Contato

**Product Owner:** AutoPredict Team  
**Scrum Master:** AutoPredict Team  
**Próxima revisão:** Sprint 10

---

## 🔗 Referências

- [Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture)
- [MinIO Documentation](https://min.io/docs)
- [Milvus Documentation](https://milvus.io/docs)
"""
        
        # Salvar índice temporário e registrar
        index_path = docs_dir / "README.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        mlflow.log_artifact(str(index_path), artifact_path="governance")
        print(f"   ✅ README.md (índice) registrado")
        
        # URL da run
        run = mlflow.active_run()
        run_id = run.info.run_id
        
        print(f"\n✅ Documentação de governança registrada com sucesso!")
        print(f"   🔗 MLflow Run ID: {run_id}")
        print(f"   🌐 Acesse: http://localhost:5001/#/experiments/{experiment_id}/runs/{run_id}")
        print(f"\n   📂 Artefatos disponíveis:")
        print(f"      - governance/bronze_layer.md")
        print(f"      - governance/silver_layer.md")
        print(f"      - governance/gold_layer.md")
        print(f"      - governance/README.md")
        
        return {
            "status": "success",
            "experiment_id": experiment_id,
            "run_id": run_id,
            "docs_registered": 4,
            "mlflow_url": f"http://localhost:5001/#/experiments/{experiment_id}/runs/{run_id}"
        }


if __name__ == "__main__":
    print("=" * 70)
    print("  Registrando Documentação de Governança no MLflow")
    print("=" * 70)
    print()
    
    try:
        result = register_governance_docs()
        
        if result and result["status"] == "success":
            print("\n" + "=" * 70)
            print("  ✅ PROCESSO CONCLUÍDO COM SUCESSO")
            print("=" * 70)
            print(f"\n  Próximos passos:")
            print(f"  1. Acesse o MLflow UI: http://localhost:5001")
            print(f"  2. Navegue até o experimento 'AutoPredict-Governance'")
            print(f"  3. Abra a run mais recente")
            print(f"  4. Baixe os artefatos na aba 'Artifacts'")
            print()
        else:
            print("\n❌ Falha no registro da documentação")
            
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
