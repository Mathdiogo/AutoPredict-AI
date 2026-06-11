# Governança de Dados — AutoPredict AI

**Data de criação:** 2026-05-14 19:43:55  
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
