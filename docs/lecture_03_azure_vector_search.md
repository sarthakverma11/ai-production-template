# Lecture 3: Azure Vector Search Retrieval

## 1. Purpose of this lecture

Lecture 3 continues from the Lecture 2 document-processing pipeline.

Lecture 2 created local processed chunks and lineage metadata. Lecture 3 moves those processed artifacts into an Azure-backed retrieval workflow.

The goal is to make policy chunks searchable using embeddings and vector search.

## 2. What we built

Lecture 3 adds:

- Azure Blob Storage artifact loading
- Azure OpenAI embedding generation
- Azure AI Search vector index schema
- offline indexing pipeline
- semantic top-k retrieval service
- Streamlit evidence display
- unit tests with mocked Azure clients

## 3. What this is not

Lecture 3 is not a full RAG application.

It does not:

- call a chat completion model
- generate a final answer
- use LangChain or LlamaIndex
- use Azure AI Search indexers or skillsets
- use Azure-side automatic chunking
- deploy the application

The app retrieves evidence only.

## 4. End-to-end flow

```text
Azure Blob Storage chunks_v1.json
-> Azure OpenAI embeddings
-> Azure AI Search vector index
-> user query embedding
-> top-k retrieved chunks
-> Streamlit displays evidence and metadata
```

## 5. Azure resources

| Resource | Name |
| -------- | ---- |
| Storage account | `stedyodallmops01` |
| Raw container | `raw-documents` |
| Processed container | `processed-artifacts` |
| Metadata container | `metadata` |
| Embedding deployment | `text-embedding-3-small` |
| Search service | `srch-edyoda-llmops01` |
| Search index | `policy-knowledge-index-v1` |

The Azure resources already exist. The Python code creates the search index programmatically.

## 6. Environment variables

Create a local `.env` file from `.env.example`.

```env
PROJECT_ENV=local

AZURE_STORAGE_CONNECTION_STRING=

AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_API_KEY=

AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_API_KEY=
AZURE_SEARCH_INDEX_NAME=policy-knowledge-index-v1
```

Secrets must stay in `.env`. They should not be placed in `configs/config.yaml`, code, docs, logs, or Git.

## 7. Configuration

Lecture 3 extends `configs/config.yaml` with non-secret settings.

| Section | Purpose |
| ------- | ------- |
| `azure_storage` | Container names and blob names |
| `embedding` | Embedding provider, deployment env var, dimensions, and batch size |
| `search` | Index name env var, vector field, top-k, profile, and algorithm names |
| `application` | Retrieval mode and Lecture 3 user-facing notice |

The config validates that:

- embedding dimensions are positive
- embedding batch size is positive
- search top-k is positive

## 8. New code paths

| Path | Purpose |
| ---- | ------- |
| `src/storage/blob_service.py` | Reads JSON artifacts and lists/uploads blobs |
| `src/embeddings/embedding_service.py` | Calls Azure OpenAI embeddings |
| `src/search/index_schema.py` | Defines Azure AI Search vector index schema |
| `src/search/index_manager.py` | Creates or reuses the search index |
| `src/search/indexing_pipeline.py` | Embeds chunks and uploads records |
| `src/search/retrieval_service.py` | Embeds user query and retrieves top-k chunks |
| `src/chat_service.py` | Routes app questions through keyword or semantic mode |
| `app/streamlit_app.py` | Displays retrieved evidence and metadata |

## 9. Run Lecture 2 first

Lecture 3 depends on Lecture 2 artifacts.

```bash
python -m src.processing.build_knowledge_artifacts
```

Expected local files:

```text
data/processed/chunks_v1.json
data/metadata/lineage_v1.json
```

Upload those files to Blob Storage:

```text
processed-artifacts/chunks_v1.json
metadata/lineage_v1.json
```

## 10. Run the indexing pipeline

```bash
python -m src.search.indexing_pipeline
```

The pipeline:

1. Loads config and `.env`.
2. Validates required environment variables.
3. Checks that required Blob containers exist.
4. Reads `chunks_v1.json` and `lineage_v1.json` from Blob Storage.
5. Validates chunk metadata.
6. Generates one test embedding.
7. Confirms embedding dimensions match the config.
8. Creates or reuses the Azure AI Search index.
9. Generates embeddings for every chunk.
10. Uploads records into the search index.
11. Saves `outputs/indexing_summary_v1.json`.

## 11. Run retrieval from CLI

```bash
python -m src.search.retrieval_service "How much time off do employees receive?"
```

Expected local file:

```text
outputs/retrieval_results.json
```

The CLI prints ranked chunk IDs, scores, and source files.

## 12. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app displays:

- semantic retrieval notice
- retrieved chunk rank
- chunk text
- source file
- document version
- chunk ID
- chunk index
- search score
- index version
- retrieval mode

It does not display:

- Azure keys
- connection strings
- raw embedding vectors
- generated final answers

## 13. Demo steps

1. Show `data/processed/chunks_v1.json`.
2. Explain that Lecture 2 created chunks locally.
3. Show `.env.example`.
4. Explain that secrets live only in local `.env`.
5. Show `configs/config.yaml`.
6. Point out Blob, embedding, and search settings.
7. Run `python -m src.search.indexing_pipeline`.
8. Open `outputs/indexing_summary_v1.json`.
9. Run a retrieval query from CLI.
10. Open `outputs/retrieval_results.json`.
11. Run Streamlit.
12. Ask a policy question.
13. Show the top-k retrieved chunks.
14. Remind learners that no final answer is generated yet.

## 14. Git versus secrets

Git tracks:

- code
- tests
- docs
- `.env.example`
- non-secret YAML config

Git does not track:

- `.env`
- logs
- generated outputs
- local credentials

## 15. Troubleshooting

| Problem | Check |
| ------- | ----- |
| Missing environment variable | Fill the local `.env` file |
| Blob not found | Confirm exact container and blob names |
| Invalid JSON blob | Re-upload `chunks_v1.json` or `lineage_v1.json` |
| Embedding deployment not found | Check `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` |
| Dimension mismatch | Confirm the embedding deployment returns 1536 dimensions |
| Search index creation fails | Check `AZURE_SEARCH_ENDPOINT` and admin key |
| Retrieval returns no chunks | Run the indexing pipeline first |
| Streamlit shows retrieval error | Check local `.env` and whether the index exists |

## 16. Quick recap

Lecture 3 turns Lecture 2 chunks into searchable knowledge.

It uses Azure Blob Storage for artifacts, Azure OpenAI for embeddings, and Azure AI Search for vector retrieval.

The application still stops at evidence retrieval. Final answer generation comes later.

## 17. What comes next

Future lectures can add:

- grounded LLM answer generation
- prompt templates
- citation formatting
- retrieval evaluation
- orchestration
- deployment
- monitoring
