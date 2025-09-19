---
title: 'A Standards-Compliant, Multi-Modal Platform for Offline Access to SRA Metadata'
title_short: 'Multi-Modal Offline Access to SRA Metadata'
tags:
  - SRA metadata
  - SQLite
  - Database
  - Biohackathon
authors:
  - name: Nishad Thalhath
    orcid: 0000-0001-9845-9714
    affiliation: 1
  - name: Kozo Nishida
    orcid: 0000-0001-8501-7319
    affiliation: [1,2]
affiliations:
  - name: RIKEN Center for Integrative Medical Sciences, Yokohama, Japan
    index: 1
    ror: 04mb6s476
  - name: Tokyo University of Agriculture and Technology, Koganei, JP
    ror: 00qg0kr10
    index: 2
date: 20 September 2025
cito-bibliography: paper.bib
event: BH25JP
biohackathon_name: "DBCLS BioHackathon 2025"
biohackathon_url:   "https://2025.biohackathon.org/"
biohackathon_location: "Mie, Japan, 2025"
group: YOUR-PROJECT-NAME-GOES-HERE
# URL to project git repo --- should contain the actual paper.md:
git_url: https://github.com/biohackathon-japan/bh25-srake
# This is the short authors description that is used at the
# bottom of the generated paper (typically the first two authors):
authors_short: "Thalhath & Nishida"
---

# Introduction

The SRAmetaDBB project, presented at BioHackathon Japan 2023, introduced an experimental JavaScript pipeline for creating SQLite databases from NCBI SRA (Sequence Read Archive) metadata dumps, with a vision for offline analysis and integration with Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems. While promising, the prototype faced significant challenges in performance, memory management, and production readiness when scaling to the full SRA dataset of over 45 million records. This paper presents SRAKE (SRA Knowledge Engine), a complete reimplementation in Go that not only addresses these limitations but extends the original vision with semantic search capabilities, quality control mechanisms, and multiple access interfaces. SRAKE achieves a 20-fold improvement in ingestion speed, maintains constant memory usage through zero-copy streaming, and provides standards-compliant interfaces following clig.dev guidelines. The platform introduces biomedical-specific semantic search using SapBERT embeddings via ONNX Runtime, implements comprehensive quality control thresholds for search results, and offers multiple access modalities including a CLI, REST API, MCP server for AI integration, and a simple web interface. Our development implementation demonstrates that SRAKE successfully transforms the experimental SRAmetaDBB concept into a production-ready platform, and seamless integration with modern AI workflows while maintaining the core vision of providing offline-capable, LLM-ready access to SRA metadata.

## Background and Previous Work

The Sequence Read Archive (SRA) is one of the largest public collections of high-throughput sequencing data, offering raw reads from diverse studies across many species and serving as a key resource for genomics and transcriptomics research. Interpreting these data relies heavily on rich metadata describing study design, experimental conditions, and processing methods. While SRA provides its own access tools, the scale and complexity of its metadata have driven the development of specialized resources such as SRAmetadb.sqlite, created under the SRAdb project [@zhuSRAdbQueryUse2013] to offer a single, queryable SQLite database for offline use. Widely adopted by tools like SRAdb and pysradb [@Choudhary2019], it allowed fast local queries and integration into bioinformatics workflows. However, maintaining this database has become increasingly challenging as sequencing output and metadata have grown rapidly. Processing the massive XML metadata into SQL now requires days of computation, no open and reproducible pipeline exists for regular updates, and the last public release (December 2023) is no longer actively maintained—underscoring the need for a more scalable and sustainable solution. Alternative solutions like ffq [@ffq] have emerged to address these challenges, but they too face limitations in performance, and usability.

The exponential growth of sequencing data in the NCBI SRA presents both opportunities and challenges for the research community. With over 45 million metadata records and counting, efficient access to this information has become crucial for dataset discovery, meta-analyses, and increasingly, for training and augmenting artificial intelligence systems in bioinformatics.

At BioHackathon Japan 2023, one of the projects introduced SRAmetaDBB (SRA metadata Database Builder), an experimental JavaScript pipeline designed to convert SRA metadata XML dumps into queryable SQLite databases [@thalhath_2024]. The project aimed to address several key use cases:

1. **Offline Analysis**: Enable researchers to work with SRA metadata without constant internet connectivity
2. **LLM Context Enhancement**: Provide structured metadata for Large Language Model applications
3. **RAG Pipeline Integration**: Support Retrieval-Augmented Generation workflows in bioinformatics
4. **Flexible Filtering**: Allow creation of species-specific or date-filtered databases

While SRAmetaDBB demonstrated the feasibility of these concepts, the JavaScript implementation encountered significant challenges when scaling to production workloads:

- **Performance Limitations**: Full dataset ingestion took several days
- **Memory Management**: Unbounded memory usage led to out-of-memory errors with large XML files
- **Lack of Resume Capability**: Interruptions required complete restart of the ingestion process
- **Limited Search Functionality**: Only basic SQL queries were supported with SQLite, lacking modern full-text and semantic search
- **Single Interface**: Command-line only access limited adoption potential


# Evolution to SRAKE

Recognizing the maintenance bottlenecks around SRAmetaDBB and the growing need to integrate AI-native retrieval into bioinformatics workflows, we built SRAKE (SRA Knowledge Engine) as a ground-up reimplementation that turns an experimental pipeline into a production-ready platform. First, we migrated the codebase from JavaScript to Go to leverage higher throughput, built-in concurrency (goroutines/channels), and tighter memory control for sustained, large-scale ingestion. Second, we replaced batch ETL with zero-copy streaming, removing intermediate disk artifacts, bounding memory usage, and enabling continuous updates rather than long, failure-prone rebuilds. Third, we expanded retrieval beyond relational queries by combining SQL, full-text search (Bleve), and semantic search via vector embeddings, supporting hybrid workflows where exact filters, keyword queries, and embedding-based similarity can be composed in a single system. Fourth, we aligned the tool with common conventions for maintainability—adopting clig.dev guidelines for a predictable, machine-friendly CLI UX. As AI agents increasingly become the primary “users” of command-line tools, the CLI’s design and interfaces are crucial: commands must not block, outputs should stream, exit codes must be deterministic, and behavior should be scriptable and composable—precisely the discipline clig.dev enforces. Finally, we extended access modes from a CLI-only tool to a multi-modal interface that includes a REST API (for automation and integration), a Model Context Protocol (MCP) server (for LLM-centric tooling), and a web UI (for interactive exploration). Together, these changes deliver faster updates, lower operational overhead, and AI-ready discovery while preserving the reproducibility and offline-friendly properties that made the original approach attractive.

## Contributions

This work makes the following contributions:
	1.	Production-Ready Platform — A robust, scalable system that streams, parses, and indexes the full SRA metadata corpus with reproducible builds and incremental updates.
	2.	Standards-Compliant Design — CLI and filesystem conventions aligned with clig.dev and the XDG Base Directory spec, emphasizing non-blocking, streamable outputs and deterministic exit codes for AI-driven automation and shell pipelines.
	3.	Semantic Search Integration — A biomedical-specific embedding layer for SRA metadata using SapBERT, composed with SQL and Bleve full-text to enable hybrid lexical-semantic retrieval.
	4.	Quality Control Framework — A threshold-based filtering and evaluation mechanism that enforces minimum relevance/precision criteria and exposes tunable guards per workflow.
	5.	Multi-Modal Platform — Unified access via CLI, REST API, MCP server (for LLM tooling), and a web interface, supporting both human researchers and autonomous agents.
	6.	Performance — Significant improvements over the prototype in build time, memory footprint, and query latency through Go concurrency and zero-copy streaming; detailed measurements are provided in the evaluation section.


## Multi-Modal Search Architecture

SRAKE implements a three-tier retrieval stack that composes structured filters, lexical relevance, and semantic similarity into a single, tunable ranking pipeline. At ingest, metadata are normalized into a relational core for SQL filtering; in parallel, text-bearing fields are indexed for full-text search (Bleve) to provide BM25 scoring, and precomputed biomedical embeddings (e.g., SapBERT) enable vector search over the same records. At query time, structured predicates (e.g., organism, assay, date ranges) are applied first in the SQL layer to prune the candidate set efficiently. The surviving candidates are scored in the full-text layer using BM25 for keyword intent, while the vector layer computes cosine similarity between the query embedding and candidate embeddings to capture synonymy and concept-level matches. Scores are min–max normalized per layer, deduplicated at the accession level, and combined via a hybrid ranker (tie-broken by recency and field priority when needed). The hybrid mechanism is configurable—users can bias toward precise keyword matches or concept similarity, and quality gates from the evaluation framework can drop results below minimum thresholds before presentation. The final score is computed as:

```go
finalScore = α * BM25Score + (1-α) * CosineSimilarity
// α is configurable via --hybrid-weight flag (default 0.7)
```

This architecture yields fast, exact filtering; robust lexical retrieval for well-specified queries; and resilient semantic matching for noisy or underspecified prompts—supporting both human workflows and AI agents that benefit from controllable, streaming, and non-blocking search behavior.


## Standards Compliance and Best Practices

SRAKE is engineered around established conventions so that it “just works” for humans and for AI agents that orchestrate command-line workflows. The CLI follows **clig.dev** guidance to ensure predictable, scriptable behavior with human-readable defaults, streaming output, and deterministic exit codes that do not block downstream consumers. In practice, a query like `srake search "breast cancer"` renders a compact table for interactive use, while `--format {table|json|csv|tsv}` switches to machine-friendly serialization without altering semantics. Flag conventions are consistent and discoverable (`-h/--help`, `--version`, `-q/--quiet`, `-v/--verbose`, `--no-color`), and retrieval quality is tunable through `--similarity-threshold` (cosine similarity, 0–1, default 0.5), `--min-score` (BM25 floor), `--top-percentile` (retain only the top N% of results), and `--show-confidence` (emit confidence indicators). Errors are actionable—if a local database is missing at `./data/SRAmetadb.sqlite`, SRAKE explains the condition and suggests `srake ingest --auto`—and long-running operations stream progress (percent complete, record counts, ETA) so agents and shells can react in real time. Beyond the CLI, SRAKE exposes an MCP server and adheres to the **Model Context Protocol** to integrate cleanly with LLM-based tooling, following the practices described in the MCP introduction ([modelcontextprotocol.io/docs/getting-started/intro](https://modelcontextprotocol.io/docs/getting-started/intro)). File placement respects the **XDG Base Directory** specification to separate durable data, cacheable artifacts, and configuration: `$XDG_DATA_HOME/srake/` (e.g., models under `models/` and SQLite files in `databases/`), `$XDG_CACHE_HOME/srake/` (e.g., Bleve indexes in `index/`, vector caches in `embeddings/`, resumable state in `checkpoints/`), and `$XDG_CONFIG_HOME/srake/` (user settings in `config.yaml`). Configuration precedence is deterministic and transparent—command-line flags override environment variables (`SRAKE_*`), which override user config (`~/.config/srake/config.yaml`), which override system config (`/etc/srake/config.yaml`), with built-in defaults applied last—ensuring reproducibility across interactive sessions, scripted pipelines, and autonomous agent runs.


```bash
# Default output optimized for human readability
$ srake search "breast cancer"
ACCESSION  TYPE   TITLE                          ORGANISM        PLATFORM
SRP123456  study  Breast cancer RNA-Seq analysis Homo sapiens   ILLUMINA
[... tabular output ...]

# Machine-readable when needed
$ srake search "breast cancer" --format json
{"results": [{"accession": "SRP123456", ...}]}
```


### Semantic Search with Biomedical Embeddings

SRAKE introduces biomedical-specific semantic retrieval by integrating **SapBERT** (Self-Alignment Pre-training for Biomedical Entity Representations)—a BERT-family model trained to cluster synonyms and canonical entity names across biomedical vocabularies [@sapbert]. This choice improves recall for domain phrases (e.g., *HER2-positive breast carcinoma* ↔ *ERBB2+ breast cancer*) and reduces vocabulary mismatch common in free-text metadata. At ingest, SRAKE generates stable, versioned embeddings for key text fields (titles, abstracts/labels, organism- and assay-specific descriptors), caches them on disk, and records the model/version hash to guarantee reproducibility. At query time, the user’s text prompt is embedded with the same tokenizer and model; similarity is computed with cosine distance and fused with BM25 in the hybrid ranker. To balance accuracy and throughput, the pipeline supports batched inference, mean pooling over token embeddings (default), optional CLS pooling, L2 normalization, and provider selection (CPU/GPU) via ONNX Runtime. Embeddings are stored in a field-aware layout to allow weighting per field during ranking, and quantization (e.g., dynamic or int8) can be enabled to shrink memory and accelerate inference with negligible loss for retrieval.


```go
// Simplified, batched embedding pipeline (pseudocode)
type Embedder struct {
    tokenizer Tokenizer
    session   *onnxruntime.Session // configured with CPU/CUDA providers
    maxLen    int                  // e.g., 512
}

func (e *Embedder) GenerateEmbeddings(texts []string) ([][]float32, error) {
    // 1) Tokenization with padding/truncation to maxLen
    //    Keep attention_mask to exclude padding from pooling.
    batchTokens := e.tokenizer.EncodeBatch(texts, e.maxLen) // returns input_ids, attention_mask

    // 2) ONNX inference (zero-copy where possible)
    inputs := []onnxruntime.Input{
        {Name: "input_ids",      Value: batchTokens.InputIDs},      // [B, L]
        {Name: "attention_mask", Value: batchTokens.AttentionMask}, // [B, L]
    }
    outputs, err := e.session.Run(inputs) // outputs[0]: [B, L, H] hidden states
    if err != nil { return nil, err }

    // 3) Pooling (mean over non-masked tokens) and L2 normalization
    //    Alternative: CLS pooling via outputs[0][:,0,:].
    pooled := make([][]float32, len(texts))
    for i := range texts {
        pooled[i] = meanPool(outputs[0][i], batchTokens.AttentionMask[i]) // [H]
        pooled[i] = l2Normalize(pooled[i])                                 // unit vector
    }
    return pooled, nil
}

// Notes:
// - Hidden size H is model-dependent (e.g., 768). Persist alongside model ID for reproducibility.
// - Quantization (e.g., int8) and batching improve latency/throughput with minor accuracy trade-offs.
// - Use a stable pre/post-processing config (tokenizer.json, lowercasing rules) to ensure determinism.

```

### Multi-Modal Access Interfaces

SRAKE exposes a unified set of access modalities so that the same retrieval and quality-control semantics are available to both human researchers and AI agents. The command-line interface (CLI) is the reference surface: it emphasizes human-readable defaults, streaming, and deterministic exit codes for robust shell composition. The REST API mirrors the CLI’s parameters and behaviors with isomorphic endpoints, enabling programmatic integration from any environment and serving as the foundation for the web application. For agentic workflows, SRAKE provides a Model Context Protocol (MCP) server that adheres to the practices in the MCP introduction, allowing LLM-based tools to invoke searches and ingest results with minimal glue code. The web interface (under active development) layers on the same API to support exploratory search, visualization, and export. Across all modalities, requests map to a single, shared pipeline—SQL filtering, BM25 full-text, and SapBERT-based vector search—so that flags, query parameters, and UI controls remain consistent, results can stream without blocking, and configurations (thresholds, formats, field weights) yield identical outcomes regardless of the interface.

Command-Line Interface (Primary):

```bash
# Advanced search with quality control
srake search "RNA-Seq human cancer" \
  --organism "homo sapiens" \
  --platform ILLUMINA \
  --date-from 2024-01-01 \
  --similarity-threshold 0.7 \
  --top-percentile 10 \
  --show-confidence
```

REST API Server:

```http
GET /api/v1/search?q=cancer&similarity_threshold=0.7
{
  "query": "cancer",
  "total_hits": 1543,
  "hits": [...],
  "mode": "hybrid",
  "time_ms": 87
}
```

The Model Context Protocol server enables direct integration with AI assistants:

```python
# Example integration with LangChain
from langchain.tools import SRAKETool

tool = SRAKETool(endpoint="http://localhost:8080/mcp")
context = tool.search(
    query="breast cancer RNA-Seq",
    filters={"organism": "homo sapiens"},
    limit=5
)
```

## Implementation

Table: Technology Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| Core Language | Go 1.21+ | Performance, concurrency, memory efficiency |
| Database | SQLite 3.40+ | Portability, FTS5 support, proven reliability |
| Full-Text Search | Bleve v2 | Pure Go, flexible analyzers, good performance |
| ML Runtime | ONNX Runtime Go | Cross-platform, no Python dependency |
| Web Framework | SvelteKit 2.0 | Modern, performant, good DX |
| UI Components | shadcn/ui | Accessible, customizable, well-designed |


# Discussion

SRAKE meets the aims originally set for SRAmetadb while materially extending them in scope and reliability. Most visibly, a roughly 20× speed improvement turns builds that once took days into workflows measured in hours, and constant-bounded memory usage allows the system to run on resource-constrained machines without thrashing. Search has evolved from purely SQL predicates to a multi-modal pipeline that blends relational filtering, BM25 full-text, and SapBERT-based semantic similarity, delivering both precision and recall for heterogeneous queries. The platform is also AI-ready: it exposes consistent, non-blocking interfaces that fit LLM and RAG workflows, and it ships with production-quality error handling, logging, and monitoring so that failures are diagnosable and recoveries are predictable.

Realizing these gains required several deliberate trade-offs. Rewriting from JavaScript to Go delivered higher throughput, built-in concurrency, and small static binaries that simplify deployment; the cost was a smaller bioinformatics ecosystem and a modest learning curve for contributors, a trade the performance and operational benefits justified. On storage and retrieval, combining SQLite with Bleve indexes and vector embeddings provided best-of-breed capabilities for structured, lexical, and semantic search while keeping SQLite compatibility for downstream tools; the downside is additional complexity and the need to maintain multiple indexes, which we accepted to unlock flexibility and speed. Investing in standards compliance (e.g., clig.dev for CLI ergonomics, XDG for filesystem layout, MCP for agent integration) imposed early development overhead but paid off in predictable UX, easier automation, and long-term maintainability—qualities that matter in production and at scale.

Several lessons emerged. First, streaming architecture is essential for very large corpora: it prevents intermediate blow-ups, stabilizes memory, and shortens time-to-first-result. Second, standards are not merely cosmetic; they accelerate adoption by humans and agents and reduce integration friction across environments. Third, quality control must be explicit in AI-augmented search: thresholded similarity, score floors, and confidence surfacing give users the assurance they need to act on results. Fourth, multi-modal access is not optional—different teams prefer CLI, API, MCP, or web interfaces, and a single semantics layer behind them ensures consistent outcomes. Finally, moving from prototype to production is less about incremental tuning and more about rethinking fundamentals (concurrency, failure modes, observability); the gains in reliability and performance validate that approach.

# Limitations and Future Work

Despite its progress toward a production-ready platform, SRAKE presently has several constraints. The web interface remains under active development, which limits discoverability for non-CLI users and constrains collaborative, exploratory analysis. Visualization is likewise minimal; beyond tabular outputs and simple summaries, there are no built-in charts or interactive plots to help users reason about large result sets or quality signals at a glance. Updates are handled in scheduled batches rather than via a real-time mechanism, so newly published or corrected records may not appear until the next ingest cycle. Finally, the system runs in a single-node configuration: while this simplifies deployment and debugging, it caps throughput under heavy concurrent load and creates a single point of failure.

To address these gaps, we plan to evolve the architecture and interfaces along several axes. First, we will introduce a distributed, horizontally scalable design with stateless API workers and sharded indexes to increase throughput and resilience under bursty traffic. Second, we aim to expose a GraphQL API alongside the existing REST surface to support flexible, client-driven queries and efficient field selection across SQL, full-text, and vector layers. Third, we will add an advanced visualization dashboard—integrated with the same search pipeline—to provide interactive filtering, facet exploration, embedding-space inspection, and export, enabling both quick diagnostics and richer analyses. Fourth, we intend to implement federation with complementary biological data sources so that SRAKE can enrich results with harmonized identifiers and context (e.g., organisms, assays, ontology terms) while preserving provenance. Lastly, we will experiment with AutoML-guided query optimization (e.g., learning-to-rank and adaptive thresholds) to tune hybrid scoring and field weights per query class, improving relevance without sacrificing the transparency and reproducibility expected in scientific workflows.

# Conclusion

SRAKE represents a successful evolution from experimental concept to production-ready platform, transforming the vision laid out by SRAmetaDBB into a robust tool for the research community. By addressing the fundamental limitations of the JavaScript prototype through architectural redesign, language migration, and standards compliance, SRAKE achieves the original goals while adding modern capabilities like semantic search and quality control.

The platform's success demonstrates several key principles:
1. Performance and efficiency are prerequisites for handling modern-scale biological data
2. Standards compliance ensures longevity and community adoption
3. Multiple access modalities serve diverse user needs and enable AI integration
4. Quality control mechanisms are essential for trustworthy results

With its open-source availability, comprehensive documentation, and active development, SRAKE is positioned to serve as critical infrastructure for SRA metadata access in both traditional bioinformatics workflows and emerging AI applications. The platform fulfills the promise of making SRA metadata truly accessible for offline analysis, LLM enhancement, and RAG pipelines while providing the performance and reliability required for production use.

## References
