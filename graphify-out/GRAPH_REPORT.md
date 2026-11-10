# Graph Report - LegalRAG  (2026-11-11)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1147 nodes · 2531 edges · 77 communities (53 shown, 24 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 171 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a6be9f6e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- json
- LegalStructureParser
- patches.py
- classifier.py
- config.py
- parse_legal_structure
- re
- LegalAwareChunker
- _parser_chunk
- BM25Index
- extractor.py
- retrieval/__init__.py
- registry.py
- ocr.py
- cleaner.py
- chunk.py
- retrieve.py
- HybridRetriever
- test_ocr.py
- .chunk
- DocumentMeta
- ClauseChunker
- audit_pdf
- legal_aware.py
- LineClassifier
- rrf_fusion
- Any
- build.py
- get_engine
- build_colab_embeddings.py
- documents
- FixedSizeChunker
- typer
- BaseRetriever
- test_build.py
- ArticleChunker
- base.yaml
- retrieval
- default.yaml
- retrieval
- generate.py
- llm
- audit.py
- variants
- hybrid
- baselines
- baselines
- corpus
- dense
- evaluation
- hybrid
- legal_aware.yaml
- corpus
- dense
- evaluation
- hybrid
- evaluation.yaml
- OperationDetector
- audit
- render_page
- legal_aware
- experiment
- legal_aware
- experiment
- load_chunks
- _safe_id_part
- TesseractEngine
- map_gold
- corpus/__init__.py
- main
- pipeline
- legal-rag

## God Nodes (most connected - your core abstractions)
1. `DocumentMeta` - 47 edges
2. `LegalStructureParser` - 35 edges
3. `LegalAwareChunker` - 32 edges
4. `parse_legal_structure()` - 24 edges
5. `extract_pdf_lines()` - 23 edges
6. `FixedSizeChunker` - 22 edges
7. `resolve_document()` - 22 edges
8. `build_document()` - 22 edges
9. `BM25Index` - 21 edges
10. `ClauseChunker` - 20 edges

## Surprising Connections (you probably didn't know these)
- `test_run_corpus_build_refuses_existing_run()` --uses--> `DocumentRegistry`  [INFERRED]
  tests/unit/test_build.py → src/legal_rag/corpus/registry.py
- `main()` --uses--> `DenseIndex`  [INFERRED]
  scripts/build_colab_embeddings.py → src/legal_rag/retrieval/dense.py
- `TestDenseIndex` --uses--> `DenseIndex`  [INFERRED]
  tests/unit/test_retrieval.py → src/legal_rag/retrieval/dense.py
- `_document()` --uses--> `DocumentMeta`  [INFERRED]
  tests/unit/test_build.py → src/legal_rag/corpus/registry.py
- `test_build_document_missing_pdf()` --uses--> `DocumentMeta`  [INFERRED]
  tests/unit/test_build.py → src/legal_rag/corpus/registry.py

## Import Cycles
- None detected.

## Communities (77 total, 24 thin omitted)

### Community 0 - "json"
Cohesion: 0.06
Nodes (51): check_coverage(), chunk_clause_key(), chunk_key(), coverage_report(), CoverageResult, load_chunks(), _ref_in_index(), _aggregate() (+43 more)

### Community 1 - "LegalStructureParser"
Cohesion: 0.10
Nodes (7): is_article_intro_start(), is_noise_line(), LineInfo, join_text(), LegalStructureParser, make_chunk(), ParserState

### Community 2 - "patches.py"
Cohesion: 0.07
Nodes (17): lines(), pdf(), _apply_one(), apply_page_patches(), _check_ocr_evidence(), lcs_coverage(), _lcs_len(), load_patches() (+9 more)

### Community 3 - "classifier.py"
Cohesion: 0.09
Nodes (15): extract_article_number(), has_signature_title_anywhere(), is_appendix_heading(), is_appendix_start(), is_legal_reference_continuation(), is_likely_legal_reference_in_content(), is_likely_money_amount(), is_signature_end() (+7 more)

### Community 4 - "config.py"
Cohesion: 0.11
Nodes (19): BaselineChunkingConfig, BM25Config, ChunkingConfig, CorpusConfig, _deep_merge(), DenseConfig, EvaluationConfig, ExperimentConfig (+11 more)

### Community 5 - "parse_legal_structure"
Cohesion: 0.10
Nodes (8): clean_line(), strip_heading_noise(), is_ref_like_title(), parse_legal_structure(), _parse(), TestEdgeCases, TestParser, TestSignatureAndAppendix

### Community 6 - "re"
Cohesion: 0.14
Nodes (11): _chunk_articles(), ends_mid_sentence(), find_legal_references(), is_short_title(), is_suspicious_ending(), is_truncated_title(), PostParseValidator, validate_chunks() (+3 more)

### Community 9 - "BM25Index"
Cohesion: 0.14
Nodes (6): bm25(), BM25Retriever, BM25Index, build_bm25_index(), tokenize_vietnamese(), TestBM25

### Community 10 - "extractor.py"
Cohesion: 0.14
Nodes (10): extract_pdf_lines(), main(), _page_lines(), parse_corpus(), save_corpus(), is_page_number(), ParseResult, _git_commit() (+2 more)

### Community 12 - "registry.py"
Cohesion: 0.13
Nodes (9): _cached_registry(), DocumentRegistry, load_registry(), RegistryError, test_invalid_date_rejected(), test_invalid_status_rejected(), test_registry_loads_all_documents(), test_resolve_unknown_id_is_neutral() (+1 more)

### Community 13 - "ocr.py"
Cohesion: 0.17
Nodes (10): _cache_matches(), _cache_params(), _cache_path(), _config_hash(), load_cached_pages(), ocr_document(), _one(), ocr_page() (+2 more)

### Community 14 - "cleaner.py"
Cohesion: 0.13
Nodes (9): clean_chunk(), clean_content(), clean_metadata_field(), is_page_number(), is_superscript_footnote(), is_suspicious_ending(), remove_known_footnote_number(), remove_superscript_footnote() (+1 more)

### Community 15 - "chunk.py"
Cohesion: 0.19
Nodes (10): save_csv(), save_jsonl(), all_baselines(), article(), clause(), clean(), fixed_size(), legal_aware() (+2 more)

### Community 16 - "retrieve.py"
Cohesion: 0.14
Nodes (5): dense(), _display_results(), DenseRetriever, build_dense_index(), DenseIndex

### Community 18 - "HybridRetriever"
Cohesion: 0.15
Nodes (5): hybrid(), rerank(), HybridEvaluatorRetriever, HybridRetriever, run_hybrid_retrieval()

### Community 19 - "test_ocr.py"
Cohesion: 0.16
Nodes (9): FakeEngine, FakePsmEngine, pdf_path(), test_legacy_cache_without_psm_is_accepted(), test_ocr_document_selected_pages(), test_ocr_page_cache_key_includes_dpi(), test_ocr_page_cache_key_includes_pdf_fingerprint(), test_ocr_page_cache_key_includes_psm() (+1 more)

### Community 21 - "DocumentMeta"
Cohesion: 0.16
Nodes (5): make_chunk(), DocumentMeta, resolve_document(), test_metadata_fields_are_copied(), test_resolve_accepts_document_meta()

### Community 22 - "ClauseChunker"
Cohesion: 0.18
Nodes (4): ClauseChunker, finish_clause(), TestBaselineChunkerCorpusMetadata, TestClauseChunker

### Community 23 - "audit_pdf"
Cohesion: 0.19
Nodes (11): audit_pdf(), count_binary_corruption(), PageAudit, _blank_pdf(), _pdf_with_page_image_and_header(), test_audit_flags_corrupt_pdf(), test_audit_keeps_full_text_layer_page(), test_audit_missing_pdf() (+3 more)

### Community 24 - "legal_aware.py"
Cohesion: 0.17
Nodes (3): BaseChunker, IdentityResolver, join_text()

### Community 26 - "rrf_fusion"
Cohesion: 0.12
Nodes (4): rrf_fusion(), weighted_fusion(), TestDenseIndex, TestHybridFusion

### Community 27 - "Any"
Cohesion: 0.25
Nodes (4): _aggregate_source_metadata(), ParagraphChunker, _source_reference(), TestParagraphChunker

### Community 28 - "build.py"
Cohesion: 0.23
Nodes (7): build_document(), DocumentBuildResult, _git_commit(), _make_chunker(), _resolve_ocr_pages(), run_corpus_build(), _sha256()

### Community 29 - "get_engine"
Cohesion: 0.17
Nodes (6): EasyOcrEngine, get_engine(), OcrUnavailable, PaddleEngine, test_get_engine_missing_dependency(), test_get_engine_unknown()

### Community 30 - "build_colab_embeddings.py"
Cohesion: 0.21
Nodes (3): load_chunks(), main(), sha256_file()

### Community 31 - "documents"
Cohesion: 0.13
Nodes (14): documents, doc_type, document_id, document_name, document_number, effective_from, effective_to, expected_articles (+6 more)

### Community 33 - "typer"
Cohesion: 0.25
Nodes (5): coverage(), end_to_end(), generation(), _parse_k_values(), retrieval()

### Community 35 - "test_build.py"
Cohesion: 0.25
Nodes (11): _audit(), _document(), _lines(), _pdf_with_text(), test_build_document_missing_pdf(), test_build_document_ocr_required(), test_build_document_ok(), test_build_document_source_corrupt_skips_ocr() (+3 more)

### Community 37 - "base.yaml"
Cohesion: 0.20
Nodes (9): generation, citation_format, llm, prompt, max_tokens, model, temperature, system (+1 more)

### Community 38 - "retrieval"
Cohesion: 0.20
Nodes (10): b, enabled, k1, tokenizer, enabled, model, top_k, retrieval (+2 more)

### Community 39 - "default.yaml"
Cohesion: 0.20
Nodes (9): generation, citation_format, llm, prompt, max_tokens, model, temperature, system (+1 more)

### Community 40 - "retrieval"
Cohesion: 0.20
Nodes (10): b, enabled, k1, tokenizer, enabled, model, top_k, retrieval (+2 more)

### Community 42 - "llm"
Cohesion: 0.22
Nodes (8): citation_format, llm, max_tokens, model, temperature, prompt, system, template

### Community 43 - "audit.py"
Cohesion: 0.36
Nodes (4): audit_registry(), main(), PdfAudit, write_audit_report()

### Community 44 - "variants"
Cohesion: 0.25
Nodes (7): document_id, enabled, variants, chunk_size, name, output_dir, overlap

### Community 45 - "hybrid"
Cohesion: 0.32
Nodes (3): bm25(), dense(), hybrid()

### Community 46 - "baselines"
Cohesion: 0.29
Nodes (7): article, clause, fixed_size, paragraph, baselines, chunk_size, overlap

### Community 47 - "baselines"
Cohesion: 0.29
Nodes (7): article, clause, fixed_size, paragraph, baselines, chunk_size, overlap

### Community 48 - "corpus"
Cohesion: 0.33
Nodes (6): corpus, document_id, document_name, document_number, expected_articles, pdf_path

### Community 49 - "dense"
Cohesion: 0.33
Nodes (6): batch_size, device, enabled, max_length, model, dense

### Community 50 - "evaluation"
Cohesion: 0.33
Nodes (6): evaluation, dev_set_path, generation_metrics, metrics, retrieval_k, test_set_path

### Community 51 - "hybrid"
Cohesion: 0.33
Nodes (6): enabled, fusion_method, rrf_k, weight_bm25, weight_dense, hybrid

### Community 52 - "legal_aware.yaml"
Cohesion: 0.33
Nodes (5): chunk_types, document_id, enabled, output_dir, preserve_hierarchy

### Community 53 - "corpus"
Cohesion: 0.33
Nodes (6): corpus, document_id, document_name, document_number, expected_articles, pdf_path

### Community 54 - "dense"
Cohesion: 0.33
Nodes (6): batch_size, device, enabled, max_length, model, dense

### Community 55 - "evaluation"
Cohesion: 0.33
Nodes (6): evaluation, dev_set_path, generation_metrics, metrics, retrieval_k, test_set_path

### Community 56 - "hybrid"
Cohesion: 0.33
Nodes (6): enabled, fusion_method, rrf_k, weight_bm25, weight_dense, hybrid

### Community 57 - "evaluation.yaml"
Cohesion: 0.33
Nodes (5): dev_set_path, generation_metrics, metrics, retrieval_k, test_set_path

### Community 61 - "render_page"
Cohesion: 0.33
Nodes (3): OcrError, render_page(), test_render_page()

### Community 62 - "legal_aware"
Cohesion: 0.40
Nodes (5): chunking, legal_aware, chunk_types, enabled, preserve_hierarchy

### Community 63 - "experiment"
Cohesion: 0.40
Nodes (5): experiment, log_level, output_dir, run_id, seed

### Community 64 - "legal_aware"
Cohesion: 0.40
Nodes (5): chunking, legal_aware, chunk_types, enabled, preserve_hierarchy

### Community 65 - "experiment"
Cohesion: 0.40
Nodes (5): experiment, log_level, output_dir, run_id, seed

## Knowledge Gaps
- **1 isolated node(s):** `legal-rag`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 437 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **24 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DocumentMeta` connect `DocumentMeta` to `FixedSizeChunker`, `LegalStructureParser`, `classifier.py`, `ArticleChunker`, `parse_legal_structure`, `test_build.py`, `LegalAwareChunker`, `_parser_chunk`, `extractor.py`, `audit.py`, `registry.py`, `.chunk`, `ClauseChunker`, `audit_pdf`, `legal_aware.py`, `Any`, `build.py`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Are the 26 inferred relationships involving `DocumentMeta` (e.g. with `_aggregate_source_metadata()` and `ArticleChunker`) actually correct?**
  _`DocumentMeta` has 26 INFERRED edges - model-reasoned connections that need verification._
- **What connects `legal-rag` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `json` be split into smaller, more focused modules?**
  _Cohesion score 0.05949312714776632 - nodes in this community are weakly interconnected._
- **Why does `LegalStructureParser` connect `LegalStructureParser` to `LineClassifier`, `parse_legal_structure`, `classifier.py`, `DocumentMeta`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `LegalStructureParser` (e.g. with `DocumentMeta` and `LineClassifier`) actually correct?**
  _`LegalStructureParser` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `LegalStructureParser` be split into smaller, more focused modules?**
  _Cohesion score 0.0988235294117647 - nodes in this community are weakly interconnected._