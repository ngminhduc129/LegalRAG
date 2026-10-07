# LegalRAG Paper Research Plan


## 00_Overview

| Recommended order |   |
|---|---|
| 1 | Literature + RQ |
| 2 | Corpus + OCR |
| 3 | Legal-aware chunking |
| 4 | Retrieval baselines |
| 5 | QA benchmark |
| 6 | Experiments |
| 7 | Error analysis |
| 8 | Paper writing |

| NEW: Result target |   |
|---|---|
| Core result story | Legal-aware chunking → better evidence retrieval; Hybrid/RRF → stronger retrieval; better retrieval → better correctness, citation and faithfulness; error analysis explains failures. |
| Main tables | Table 1 Corpus; Table 5 Retrieval; Table 6 Chunking; Table 7 End-to-End QA; Table 9 Ablation; Table 12 Error Analysis |
| Important rule | All target numbers are planning targets only. Actual paper values must come from experiments; never fabricate results. |


## 00_Overview1

| Item | Plan |
|---|---|
| Project | Vietnamese Labor Law Legal RAG |
| Research direction | Legal-aware Chunking + Hybrid Retrieval + Vietnamese Labor-law QA |
| Main objective | Build and empirically evaluate a RAG system for Vietnamese labor-law question answering. |
| Core research question | How do legal-aware document structure and retrieval strategies affect retrieval quality and grounded legal QA? |
| Target paper type | Empirical research paper / NLP / Information Retrieval / Legal AI |
| Primary contributions | 1) Legal corpus; 2) legal-aware chunking; 3) retrieval comparison; 4) QA benchmark; 5) error analysis |
| Recommended dataset size | 300 questions minimum; 500–1000 preferred |
| Main retrieval baselines | BM25; Dense Retrieval; Hybrid; Hybrid + RRF; optional reranker |
| Main chunking baselines | Fixed-size; paragraph; article-based; legal-aware hierarchical |
| Main evaluation | Recall@k, Precision@k, MRR, nDCG, answer correctness, faithfulness, citation accuracy, completeness, hallucination rate |
| Final paper sections | Introduction; Related Work; Corpus; Methodology; Experimental Setup; Results; Error Analysis; Discussion; Conclusion |
| Recommended duration | 11 weeks |


## 01_Roadmap

| ID | Phase/Task | Detailed action | Expected output | Target | Priority | Status |
|---|---|---|---|---|---|---|
| P01 | Literature & research gap | Collect 30–50 relevant papers and build literature matrix | Literature matrix + research gap | Week 1 | High | Not started |
| P02 | Research questions | Finalize RQ1–RQ4 and hypotheses | Approved RQ document | Week 1 | Critical | Not started |
| P03 | Corpus design | Define document inclusion/exclusion and metadata schema | Corpus specification | Week 2 | Critical | Not started |
| P04 | Document collection | Collect labor-law documents and related instruments | Raw PDF corpus | Week 2 | Critical | Not started |
| P05 | PDF quality audit | Detect text-extractable vs scanned/problematic PDFs | Quality report | Week 2 | High | Not started |
| P06 | OCR pipeline | Run PaddleOCR where extraction is problematic | OCR corpus | Week 2–3 | High | Not started |
| P07 | Text normalization | Clean headers, footers, page numbers, OCR artifacts | Clean text corpus | Week 3 | High | Not started |
| P08 | Legal structure parser | Detect Chapter/Section/Article/Clause/Point hierarchy | Structured corpus | Week 3–4 | Critical | Not started |
| P09 | Chunking baselines | Implement fixed, paragraph and article chunking | Baseline chunks | Week 4 | High | Not started |
| P10 | Legal-aware chunking | Implement hierarchy-preserving chunks + metadata | Proposed chunker | Week 4 | Critical | Not started |
| P11 | Chunk validation | Check duplicate content, repeated keys, missing hierarchy | Validation report | Week 4 | High | Not started |
| P12 | Embedding/indexing | Create dense vector index and metadata store | Dense index | Week 5 | High | Not started |
| P13 | BM25 retrieval | Implement lexical baseline | BM25 retriever | Week 5 | High | Not started |
| P14 | Dense retrieval | Implement embedding-based retrieval | Dense retriever | Week 5 | High | Not started |
| P15 | Hybrid + RRF | Combine BM25 and dense rankings | Hybrid retriever | Week 6 | Critical | Not started |
| P16 | Reranker | Optional cross-encoder/reranker stage | Reranked retriever | Week 6 | Medium | Not started |
| P17 | QA benchmark | Create questions, gold articles, answers and categories | Evaluation dataset | Week 6–7 | Critical | Not started |
| P18 | Evaluation protocol | Define metrics and annotation guidelines | Evaluation protocol | Week 7 | Critical | Not started |
| P19 | Retrieval experiments | Compare retrieval methods and k values | Retrieval results | Week 8 | Critical | Not started |
| P20 | Chunking experiments | Compare chunking strategies | Chunking results | Week 8 | Critical | Not started |
| P21 | RAG generation | Connect best retrieval systems to LLM | RAG systems | Week 8 | High | Not started |
| P22 | Generation experiments | Compare no-RAG vs RAG variants | Generation results | Week 9 | Critical | Not started |
| P23 | Ablation study | Remove components to measure contribution | Ablation results | Week 9 | High | Not started |
| P24 | Human evaluation | Evaluate correctness, citation, completeness, clarity | Human evaluation results | Week 9 | High | Not started |
| P25 | Error analysis | Classify 50–100 failures | Error taxonomy + examples | Week 9 | High | Not started |
| P26 | Statistical analysis | Confidence intervals / significance where appropriate | Statistical analysis | Week 10 | Medium | Not started |
| P27 | Paper figures/tables | Prepare final tables and diagrams | Publication-ready figures | Week 10 | High | Not started |
| P28 | Paper writing | Write all sections | Full manuscript | Week 10–11 | Critical | Not started |
| P29 | Internal review | Check reproducibility, citations, claims and limitations | Revised manuscript | Week 11 | Critical | Not started |
| P30 | Submission preparation | Format according to target venue and prepare supplementary material | Submission package | Week 11 | High | Not started |


## 02_RQs

| RQ | Theme | Research question | Working hypothesis | Experiment |
|---|---|---|---|---|
| RQ1 | Chunking | How does legal-aware chunking affect retrieval performance in Vietnamese labor-law documents? | Legal-aware chunking improves Recall@k/MRR/nDCG while preserving legal context. | Chunking experiment |
| RQ2 | Retrieval | Which retrieval strategy is most effective for Vietnamese labor-law QA? | Hybrid retrieval outperforms BM25 or dense retrieval alone. | Retrieval experiment |
| RQ3 | Generation | Does improved retrieval improve grounded and legally correct answers? | Better retrieval improves correctness, citation accuracy and faithfulness. | RAG experiment |
| RQ4 | Error analysis | What are the dominant failure modes in Vietnamese labor-law RAG? | Failures can be categorized into retrieval, OCR, chunking, legal-version, citation and generation errors. | Error analysis |


## 03_Literature

| ID | Area | Reference target | Why needed | Action | Notes |
|---|---|---|---|---|---|
| L01 | RAG foundation | Lewis et al. (2020), Retrieval-Augmented Generation | RAG architecture | Read + cite | Need exact bibliographic metadata |
| L02 | Dense retrieval | DPR / dense passage retrieval literature | Dense retrieval baseline | Read + cite | Need exact bibliographic metadata |
| L03 | Hybrid retrieval | BM25 + dense / RRF literature | Hybrid retrieval and fusion | Read + cite | Need exact bibliographic metadata |
| L04 | Legal NLP | Legal QA / legal retrieval papers | Legal-domain challenges | Read + cite | Find Vietnamese + international work |
| L05 | Legal RAG | Recent legal RAG studies | Research gap | Read + compare | Focus on retrieval, grounding, citations |
| L06 | Vietnamese NLP | Vietnamese embedding / retrieval papers | Language-specific challenges | Read + cite | Find benchmark/model choices |
| L07 | Vietnamese legal AI | Vietnamese legal QA / legal AI papers | Local research gap | Read + cite | Prioritize peer-reviewed sources |
| L08 | OCR | OCR for scanned legal documents | Corpus quality | Read + cite | Support OCR pipeline |
| L09 | Evaluation | RAG/QA evaluation literature | Metric justification | Read + cite | Human + automatic evaluation |
| L10 | Hallucination | Grounded generation / factuality literature | Legal hallucination | Read + cite | Support faithfulness metrics |


## 04_Corpus_OCR

| ID | Task | Detailed requirement | Output | Research note |
|---|---|---|---|---|
| C01 | Document inventory | List all included laws, decrees, circulars, resolutions and related instruments | Document list + spreadsheet | Must define inclusion criteria |
| C02 | Metadata | Document number, title, issuer, issue date, effective date, status, source | Metadata table | Critical |
| C03 | Version control | Record effective/expired/amended status | Version metadata | Avoid outdated answers |
| C04 | PDF audit | Measure extractable text quality and OCR need | PDF quality report | Use character count/Vietnamese ratio/suspicious patterns |
| C05 | OCR | Run PaddleOCR on problematic PDFs | OCR text | Keep original page references |
| C06 | Normalization | Remove page headers/footers, normalize whitespace, preserve legal numbering | Clean text | Do not destroy legal structure |
| C07 | Structure parsing | Detect Chapter/Section/Article/Clause/Point | Hierarchy tree | Critical |
| C08 | Source traceability | Keep page/document/source identifiers for citations | Traceability metadata | Critical for legal citation |
| C09 | Corpus statistics | Documents, pages, chars, articles, clauses, chunks | Corpus statistics table | Paper Table 1 |


## 05_Chunking

| ID | Strategy | Implementation | Role | Metrics | Action |
|---|---|---|---|---|---|
| CH01 | Fixed-size | e.g. 300/500/800 tokens with overlap | Baseline | Recall@k, MRR | Implement |
| CH02 | Paragraph | Split by paragraph | Baseline | Recall@k, MRR | Implement |
| CH03 | Article | One Article or logical Article units | Legal baseline | Recall@k, MRR | Implement |
| CH04 | Clause | Article + Clause as retrieval unit | Fine-grained baseline | Recall@k, MRR | Implement |
| CH05 | Legal-aware | Preserve Article–Clause–Point hierarchy and metadata | Proposed | Recall@k, MRR, context completeness | Main contribution |
| CH06 | Metadata enrichment | Add document type, number, Article, Chapter, effective status | Enhancement | Retrieval + citation | Test separately if possible |
| CH07 | Validation | Detect duplicate content, repeated structure keys, missing hierarchy | Quality control | Error counts | Use existing checker |
| CH08 | Context preservation | Check whether references like 'khoản 1 Điều 35' remain resolvable | Legal validity | Manual sample | Important |


## 06_Retrieval

| ID | Method | Implementation | Role | Evaluation | Priority |
|---|---|---|---|---|---|
| R01 | BM25 | Tokenize/normalize Vietnamese legal text and build inverted index | Lexical baseline | Recall@5/10/20, MRR, nDCG | Required |
| R02 | Dense | Embed chunks and query; cosine/dot-product search | Semantic baseline | Recall@5/10/20, MRR, nDCG | Required |
| R03 | Hybrid | Combine BM25 and dense candidate lists | Main method | Recall@5/10/20, MRR, nDCG | Required |
| R04 | RRF | Fuse ranked lists without requiring score calibration | Fusion method | Compare against individual retrievers | Required |
| R05 | Reranker | Rerank top 20/50 candidates with cross-encoder or suitable model | Optional enhancement | MRR/nDCG + latency | Recommended if resources allow |
| R06 | k sensitivity | Evaluate k = 1, 3, 5, 10, 20 | Robustness | Metric curves | Required |
| R07 | Latency | Measure retrieval latency and end-to-end latency | Practicality | ms/query | Recommended |


## 07_QA_Dataset

| ID | Task | Requirement | Output | Target/Note |
|---|---|---|---|---|
| D01 | Question collection | Create realistic questions from labor-law topics | Question set | 300 minimum / 500–1000 preferred |
| D02 | Categories | Contract; wage; working time; leave; discipline; termination; social insurance; etc. | Category labels | Balanced categories |
| D03 | Difficulty | Easy factual; multi-clause; multi-document; reasoning; ambiguous | Difficulty labels | At least 3 levels |
| D04 | Gold source | Identify gold Article/Clause/Point(s) | Gold document IDs | Critical |
| D05 | Gold answer | Expert-written or verified reference answer | Reference answer | Critical |
| D06 | Question provenance | Record whether question is synthetic, manually written or adapted | Provenance field | Transparency |
| D07 | Split | Train/dev/test only if needed; otherwise test-only benchmark for retrieval/QA | Dataset split | Avoid leakage |
| D08 | Annotation guideline | Define relevance, answer correctness and citation correctness | Guideline | Needed for human evaluation |
| D09 | Quality control | Second reviewer for a subset | Agreement report | Recommended |


## 08_Experiments

| ID | Experiment | Comparison | RQ | Metrics | Controls | Expected insight |
|---|---|---|---|---|---|---|
| E01 | Chunking | Fixed vs paragraph vs article vs legal-aware | RQ1 | Recall@5/10, MRR, nDCG | Same corpus/query set | Legal-aware should improve context preservation |
| E02 | Retrieval | BM25 vs dense vs hybrid | RQ2 | Recall@5/10/20, MRR, nDCG | Same chunks + queries | Hybrid expected strongest |
| E03 | Fusion | Dense/BM25 vs RRF | RQ2 | MRR/nDCG | Same candidates | Test RRF contribution |
| E04 | Reranking | Hybrid vs hybrid+reranker | RQ2 | MRR/nDCG + latency | Same candidate pool | Optional |
| E05 | RAG baseline | LLM without retrieval vs RAG | RQ3 | Correctness, faithfulness, citation accuracy | Same question set | Measure hallucination |
| E06 | RAG retrieval | BM25-RAG vs Dense-RAG vs Hybrid-RAG | RQ3 | Correctness, faithfulness, citation accuracy | Same LLM/prompt | Isolate retrieval effect |
| E07 | Ablation chunking | Best retrieval + different chunkers | RQ1/RQ3 | Retrieval + answer metrics | Same model | Shows chunking impact |
| E08 | Ablation metadata | With vs without legal metadata | RQ1/RQ2 | Retrieval metrics | Same chunks | Optional |
| E09 | Error analysis | 50–100 failed cases | RQ4 | Error category distribution | Best system | Qualitative + quantitative |
| E10 | Human evaluation | Expert/annotator scoring | RQ3 | Correctness, citation, completeness, clarity | Sample of test set | Recommended |


## 09_Metrics

| ID | Metric | Definition | Stage | Importance | Implementation note |
|---|---|---|---|---|---|
| M01 | Recall@k | Fraction of queries whose gold source appears in top-k | Retrieval | Primary | Gold article IDs required |
| M02 | Precision@k | Relevant retrieved chunks / k | Retrieval | Secondary | Requires relevance labels |
| M03 | MRR | Average reciprocal rank of first relevant result | Retrieval | Primary | Good for ranking quality |
| M04 | nDCG@k | Ranking quality with graded relevance | Retrieval | Primary | Useful if multiple relevant provisions |
| M05 | Answer correctness | Whether answer is legally correct | Generation | Primary | Human or validated judge |
| M06 | Faithfulness | Whether claims are supported by retrieved evidence | Generation | Primary | Critical for legal RAG |
| M07 | Citation accuracy | Whether cited Article/Clause actually supports claim | Generation | Primary | Critical |
| M08 | Completeness | Whether important legal conditions/exceptions are included | Generation | Secondary | Human evaluation |
| M09 | Hallucination rate | Unsupported legal claims / total legal claims | Generation | Primary | Define annotation rule |
| M10 | Latency | Retrieval and end-to-end response time | System | Secondary | Practicality |


## 10_Error_Analysis

| ID | Error type | Definition | Typical symptom | Analysis output |
|---|---|---|---|---|
| ER01 | Retrieval failure | Gold Article not retrieved | Retriever returns irrelevant chunks | Count + example |
| ER02 | OCR error | OCR corrupts legal text/numbering | Missing or malformed words/numbers | Count + example |
| ER03 | Chunking failure | Relevant information split incorrectly | Missing clause/point context | Count + example |
| ER04 | Version error | System retrieves outdated/ineffective provision | Wrong legal status | Count + example |
| ER05 | Cross-reference failure | Question requires another Article/document | Retrieved context incomplete | Count + example |
| ER06 | Citation error | Citation does not support claim | Wrong Article/Clause | Count + example |
| ER07 | Hallucination | LLM adds unsupported legal statement | Not grounded in context | Count + example |
| ER08 | Incomplete answer | Correct but missing condition/exception | Partial legal answer | Count + example |
| ER09 | Ambiguous query | Question underspecified | Multiple plausible interpretations | Count + example |
| ER10 | Multi-document reasoning | Answer requires combining provisions | Single retrieval insufficient | Count + example |


## 11_Paper_Structure

| Section | Title | Required content | Suggested length | Writing requirements | Purpose |
|---|---|---|---|---|---|
| 1 | Introduction | 1.1 Background; 1.2 Problem; 1.3 Challenges; 1.4 Research gap; 1.5 Contributions | 1–1.5 pages | Need citations for every factual claim | Research motivation + gap |
| 2 | Related Work | RAG; Legal NLP/RAG; Vietnamese NLP; Vietnamese legal AI | 1.5–2.5 pages | Compare studies, do not only summarize | Positioning |
| 3 | Legal Corpus Construction | Document selection; PDF/OCR; normalization; legal structure; metadata; corpus statistics | 1.5–2 pages | Explain inclusion/exclusion and reproducibility | Dataset contribution |
| 4 | Methodology | Problem formulation; chunking; BM25; dense; hybrid/RRF; reranking; generation; citation mechanism | 2–3 pages | Use architecture diagram | Technical contribution |
| 5 | Experimental Setup | Dataset; baselines; implementation; metrics; evaluation protocol | 1.5–2 pages | Specify all settings | Reproducibility |
| 6 | Results | Chunking; retrieval; generation; ablation; human evaluation | 2–3 pages | Tables + significance/CI where possible | Main evidence |
| 7 | Error Analysis | Failure taxonomy; representative examples; causes | 1–2 pages | Show real legal cases | Research insight |
| 8 | Discussion | Why methods work/fail; implications; limitations; ethics/legal disclaimer | 1–1.5 pages | Avoid overclaiming | Interpretation |
| 9 | Conclusion | Findings; contributions; future work | 0.5–1 page | Answer RQs explicitly | Closure |


## 12_Timeline

| Week | Focus | Tasks | Deliverable |
|---|---|---|---|
| Week 1 | Literature + RQ | 30–50 papers; literature matrix; finalize RQ1–RQ4 | Research gap + RQ document |
| Week 2 | Corpus | Collect documents; metadata; PDF quality audit | Raw corpus + audit |
| Week 3 | OCR + cleaning | PaddleOCR; normalization; source traceability | Clean corpus |
| Week 4 | Structure + chunking | Parse hierarchy; implement all chunking strategies; validate | Chunk datasets + validation report |
| Week 5 | Retrieval baselines | BM25 + dense retrieval | Baseline results |
| Week 6 | Hybrid + dataset | RRF; optional reranker; start 300–1000 QA benchmark | Hybrid retriever + dataset draft |
| Week 7 | Evaluation protocol | Gold sources; annotation guidelines; finalize test set | Frozen evaluation set |
| Week 8 | Main experiments | Chunking + retrieval + RAG experiments | Raw experiment results |
| Week 9 | Ablation + human eval | Ablations; human evaluation; collect failure cases | Final evaluation results |
| Week 10 | Analysis + figures | Statistics; error analysis; tables/figures | Results section material |
| Week 11 | Writing + submission | Complete manuscript; review; formatting; supplementary files | Submission-ready paper |


## 13_Final_Checklist

| ID | Section | Completion criterion | Evidence/link | Status |
|---|---|---|---|---|
| C01 | Introduction | Research problem is explicit |  | Not started |
| C02 | Introduction | Research gap is supported by literature |  | Not started |
| C03 | Introduction | Contributions are measurable |  | Not started |
| C04 | Related Work | At least 30 strong references reviewed |  | Not started |
| C05 | Corpus | Document inclusion/exclusion criteria defined |  | Not started |
| C06 | Corpus | OCR quality procedure documented |  | Not started |
| C07 | Corpus | Legal structure parsing validated |  | Not started |
| C08 | Chunking | Baselines implemented |  | Not started |
| C09 | Chunking | Legal-aware chunking implemented |  | Not started |
| C10 | Chunking | Chunk validation report completed |  | Not started |
| C11 | Retrieval | BM25 implemented |  | Not started |
| C12 | Retrieval | Dense retrieval implemented |  | Not started |
| C13 | Retrieval | Hybrid/RRF implemented |  | Not started |
| C14 | Dataset | Gold Article/Clause labels completed |  | Not started |
| C15 | Dataset | Test set frozen before final comparison |  | Not started |
| C16 | Evaluation | Recall/MRR/nDCG implemented |  | Not started |
| C17 | Evaluation | Generation evaluation protocol defined |  | Not started |
| C18 | Experiments | Chunking experiment completed |  | Not started |
| C19 | Experiments | Retrieval experiment completed |  | Not started |
| C20 | Experiments | RAG comparison completed |  | Not started |
| C21 | Experiments | Ablation completed |  | Not started |
| C22 | Evaluation | Human evaluation completed if feasible |  | Not started |
| C23 | Analysis | 50–100 failure cases classified |  | Not started |
| C24 | Paper | All RQs explicitly answered in Results/Discussion |  | Not started |
| C25 | Reproducibility | Code/config/model/version recorded |  | Not started |
| C26 | Final | Limitations and ethical/legal disclaimer included |  | Not started |


## 14_Data_Schema

| Table | Field | Purpose |
|---|---|---|
| documents | document_id | Unique ID for each legal document |
| documents | document_number | e.g. 45/2019/QH14 |
| documents | title | Official title |
| documents | document_type | Law/Decree/Circular/Resolution/etc. |
| documents | issuer | Issuing authority |
| documents | issue_date | Date issued |
| documents | effective_date | Effective date |
| documents | status | Effective/expired/amended/etc. |
| documents | source_url | Official/public source |
| chunks | chunk_id | Unique chunk ID |
| chunks | document_id | Parent document |
| chunks | chapter | Chapter metadata |
| chunks | section | Section metadata |
| chunks | article | Article number |
| chunks | clause | Clause number |
| chunks | point | Point letter |
| chunks | text | Chunk content |
| chunks | page_start | Original page start |
| chunks | page_end | Original page end |
| questions | question_id | Unique QA ID |
| questions | question | User question |
| questions | category | Labor-law topic |
| questions | difficulty | Easy/Medium/Hard |
| questions | gold_chunk_ids | Relevant chunk IDs |
| questions | gold_article_ids | Relevant Article/Clause IDs |
| questions | reference_answer | Verified reference answer |
| questions | provenance | Manual/Synthetic/Adapted |


## 15_Results_Overview

| ID | RQ/Area | Result/Table | What it should demonstrate | Main metrics | Target/Note |
|---|---|---|---|---|---|
| RES01 | Corpus | Corpus statistics | Demonstrate that the corpus is sufficiently broad and structurally rich for Vietnamese labor-law QA. | Documents; pages; Articles; Clauses; chunks; OCR-required PDFs | Target ranges only; do not fabricate numbers |
| RES02 | OCR | PDF extraction/OCR quality | Show that OCR is necessary for part of the legal corpus and that the pipeline improves text usability. | Vietnamese character ratio; suspicious patterns; manual quality | Report before/after where possible |
| RES03 | Chunking | Chunking statistics | Show how each chunking strategy changes number/size/structure of retrieval units. | # chunks; avg tokens; structure preservation | Legal-aware should preserve hierarchy |
| RES04 | Chunking | Chunk validation | Demonstrate structural integrity of proposed chunks. | Structural errors; duplicates; repeated keys; missing hierarchy | Target: structural errors ≈ 0; duplicates very low |
| RES05 | RQ1 | Chunking comparison | Test whether legal-aware chunking improves retrieval. | Recall@5/10/20; MRR; nDCG@10 | Expected pattern: legal-aware competitive/best |
| RES06 | RQ2 | Retrieval comparison | Compare BM25, dense, hybrid, RRF and optional reranker. | Recall@k; MRR; nDCG | Expected pattern: hybrid/RRF > single retriever |
| RES07 | RQ3 | End-to-end RAG | Test whether better retrieval leads to better legal answers. | Correctness; faithfulness; citation accuracy; completeness; hallucination | Expected pattern: best retriever produces strongest grounded QA |
| RES08 | RQ3 | Citation evaluation | Measure whether cited Articles/Clauses actually support claims. | Correct; partial; incorrect; missing citation | Critical for Legal RAG |
| RES09 | RQ1/RQ2/RQ3 | Ablation | Measure contribution of chunking, BM25, dense, RRF and reranking. | Full system vs component removal | Full system should be competitive/best |
| RES10 | RQ4 | Category-wise analysis | Identify strong/weak labor-law categories. | Contracts; wages; hours; leave; discipline; termination; etc. | Explain differences |
| RES11 | RQ4 | Difficulty-wise analysis | Measure robustness on easy/medium/hard questions. | Recall; correctness; faithfulness | Performance will normally decline with difficulty |
| RES12 | RQ4 | Error analysis | Identify dominant failure modes. | Retrieval; OCR; chunking; version; cross-reference; citation; hallucination; incompleteness | 50–100 failures recommended |
| RES13 | System | Efficiency | Quantify practical cost of retrieval/reranking/generation. | Latency per stage; total latency | Use for trade-off discussion |


## 16_Result_Tables

| ID | Table | Title | Purpose | Columns/Metrics | Desired outcome | Priority/Note |
|---|---|---|---|---|---|---|
| T01 | Table 1 | Corpus Statistics | Corpus | Documents; laws; decrees; circulars; pages; Articles; Clauses; chunks | 50–150 docs; 3,000–10,000 pages; 1,000–3,000 Articles; 5,000–20,000 chunks | Descriptive, not a performance target |
| T02 | Table 2 | PDF Extraction and OCR Statistics | OCR | # docs; avg pages; Vietnamese ratio; suspicious pattern rate; OCR-required % | Clearly separate native/scanned/mixed PDFs | Show OCR necessity |
| T03 | Table 3 | Chunking Statistics | Chunking | # chunks; avg tokens; avg chars; structure preserved | Legal-aware should preserve Article–Clause–Point hierarchy | Descriptive |
| T04 | Table 4 | Chunk Validation | Chunking | Structural errors; duplicate content; repeated keys; missing hierarchy | Structural errors = 0; duplicates/repeated keys ≈ 0 or very low | Use current checker as baseline |
| T05 | Table 5 | Retrieval Performance | RQ2 | Recall@5/10/20; MRR; nDCG@10 | Hybrid/RRF should outperform BM25/dense if hypothesis holds | Main result |
| T06 | Table 6 | Impact of Chunking Strategy | RQ1 | Recall@k; MRR; nDCG | Legal-aware should be competitive/best | Main result |
| T07 | Table 7 | End-to-End QA Performance | RQ3 | Correctness; faithfulness; citation; completeness; hallucination | RAG > LLM-only; stronger retrieval should improve grounded QA | Main result |
| T08 | Table 8 | Legal Citation Evaluation | RQ3 | Correct/partial/incorrect/missing citations | Correct citation ↑; incorrect/missing ↓ | Main result |
| T09 | Table 9 | Ablation Study | RQ1–RQ3 | Recall; MRR; correctness; citation | Removing useful components should reduce performance | Main result |
| T10 | Table 10 | Category-wise Performance | RQ4 | # questions; Recall@10; correctness; citation | Explain weak/strong labor-law categories | Analysis |
| T11 | Table 11 | Difficulty-wise Performance | RQ4 | # questions; Recall; correctness; faithfulness | Easy ≥ Medium ≥ Hard is expected but must be measured | Analysis |
| T12 | Table 12 | Error Distribution | RQ4 | Count + percentage by error type | Total errors = 100%; identify dominant failures | Analysis |
| T13 | Table 13 | Representative Failure Cases | RQ4 | Question; gold provision; retrieved provision; error; explanation | 5–10 representative cases | Qualitative |
| T14 | Table 14 | Human Evaluation | RQ3 | Correctness; relevance; citation; completeness; clarity (1–5) | Hybrid/Reranker expected strongest | Recommended |
| T15 | Table 15 | System Efficiency | System | Retrieval; reranking; generation; total latency | Report accuracy/latency trade-off | Optional |


## 17_Target_Patterns

| ID | Stage | System | Metric | Role | Value to enter after experiment | Desired pattern |
|---|---|---|---|---|---|---|
| R01 | Retrieval | BM25 | Recall@10 | Baseline | Record actual | Should be a strong lexical baseline |
| R02 | Retrieval | Dense | Recall@10 | Baseline | Record actual | Should help semantic paraphrases |
| R03 | Retrieval | Hybrid | Recall@10 | Main | Record actual | Target: higher than both single methods |
| R04 | Retrieval | Hybrid + RRF | Recall@10 | Main | Record actual | Target: highest/near-highest |
| R05 | Retrieval | Hybrid + Reranker | MRR | Optional | Record actual | Target: highest/near-highest |
| R06 | Chunking | Fixed-size | MRR | Baseline | Record actual | Reference point |
| R07 | Chunking | Paragraph | MRR | Baseline | Record actual | Reference point |
| R08 | Chunking | Article | MRR | Baseline | Record actual | Should preserve legal unit |
| R09 | Chunking | Legal-aware | MRR | Proposed | Record actual | Target: best/competitive |
| R10 | Generation | LLM only | Correctness | Baseline | Record actual | Expected weaker grounding |
| R11 | Generation | BM25-RAG | Correctness | Baseline | Record actual | Should improve over LLM-only if retrieval is useful |
| R12 | Generation | Dense-RAG | Correctness | Baseline | Record actual | Should improve semantic matching |
| R13 | Generation | Hybrid-RAG | Correctness | Main | Record actual | Target: higher than single-retriever RAG |
| R14 | Generation | Hybrid + Reranker | Correctness | Optional | Record actual | Target: highest/near-highest |
| R15 | Generation | LLM only | Hallucination rate | Baseline | Record actual | Expected highest |
| R16 | Generation | Hybrid + Reranker | Hallucination rate | Main | Record actual | Target: lower than LLM-only and simpler RAG |


## 18_Desired_Figures

| ID | Figure | Title | What to visualize | Paper section | Priority |
|---|---|---|---|---|---|
| F01 | Figure 1 | Overall Legal RAG Architecture | Documents → OCR/extraction → legal parser → chunking → BM25/dense → RRF → reranker → LLM → answer+citation | Methodology | Required |
| F02 | Figure 2 | Retrieval Performance | Bar chart of Recall@10 / MRR across retrieval methods | Results | Required |
| F03 | Figure 3 | Chunking Comparison | MRR or Recall@10 across chunking strategies | Results | Required |
| F04 | Figure 4 | Hallucination Comparison | Hallucination rate across LLM/RAG systems | Results | Recommended |
| F05 | Figure 5 | Performance by Difficulty | Performance across easy/medium/hard questions | Analysis | Recommended |
| F06 | Figure 6 | Error Distribution | Bar chart of error categories | Error analysis | Recommended |


## 19_Final_Result_Package

| ID | Type | Artifact | Requirement | Placement |
|---|---|---|---|---|
| M01 | Required main table | Corpus Statistics | Must have | Table 1 |
| M02 | Required main table | Chunking Comparison | Must have | Table 6 |
| M03 | Required main table | Retrieval Comparison | Must have | Table 5 |
| M04 | Required main table | End-to-End RAG Evaluation | Must have | Table 7 |
| M05 | Required main table | Ablation Study | Must have | Table 9 |
| M06 | Required main table | Error Analysis | Must have | Table 12 |
| M07 | Appendix | OCR statistics | Recommended | Table 2 |
| M08 | Appendix | Chunk validation | Recommended | Table 4 |
| M09 | Appendix | Category/difficulty breakdown | Recommended | Tables 10–11 |
| M10 | Appendix | Representative examples | Recommended | Table 13 |
| M11 | Optional | Human evaluation | Strongly recommended | Table 14 |
| M12 | Optional | Efficiency | Useful for system paper | Table 15 |
| M13 | Required figure | Architecture | Must have | Figure 1 |
| M14 | Required figure | Retrieval comparison | Must have | Figure 2 |
| M15 | Recommended figure | Chunking comparison | Recommended | Figure 3 |
| M16 | Recommended figure | Hallucination/error | Recommended | Figures 4–6 |
