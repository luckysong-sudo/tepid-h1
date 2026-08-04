# M0 Completion Requirements

## Glossary

- **Production inventory**: The signed inventory describing data sources eligible for training.
- **Evaluation corpus**: The governed zh, en and code corpus used for tokenizer comparison.
- **Tokenizer report**: The machine-readable comparison and selection result for 64K, 80K and 96K candidates.

## Requirements

### Production Data Admission

**User Story:** AS a data governance owner, I want a production inventory that records source lineage and rights evidence, so that training sources have documented commercial approval.

1. WHEN a production source enters the inventory, the audit SHALL require a stable snapshot, SHA-256 digest, license identifier, commercial-use approval, rights evidence, PII disposition, language classification, domain classification and token estimate.
2. IF a production source has a restricted, prohibited, unknown, PII-present or PII-unassessed status, the audit SHALL return a blocking finding.
3. WHEN the production inventory is audited, the audit SHALL report zero blocking findings before M0 acceptance.

### Tokenizer Selection

**User Story:** AS a model owner, I want a reproducible tokenizer selection report, so that the selected vocabulary has measurable zh, en and code performance.

1. WHEN the evaluation corpus and three candidate tokenizer JSON files are supplied, the comparison SHALL measure 64K, 80K and 96K candidates.
2. WHEN the comparison completes, the report SHALL bind results to the evaluation corpus digest and report compression and throughput for zh, en and code.
3. WHEN a candidate is selected, the report SHALL record the ranking and scoring weights.

### Decontamination Evidence

**User Story:** AS an evaluation owner, I want repository-level decontamination evidence, so that held-out benchmarks remain independent from training sources.

1. WHEN the production inventory is accepted, the inventory SHALL record the comparison method, benchmark sets, completion timestamp and report location.
2. WHEN near-duplicate detection executes, the report SHALL record its n-gram size and similarity threshold.
