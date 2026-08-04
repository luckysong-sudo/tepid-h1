# M0 Completion Design

Feature Name: m0-completion
Updated: 2026-08-04

## Description

M0 completion uses the existing inventory audit, decontamination command and tokenizer benchmark to produce acceptance evidence for real training assets.

## Components and Interfaces

- `configs/data_inventory.example.json` defines the required inventory shape.
- `tepid_h1.data.audit` validates rights, lineage and PII fields.
- `tepid_h1.data.decontamination` compares training and held-out records.
- `tepid_h1.data.tokenizer_benchmark` produces per-domain ranking evidence.

## Correctness Properties

- Each admitted source has an audited digest and rights record.
- Each tokenizer report includes all required vocabulary sizes and domains.
- Each decontamination report identifies the data and threshold used for comparison.

## Test Strategy

- Unit tests validate inventory rejection conditions and tokenizer candidate selection.
- Acceptance execution runs data audit, decontamination and tokenizer comparison against user-supplied governed assets.
