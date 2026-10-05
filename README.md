# Federated Collaborative IDS for Hybrid Cloud

Starter repository for the finalized research roadmap.

## Phase 1 scope
- CICIDS2017 loading
- cleaning, validation, deduplication
- fine-grained + 5-class label harmonization
- Train/Validation/Test split BEFORE feature optimization
- locked test set
- IID, mild non-IID, and extreme non-IID client partitions
- leakage checks and summary reports

## Client definitions
- Client 1 — Private Cloud: Normal + Web Attacks (SQLi, XSS, Brute Force)
- Client 2 — Public Cloud: Normal + Scan/Probe (PortScan, Probe, Infiltration)
- Client 3 — Edge/Hybrid Cloud: Normal + DDoS/DoS/Botnet

## Quick start
```bash
python -m venv .venv
```
Windows:
```bash
.venv\Scripts\activate
```
Install:
```bash
pip install -r requirements.txt
```
Place CICIDS2017 CSVs in `data/raw/` and run:
```bash
python scripts/phase1_prepare_data.py
```
A small sample CSV is already included so you can test the scaffold immediately.

## Research rule
The Test set is split and locked before EVO-GA or any feature selection.
