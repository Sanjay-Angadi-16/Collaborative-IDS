PHASE 17A — POISONING THREAT-MODEL + HARNESS VALIDATION
=======================================================

Purpose
-------
This package wraps your existing FedProx70 baseline WITHOUT changing the
baseline training objective or sample-weighted aggregation.

It adds one explicit malicious-client routing boundary and validates it with
identity controls only:

A1 CLEAN
A2 1 malicious client, NOOP
A3 2 malicious clients, NOOP
A4 1 malicious client, UPDATE_SCALE=1.0
A5 2 malicious clients, UPDATE_SCALE=1.0

No damaging poisoning is injected in Phase17A.

Required existing project files
-------------------------------
scripts\phase10_fedprox70.py
scripts\phase9_fedavg70.py

The package was built against these exact SHA256 hashes:

phase10_fedprox70.py
e29928a1885eb949bfb58e97ded154875607abf910d8cb65f6b134c17883194b

phase9_fedavg70.py
5fb9b40cd44ecab926f55fe6e51b32b9209a8c208011d2388e5a2be3086696a5

Data policy
-----------
TRAIN:
data\federated\non_iid\client_1.csv ... client_5.csv

DEVELOPMENT EVALUATION:
data\processed\validation.csv

NOT USED:
data\processed\test_locked.csv

Installation
------------
Extract/copy the ZIP CONTENTS into the PROJECT ROOT so these files appear:

src\federated\poisoning\__init__.py
src\federated\poisoning\attacks.py
src\federated\poisoning\metrics.py
configs\phase17a_threat_model.json
scripts\phase17a_poisoning_harness.py

Run a quick smoke test first
----------------------------
python scripts\phase17a_poisoning_harness.py --scenario non_iid --seed 42 --rounds 1 --matrix smoke

Run the full Phase17A matrix
----------------------------
python scripts\phase17a_poisoning_harness.py --scenario non_iid --seed 42

Expected scientific outcome
---------------------------
A1 ~= A2 ~= A3 ~= A4 ~= A5

NOOP should reproduce the clean state exactly.
UPDATE_SCALE=1.0 should be within the frozen numerical tolerance.

If this passes:
    the attack-routing harness is frozen
    Phase17B may introduce label flipping

If this fails:
    DO NOT continue to damaging attacks
    first fix the harness because later degradation would be ambiguous.

Outputs
-------
results\phase17\phase17a\seed_42\...
artifacts\poisoning\phase17a\phase17a_threat_model_manifest.json
