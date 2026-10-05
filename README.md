Federated Collaborative Intrusion Detection System (IDS)
Project status: Phase 17H.4 completed  
Next planned phase: Phase 18 — Uncertainty-Aware Open-World IDS  
Last updated: October 2026
---
1. Project Overview
This project implements a Federated Collaborative Intrusion Detection System for hybrid / multi-cloud environments.
The current research pipeline includes:
Federated Learning
Severe controlled non-IID clients
Centralized and local baselines
FedAvg / FedProx and other aggregation methods
Evolutionary and GA-based feature selection
Random Forest
Personalized Autoencoders
FedProx70 MLP
Federated CNN-BiLSTM
Adaptive FAST / DEEP inference
Risk fusion
Privacy-aware alert sharing
Cross-cloud correlation
GNN / GCN-based correlation
Byzantine / poisoning robustness
Dashboard integration
CSV and NPZ sequence inference
False-negative diagnostics
---
2. Frozen Core Configuration
Item	Current Setting
Dataset family	CICIDS2017-derived
Clients	5
Known classes	6
Original features	80
Active features	70
Sequence length	20 flows
FL setting	Severe controlled non-IID
Primary hardware	CPU
Current development evaluation	Validation
Locked test in Phase 17H	NO
Sequence semantics	Source-ordered, not timestamp-confirmed temporal
Classes
ID	Class
0	BENIGN
1	DoS Hulk
2	DDoS
3	PortScan
4	DoS GoldenEye
5	FTP-Patator
---
3. Severe Non-IID Client Design
Client	Primary Attack
Client 1	DoS Hulk
Client 2	DDoS
Client 3	PortScan
Client 4	DoS GoldenEye
Client 5	FTP-Patator
Each client contains BENIGN traffic plus one primary attack family.
---
4. Overall Detection Architecture
```text
                     Network Flow Data
                            |
                            v
                  Frozen Preprocessing
                     80 -> 70 Features
                            |
                +-----------+-----------+
                |                       |
                v                       v
         Current Flow              Flow Sequence
        X[:, -1, :]                 X[:, :, :]
                |                       |
       +--------+--------+              |
       |        |        |              |
       v        v        v              v
   Local RF   Local AE  FedProx70   CNN-BiLSTM
       |        |        |              |
       +--------+--------+--------------+
                            |
                            v
                  Adaptive Gate / Fusion
                            |
                            v
                  Final IDS Prediction
                            |
                            v
                   Severity / Local Alert
                            |
                            v
                 Privacy-Safe Alert Sharing
                            |
                            v
                  Cross-Cloud Correlation
```
---
5. Project Phase History
Phase 1 / 1B — Dataset Preparation
Loaded CICIDS2017-derived data.
Cleaned invalid values.
Standardized labels.
Created six-class schema.
Prepared train / validation splits.
Prepared client datasets.
Phase 2 / 2B — Federated Client Construction
Created:
IID federation
Natural non-IID federation
Controlled severe non-IID federation
Final project configuration uses controlled severe non-IID.
Phase 3A — Frozen Preprocessing
Frozen:
Feature order
Scaling / preprocessing
Label mapping
Active feature list
```text
80 original features
        |
        v
10 constant / unusable removed
        |
        v
70 active features
```
Phase 3B — Centralized MLP Baseline
Established a centralized non-federated reference.
Phase 3C — Local-Only Baselines
Measured client-local performance before federation.
Phase 4 — FedAvg
Implemented sample-weighted Federated Averaging.
Phase 5 / 5B — Evolutionary Feature Selection
Implemented evolutionary feature selection, later made federation-aware.
Phase 6 — Federated-Aware Genetic Algorithm
Implemented GA-based feature search using federated validation performance.
Phase 7 — Hybrid EVO-GA
Combined evolutionary optimization and GA.
```text
70 active features
      |
      v
37 selected features
```
Phase 8 — Aggregation Algorithm Comparison
Compared:
FedAvg
FedProx
FedAvgM
FedNova-style
SCAFFOLD-style
Coordinate Median
Trimmed Mean
Phase 9 — Repeated-Seed Statistical Analysis
Seeds:
```text
[42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
```
Added:
Rankings
Confidence intervals
Wilcoxon tests
Friedman tests
Multi-seed stability analysis
Phase 10 — FedProx70 Feature-Preservation Ablation
Compared reduced features vs full 70-feature representation.
Phase 11 — Knowledge Transfer
Evaluated seen, unseen, and cross-client attack transfer.
Phase 12 / 12B — Personalized FedProx
Added client personalization and regularized personalization.
Phase 13 — Minority / Attack Retention
Studied whether minority attack behavior was lost during feature selection, federation, or personalization.
---
6. Phase 14 — Adaptive Client-Side IDS
Client-Local Random Forest
One RF per client.
Each RF primarily learns:
```text
BENIGN + local attack family
```
The RF is a local specialist, not a universal six-class classifier.
Personalized Autoencoder
Architecture:
```text
70 -> 48 -> 24 -> 12 -> 24 -> 48 -> 70
```
Training:
```text
Epochs          = 20
Batch size      = 2048
Learning rate   = 0.001
Weight decay    = 1e-5
Training data   = local BENIGN only
Threshold       = held-out BENIGN P99
Loss            = MSE
```
Role: anomaly / novelty supporting evidence.
FedProx70 MLP
```text
Seed = 42
mu   = 0.01
```
Federated CNN-BiLSTM
Input:
```text
20 x 70 source-ordered flow sequence
```
Parameters:
```text
126,982
```
Frozen configuration:
```text
FedProx mu      = 0.01
Best round      = 20
Local epochs    = 1
Batch size      = 256
Learning rate   = 0.001
Weight decay    = 0.0001
Gradient clip   = 5.0
Seed            = 42
```
---
7. Phase 14 Adaptive Gate
FAST Path
Condition:
```text
RF predicts non-BENIGN
AND
confidence >= 0.95
```
Policy:
```text
Prediction  = RF class
Confidence  = RF confidence
Severity    = HIGH
risk_score  = NULL
```
DEEP Path
Fusion weights:
```text
Random Forest = 0.20
FedProx70     = 0.30
CNN-BiLSTM   = 0.35
Autoencoder  = 0.15
```
Severity:
```text
risk < 0.20  -> NORMAL
risk < 0.40  -> LOW
risk < 0.60  -> MEDIUM
risk < 0.80  -> HIGH
risk >= 0.80 -> CRITICAL
```
CRITICAL is full-fusion only.
---
8. Phase 14H — Full Validation
```text
50,000 unique validation sequences
x
5 client exposures
=
250,000 client exposures
```
Metric	Adaptive Gate
Accuracy	94.088%
Balanced Accuracy	79.339%
Macro F1	86.220%
Weighted F1	94.042%
Mean FAST path	~3.97%
Mean DEEP path	~96.03%
---
9. Phase 15 — Privacy-Aware Alert Sharing
Implemented concepts:
HMAC pseudonymization
Allowlisted metadata
Privacy-safe alert schema
TLS / mTLS-oriented design
Persistence
Retention controls
Performance / privacy audit
Raw flow, packet, payload, and sensitive identifiers remain local.
---
10. Phase 16 — Cross-Cloud Correlation
Implemented:
Rule-based correlation
Temporal correlation
GNN / GCN correlation
Rule vs GNN vs Hybrid comparison
---
11. Phase 17 — Byzantine / Poisoning Robustness
Threats:
Random poisoning
Scale poisoning
Label flipping
Different attacker proportions
Robust aggregators:
Coordinate Median
Trimmed Mean
Experiment design:
```text
10 seeds
5 experiment conditions
3 aggregators
=
150 matched runs
```
Macro F1
Threat	FedAvg	Median	Trimmed Mean
Clean	0.7035	0.7045	0.7484
Random ×5, 20%	0.4888	0.6117	0.6506
Random ×5, 40%	0.3118	0.4352	0.4395
Scale ×5, 20%	0.4550	0.6800	0.7212
Scale ×5, 40%	0.4510	0.7227	0.7541
Conclusion:
> Robust aggregation substantially mitigated poisoning under the evaluated controlled threat model.
---
12. Phase 17H — Dashboard Integration
Current dashboard:
```text
ids_dashboard_connected_v6_npz_sequence_test.py
```
Run:
```cmd
python ids_dashboard_connected_v6_npz_sequence_test.py
```
Open:
```text
http://127.0.0.1:9999
```
Features:
Natural / random validation replay
Class-balanced validation stress test
External CSV inference
Direct NPZ sequence inference
Independent model status
FAST / DEEP statistics
Predicted class distribution
Attack timeline
Client attack load
Live replay
Per-class metrics
Confusion matrix
Missed-attack diagnostics
---
13. Phase 17H.1 — Sampling / Client-Assignment Comparison
A. Balanced Six-Class + Round Robin
```text
Accuracy          = 79.20%
Balanced Accuracy = 79.14%
Macro F1          = 80.99%
FAST path         = 16.5%
```
B. Natural Validation + Round Robin
```text
Accuracy          = 93.90%
Balanced Accuracy = 80.52%
Macro F1          = 85.35%
FAST path         = 3.9%
```
C. Natural + Phase14H-Style Replay
```text
Accuracy          = 93.68%
Balanced Accuracy = 78.92%
Macro F1          = 84.41%
FAST path         = 4.3%
```
Conclusion:
> 79.2% is a balanced stress-test result, not normal validation accuracy.
---
14. Phase 17H.2 — Five-Attack Sequence Test Dataset
Created:
```text
data/dashboard_samples/ids_5_attacks_1000_sequences.npz
```
Shape:
```text
(1000, 20, 70)
```
Distribution:
```text
DoS Hulk         200
DDoS             200
PortScan         200
DoS GoldenEye    200
FTP-Patator      200
BENIGN             0
```
Properties:
Validation-only
No training rows
No locked-test rows
Sequence internals preserved
No internal sequence shuffling
Model inputs:
```text
RF          -> X[:, -1, :]
AE          -> X[:, -1, :]
FedProx70   -> X[:, -1, :]
CNN-BiLSTM -> X[:, :, :]
```
---
15. Phase 17H.3 — NPZ Sequence Test
Result:
```text
Accuracy              = 74.20%
Balanced Accuracy     = 74.20%
Macro Precision       = 96.82%
Macro Recall          = 74.20%
Macro F1              = 83.48%

Sequences             = 1000
Attack alerts         = 779
Predicted BENIGN      = 221
ANOMALOUS             = 11
```
Independent model indicators:
```text
Random Forest accuracy = 20.00%
FedProx70 accuracy     = 57.00%
CNN-BiLSTM accuracy    = 63.40%
AE anomaly rate        = 22.30%
```
---
16. Phase 17H.4 — False-Negative Analysis
Overall:
```text
Accuracy                 = 74.20%
Balanced Accuracy        = 74.20%
Macro F1                 = 83.4847%

Total errors             = 258
Missed attacks -> BENIGN = 221
Wrong attack family      = 26
Predicted ANOMALOUS      = 11
```
False Negatives by Attack
True Class	Total	Predicted BENIGN	Miss Rate
DDoS	200	1	0.5%
DoS GoldenEye	200	30	15.0%
DoS Hulk	200	60	30.0%
FTP-Patator	200	63	31.5%
PortScan	200	67	33.5%
Model Behavior on the 221 BENIGN Misses
```text
RF predicted BENIGN             = 221 / 221
FedProx70 predicted BENIGN      = 221 / 221
CNN-BiLSTM predicted BENIGN     = 221 / 221
FedProx70 + CNN both BENIGN     = 221 / 221
RF + FedProx70 + CNN all BENIGN = 221 / 221
AE flagged anomaly              =   0 / 221
```
Failure pattern:
```text
Foreign-client attack
        |
        v
Local RF -> BENIGN
        |
        v
FedProx70 -> BENIGN
        |
        v
CNN-BiLSTM -> BENIGN
        |
        v
AE does not flag anomaly
        |
        v
Final prediction -> BENIGN
```
---
17. Routing Diagnostic
`specialist_client_id` is analysis-only and was not used to route primary predictions.
Matching Specialist
```text
Sequences      = 200
Correct        = 200
Accuracy       = 100%
BENIGN misses  = 0
```
Foreign Client
```text
Sequences      = 800
Correct        = 542
Accuracy       = 67.75%
BENIGN misses  = 221
Miss rate      = 27.625%
```
Conclusion:
> Cross-client foreign-attack exposure under severe non-IID conditions is the dominant current failure mode.
The specialist result is an oracle diagnostic and must not be reported as deployable routing accuracy.
---
18. Current Scientific Interpretation
The frozen IDS performs strongly:
On natural validation
On matching specialist-client attacks
On DDoS under foreign-client exposure
The main failure mode is:
```text
Attack
  |
  v
All supervised models agree on BENIGN
  |
  v
AE also fails to flag anomaly
  |
  v
Silent false negative
```
Simple disagreement or entropy alone may therefore be insufficient.
---
19. Planned Phase 18 — Uncertainty-Aware Open-World IDS
Planned signals:
Prediction entropy
Model disagreement
Known-class confidence margin
Prototype distance
Class-centroid distance
Embedding distance
AE anomaly evidence
Conformal rejection
Selective classification
Planned states:
```text
KNOWN
AMBIGUOUS
UNKNOWN / SUSPICIOUS
```
Recommended evaluation:
Strict leave-one-attack-out
No unknown-class leakage
Validation-driven development
Final locked test only after Phase 18 is frozen
---
20. Important Current Scripts
```text
ids_dashboard_connected_v6_npz_sequence_test.py

scripts/phase17h1_sampling_client_assignment_comparison.py
scripts/phase17h2_create_5attack_sequence_test.py
scripts/phase17h4_false_negative_analysis.py

scripts/phase14g1_real_model_wiring.py
scripts/phase14h_full_validation_adaptive_gate.py
```
---
21. Important Data
Validation Sequences
```text
data/sequences/validation_sequences.npz
```
Expected shape:
```text
(N, 20, 70)
```
Five-Attack Stress Test
```text
data/dashboard_samples/ids_5_attacks_1000_sequences.npz
```
Shape:
```text
(1000, 20, 70)
```
---
22. Important Results
Phase 17H.1
```text
results/dashboard/phase17h1_comparison/
```
Phase 17H.4
```text
results/dashboard/phase17h4_false_negative_analysis/
```
Files:
```text
phase17h4_all_predictions.csv
phase17h4_missed_attacks.csv
phase17h4_missed_by_class.csv
phase17h4_missed_by_client.csv
phase17h4_model_behavior_on_misses.csv
phase17h4_routing_analysis.csv
phase17h4_confusion_matrix.csv
phase17h4_summary.json
```
---
23. Running the Dashboard
Activate environment:
```cmd
.venv\Scripts\activate
```
Run:
```cmd
python ids_dashboard_connected_v6_npz_sequence_test.py
```
Open:
```text
http://127.0.0.1:9999
```
---
24. Dashboard Input Modes
Natural / Random Validation
Representative validation mode preserving the natural class distribution.
Class-Balanced Stress Test
Harder balanced evaluation.
Do not report it as normal overall accuracy.
External CSV
CSV flow rows are:
Passed through frozen preprocessing
Converted to 70 active features
Rebuilt into row-ordered rolling 20-flow sequences
Passed to the frozen IDS
External NPZ
Preferred for sequence-valid evaluation.
Expected:
```text
X -> (N, 20, 70)
y -> optional ground truth
```
No sequence reconstruction is performed.
---
25. Scientific Reporting Rules
Use:
Validation accuracy
Balanced stress-test accuracy
Pooled client-exposure metrics
Source-ordered sequence
Controlled severe non-IID
Oracle diagnostic
Locked test not used
Robust aggregation substantially mitigated the evaluated controlled poisoning threat model
Avoid:
Final real-world accuracy before independent confirmation
Timestamp-confirmed temporal sequence unless timestamps are actually available
“Byzantine attacks solved”
“100% deployment accuracy” from specialist routing
Treating client exposures as independent unique flows
Treating 79.2% balanced stress accuracy as normal validation accuracy
---
26. Current Status
```text
Phase 1–13     COMPLETE
Phase 14       COMPLETE / FROZEN
Phase 15       COMPLETE
Phase 16       COMPLETE
Phase 17       COMPLETE
Phase 17H.1    COMPLETE
Phase 17H.2    COMPLETE
Phase 17H.3    COMPLETE
Phase 17H.4    COMPLETE

Phase 18       NEXT
```
---
27. Current Conclusion
The project currently contains:
A working federated IDS pipeline
A frozen adaptive multi-model detection stack
Severe non-IID experimentation
Robust aggregation under controlled poisoning
Privacy-aware alert sharing
Cross-cloud correlation
GNN-based correlation
Dashboard integration
Sequence-preserving NPZ evaluation
Detailed false-negative diagnostics
The main remaining weakness is foreign-client attacks collapsing to BENIGN, especially:
```text
PortScan
FTP-Patator
DoS Hulk
```
Phase 18 will target this using uncertainty, distance-based evidence, and rejection mechanisms.
---
Research Note
This repository is a research / academic prototype and should not be treated as a production network-security appliance without additional:
External validation
Real flow-collector validation
Security hardening
Operational latency testing
Deployment-specific calibration
Independent confirmatory testing
