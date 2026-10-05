from dataclasses import dataclass


# ============================================================
# FINAL RISK RESULT
# ============================================================

@dataclass
class RiskResult:

    risk_score: float
    severity: str
    final_class: str
    confidence: float
    decision_source: str


# ============================================================
# RISK FUSION ENGINE
# ============================================================

class RiskFusion:

    def __init__(self):

        # ----------------------------------------------------
        # HARDCODED INITIAL WEIGHTS
        #
        # These are engineering defaults.
        # Later we can tune them using VALIDATION data only.
        # ----------------------------------------------------

        self.rf_weight = 0.20

        self.fedprox_weight = 0.30

        self.cnn_bilstm_weight = 0.35

        self.autoencoder_weight = 0.15


    # ========================================================
    # CONVERT MODEL OUTPUT INTO ATTACK RISK
    # ========================================================

    def classification_risk(
        self,
        predicted_class: str,
        confidence: float,
    ) -> float:

        if predicted_class.upper() == "BENIGN":

            return 1.0 - confidence

        return confidence


    # ========================================================
    # SEVERITY
    # ========================================================

    def get_severity(
        self,
        risk_score: float,
    ) -> str:

        if risk_score < 0.20:
            return "NORMAL"

        elif risk_score < 0.40:
            return "LOW"

        elif risk_score < 0.60:
            return "MEDIUM"

        elif risk_score < 0.80:
            return "HIGH"

        return "CRITICAL"


    # ========================================================
    # FINAL FUSION
    # ========================================================

    def fuse(
        self,

        rf_class: str,
        rf_confidence: float,

        fedprox_class: str,
        fedprox_confidence: float,

        cnn_bilstm_class: str,
        cnn_bilstm_confidence: float,

        anomaly_score: float,
    ) -> RiskResult:

        # ----------------------------------------------------
        # RANDOM FOREST RISK
        # ----------------------------------------------------

        rf_risk = self.classification_risk(
            rf_class,
            rf_confidence,
        )


        # ----------------------------------------------------
        # FEDPROX70 MLP RISK
        # ----------------------------------------------------

        fedprox_risk = self.classification_risk(
            fedprox_class,
            fedprox_confidence,
        )


        # ----------------------------------------------------
        # CNN-BILSTM RISK
        # ----------------------------------------------------

        cnn_bilstm_risk = self.classification_risk(
            cnn_bilstm_class,
            cnn_bilstm_confidence,
        )


        # ----------------------------------------------------
        # AUTOENCODER
        #
        # anomaly_score expected in range:
        # 0.0 -> normal
        # 1.0 -> highly anomalous
        # ----------------------------------------------------

        anomaly_risk = max(
            0.0,
            min(
                1.0,
                anomaly_score,
            ),
        )


        # ----------------------------------------------------
        # WEIGHTED RISK FUSION
        # ----------------------------------------------------

        risk_score = (

            self.rf_weight
            * rf_risk

            +

            self.fedprox_weight
            * fedprox_risk

            +

            self.cnn_bilstm_weight
            * cnn_bilstm_risk

            +

            self.autoencoder_weight
            * anomaly_risk
        )


        risk_score = max(
            0.0,
            min(
                1.0,
                risk_score,
            ),
        )


        # ----------------------------------------------------
        # COLLECT NON-BENIGN CLASSIFIER PREDICTIONS
        # ----------------------------------------------------

        attack_candidates = []


        if rf_class.upper() != "BENIGN":

            attack_candidates.append(
                (
                    rf_confidence,
                    rf_class,
                    "RandomForest",
                )
            )


        if fedprox_class.upper() != "BENIGN":

            attack_candidates.append(
                (
                    fedprox_confidence,
                    fedprox_class,
                    "FedProx70",
                )
            )


        if cnn_bilstm_class.upper() != "BENIGN":

            attack_candidates.append(
                (
                    cnn_bilstm_confidence,
                    cnn_bilstm_class,
                    "CNN-BiLSTM",
                )
            )


        # ----------------------------------------------------
        # FINAL CLASS
        # ----------------------------------------------------

        if attack_candidates:

            best_prediction = max(
                attack_candidates,
                key=lambda item: item[0],
            )

            final_confidence = best_prediction[0]

            final_class = best_prediction[1]

            decision_source = best_prediction[2]


        elif anomaly_score >= 0.70:

            final_class = "ANOMALOUS_TRAFFIC"

            final_confidence = anomaly_score

            decision_source = "PersonalizedAutoencoder"


        else:

            final_class = "BENIGN"

            final_confidence = max(
                rf_confidence,
                fedprox_confidence,
                cnn_bilstm_confidence,
            )

            decision_source = "ClassifierConsensus"


        # ----------------------------------------------------
        # SEVERITY
        # ----------------------------------------------------

        severity = self.get_severity(
            risk_score
        )


        return RiskResult(

            risk_score=round(
                risk_score,
                6,
            ),

            severity=severity,

            final_class=final_class,

            confidence=round(
                final_confidence,
                6,
            ),

            decision_source=decision_source,
        )