from datetime import datetime

from src.detection.risk_fusion import RiskFusion


# ============================================================
# ADAPTIVE CLIENT-SIDE IDS ENGINE
# ============================================================

class DetectionEngine:

    def __init__(self):

        # ----------------------------------------------------
        # HARDCODED THRESHOLDS
        # ----------------------------------------------------

        self.rf_fast_attack_threshold = 0.95

        self.autoencoder_anomaly_threshold = 0.70

        self.deep_analysis_threshold = 0.90


        # ----------------------------------------------------
        # RISK FUSION
        # ----------------------------------------------------

        self.risk_fusion = RiskFusion()


    # ========================================================
    # ADAPTIVE GATE
    # ========================================================

    def adaptive_gate(
        self,
        rf_class: str,
        rf_confidence: float,
    ):

        # ----------------------------------------------------
        # CASE 1:
        # Extremely confident RF ATTACK.
        #
        # Even here we keep the architecture capable of
        # calling deeper models when required.
        # ----------------------------------------------------

        if (
            rf_class.upper() != "BENIGN"
            and
            rf_confidence >= self.rf_fast_attack_threshold
        ):

            return {

                "path":
                    "FAST_ATTACK_PATH",

                "run_autoencoder":
                    False,

                "run_fedprox70":
                    False,

                "run_cnn_bilstm":
                    False,
            }


        # ----------------------------------------------------
        # CASE 2:
        # EVERYTHING ELSE GOES THROUGH DEEP ANALYSIS
        #
        # This includes:
        #   BENIGN
        #   uncertain attack
        #   moderate-confidence predictions
        # ----------------------------------------------------

        return {

            "path":
                "DEEP_ANALYSIS",

            "run_autoencoder":
                True,

            "run_fedprox70":
                True,

            "run_cnn_bilstm":
                True,
        }


    # ========================================================
    # BUILD LOCAL IDS ALERT
    # ========================================================

    def build_alert(
        self,

        client_id: int,

        final_result,

        rf_result,

        fedprox_result,

        cnn_bilstm_result,

        anomaly_score,

        processing_path,
    ):

        return {

            "timestamp":
                datetime.now().isoformat(),

            "client_id":
                client_id,


            # =================================================
            # FINAL DECISION
            # =================================================

            "classification":
                final_result.final_class,

            "severity":
                final_result.severity,

            "risk_score":
                final_result.risk_score,

            "confidence":
                final_result.confidence,

            "decision_source":
                final_result.decision_source,

            "processing_path":
                processing_path,


            # =================================================
            # INDIVIDUAL MODEL OUTPUTS
            # =================================================

            "models": {


                # ---------------------------------------------
                # RANDOM FOREST
                # ---------------------------------------------

                "random_forest": {

                    "prediction":
                        rf_result[
                            "prediction"
                        ],

                    "confidence":
                        rf_result[
                            "confidence"
                        ],
                },


                # ---------------------------------------------
                # FEDPROX70 MLP
                # ---------------------------------------------

                "fedprox70_mlp": {

                    "prediction":
                        fedprox_result[
                            "prediction"
                        ],

                    "confidence":
                        fedprox_result[
                            "confidence"
                        ],
                },


                # ---------------------------------------------
                # FEDERATED CNN-BILSTM
                # ---------------------------------------------

                "cnn_bilstm": {

                    "prediction":
                        cnn_bilstm_result[
                            "prediction"
                        ],

                    "confidence":
                        cnn_bilstm_result[
                            "confidence"
                        ],
                },


                # ---------------------------------------------
                # PERSONALIZED AUTOENCODER
                # ---------------------------------------------

                "personalized_autoencoder": {

                    "anomaly_score":
                        anomaly_score,

                    "is_anomaly":
                        bool(
                            anomaly_score
                            >=
                            self.autoencoder_anomaly_threshold
                        ),
                },
            },
        }