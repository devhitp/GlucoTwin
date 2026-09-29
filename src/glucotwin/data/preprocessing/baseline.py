import pandas as pd


class BaselineFeatures:
    @staticmethod
    def generate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate causal patient-specific baseline features.

        Uses a long backward-looking rolling window (24 h) to capture the
        patient's running glucose baseline WITHOUT leaking future information.
        All computations are strictly backward-looking.
        """
        if df.empty:
            return df

        features = df.copy()

        if 'glucose' in features.columns:
            # 24-hour rolling mean and std of glucose (causal, min_periods=1)
            features['personal_glucose_baseline_24h'] = (
                features['glucose'].rolling('24h', min_periods=1).mean()
            )
            features['personal_glucose_std_24h'] = (
                features['glucose'].rolling('24h', min_periods=1).std()
            )
            features['glucose_deviation_from_baseline'] = (
                features['glucose'] - features['personal_glucose_baseline_24h']
            )

            # Sprint-7 canonical aliases used by feature_matrix.py
            features['patient_glucose_baseline'] = features['personal_glucose_baseline_24h']
            features['patient_glucose_std']      = features['personal_glucose_std_24h'].fillna(0.0)

        if 'heart_rate' in features.columns:
            features['personal_hr_baseline_24h'] = (
                features['heart_rate'].rolling('24h', min_periods=1).mean()
            )

        return features
