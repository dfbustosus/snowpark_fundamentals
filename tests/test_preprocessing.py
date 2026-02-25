"""Tests for preprocessing module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestScaleNumericFeatures:
    @patch("snowpark_fundamentals.preprocessing.transformers.StandardScaler")
    def test_standard_scaler_default_output_cols(self, mock_scaler_cls, mock_dataframe):
        mock_scaler = MagicMock()
        mock_scaler.fit.return_value = mock_scaler
        mock_scaler.transform.return_value = mock_dataframe
        mock_scaler_cls.return_value = mock_scaler

        from snowpark_fundamentals.preprocessing.transformers import scale_numeric_features

        df_result, scaler = scale_numeric_features(mock_dataframe, ["AGE", "SALARY"])

        mock_scaler_cls.assert_called_once_with(
            input_cols=["AGE", "SALARY"],
            output_cols=["AGE_SCALED", "SALARY_SCALED"],
        )
        mock_scaler.fit.assert_called_once_with(mock_dataframe)
        assert df_result is not None

    @patch("snowpark_fundamentals.preprocessing.transformers.MinMaxScaler")
    def test_minmax_scaler(self, mock_scaler_cls, mock_dataframe):
        mock_scaler = MagicMock()
        mock_scaler.fit.return_value = mock_scaler
        mock_scaler.transform.return_value = mock_dataframe
        mock_scaler_cls.return_value = mock_scaler

        from snowpark_fundamentals.preprocessing.transformers import scale_numeric_features

        scale_numeric_features(mock_dataframe, ["AGE"], method="minmax")
        mock_scaler_cls.assert_called_once()


class TestEncodeCategoricalFeatures:
    @patch("snowpark_fundamentals.preprocessing.transformers.OneHotEncoder")
    def test_onehot_encoder(self, mock_encoder_cls, mock_dataframe):
        mock_encoder = MagicMock()
        mock_encoder.fit.return_value = mock_encoder
        mock_encoder.transform.return_value = mock_dataframe
        mock_encoder_cls.return_value = mock_encoder

        from snowpark_fundamentals.preprocessing.transformers import encode_categorical_features

        encode_categorical_features(mock_dataframe, ["CITY"], method="onehot")
        mock_encoder_cls.assert_called_once_with(
            input_cols=["CITY"],
            output_cols=["CITY_ENCODED"],
        )

    @patch("snowpark_fundamentals.preprocessing.transformers.OrdinalEncoder")
    def test_ordinal_encoder(self, mock_encoder_cls, mock_dataframe):
        mock_encoder = MagicMock()
        mock_encoder.fit.return_value = mock_encoder
        mock_encoder.transform.return_value = mock_dataframe
        mock_encoder_cls.return_value = mock_encoder

        from snowpark_fundamentals.preprocessing.transformers import encode_categorical_features

        encode_categorical_features(mock_dataframe, ["CITY"], method="ordinal")
        mock_encoder_cls.assert_called_once()


class TestBuildPreprocessingPipeline:
    @patch("snowpark_fundamentals.preprocessing.transformers.encode_categorical_features")
    @patch("snowpark_fundamentals.preprocessing.transformers.scale_numeric_features")
    def test_pipeline_chains_transformers(self, mock_scale, mock_encode, mock_dataframe):
        mock_scale.return_value = (mock_dataframe, MagicMock())
        mock_encode.return_value = (mock_dataframe, MagicMock())

        from snowpark_fundamentals.preprocessing.transformers import build_preprocessing_pipeline

        df_result, transformers = build_preprocessing_pipeline(mock_dataframe, ["AGE"], ["CITY"])

        mock_scale.assert_called_once()
        mock_encode.assert_called_once()
        assert "scaler" in transformers
        assert "encoder" in transformers
