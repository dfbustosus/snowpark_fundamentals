"""Tests for Feature View creation and registration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from snowpark_fundamentals.feature_store.feature_views import (
    create_external_feature_view,
    create_managed_feature_view,
    delete_feature_view,
    get_feature_view,
    list_feature_views,
    register_feature_view,
)


class TestCreateManagedFeatureView:
    """Tests for create_managed_feature_view()."""

    @patch("snowpark_fundamentals.feature_store.feature_views.FeatureView")
    def test_managed_view_defaults(self, mock_fv_class, mock_entity):
        """Should create a managed view with default refresh frequency."""
        mock_df = MagicMock()

        create_managed_feature_view(
            name="RFM_FEATURES",
            entities=[mock_entity],
            feature_df=mock_df,
        )

        mock_fv_class.assert_called_once_with(
            name="RFM_FEATURES",
            entities=[mock_entity],
            feature_df=mock_df,
            refresh_freq="1 day",
            desc="",
            timestamp_col=None,
        )

    @patch("snowpark_fundamentals.feature_store.feature_views.FeatureView")
    def test_managed_view_custom_refresh(self, mock_fv_class, mock_entity):
        """Should accept custom refresh frequency and desc."""
        mock_df = MagicMock()

        create_managed_feature_view(
            name="HOURLY_FEATURES",
            entities=[mock_entity],
            feature_df=mock_df,
            refresh_freq="1 hour",
            desc="Hourly refresh",
        )

        call_kwargs = mock_fv_class.call_args[1]
        assert call_kwargs["refresh_freq"] == "1 hour"
        assert call_kwargs["desc"] == "Hourly refresh"

    @patch("snowpark_fundamentals.feature_store.feature_views.FeatureView")
    def test_managed_view_with_timestamp(self, mock_fv_class, mock_entity):
        """Should pass timestamp_col for point-in-time lookups."""
        mock_df = MagicMock()

        create_managed_feature_view(
            name="RFM_FEATURES",
            entities=[mock_entity],
            feature_df=mock_df,
            timestamp_col="_FEATURE_TIMESTAMP",
        )

        call_kwargs = mock_fv_class.call_args[1]
        assert call_kwargs["timestamp_col"] == "_FEATURE_TIMESTAMP"


class TestCreateExternalFeatureView:
    """Tests for create_external_feature_view()."""

    @patch("snowpark_fundamentals.feature_store.feature_views.FeatureView")
    def test_external_view_no_refresh(self, mock_fv_class, mock_entity):
        """External views must have refresh_freq=None."""
        mock_df = MagicMock()

        create_external_feature_view(
            name="DBT_FEATURES",
            entities=[mock_entity],
            feature_df=mock_df,
        )

        call_kwargs = mock_fv_class.call_args[1]
        assert call_kwargs["refresh_freq"] is None

    @patch("snowpark_fundamentals.feature_store.feature_views.FeatureView")
    def test_external_view_with_desc(self, mock_fv_class, mock_entity):
        """Should pass desc through."""
        mock_df = MagicMock()

        create_external_feature_view(
            name="DBT_FEATURES",
            entities=[mock_entity],
            feature_df=mock_df,
            desc="Built by dbt",
        )

        call_kwargs = mock_fv_class.call_args[1]
        assert call_kwargs["desc"] == "Built by dbt"


class TestRegisterFeatureView:
    """Tests for register_feature_view()."""

    def test_register_default_version(self, mock_feature_store, mock_feature_view):
        """Should register with default version V1."""
        register_feature_view(mock_feature_store, mock_feature_view)

        mock_feature_store.register_feature_view.assert_called_once_with(
            feature_view=mock_feature_view,
            version="V1",
            overwrite=False,
        )

    def test_register_custom_version(self, mock_feature_store, mock_feature_view):
        """Should accept custom version."""
        register_feature_view(mock_feature_store, mock_feature_view, version="V2")

        call_kwargs = mock_feature_store.register_feature_view.call_args[1]
        assert call_kwargs["version"] == "V2"

    def test_register_with_overwrite(self, mock_feature_store, mock_feature_view):
        """Should pass overwrite flag."""
        register_feature_view(mock_feature_store, mock_feature_view, overwrite=True)

        call_kwargs = mock_feature_store.register_feature_view.call_args[1]
        assert call_kwargs["overwrite"] is True

    def test_register_returns_result(self, mock_feature_store, mock_feature_view):
        """Should return the registered feature view reference."""
        expected = MagicMock()
        mock_feature_store.register_feature_view.return_value = expected

        result = register_feature_view(mock_feature_store, mock_feature_view)
        assert result == expected


class TestGetFeatureView:
    """Tests for get_feature_view()."""

    def test_get_feature_view(self, mock_feature_store):
        """Should call fs.get_feature_view with name and version."""
        get_feature_view(mock_feature_store, "RFM_FV", "V1")
        mock_feature_store.get_feature_view.assert_called_once_with("RFM_FV", "V1")


class TestListFeatureViews:
    """Tests for list_feature_views()."""

    def test_list_feature_views(self, mock_feature_store):
        """Should call fs.list_feature_views."""
        list_feature_views(mock_feature_store)
        mock_feature_store.list_feature_views.assert_called_once()


class TestDeleteFeatureView:
    """Tests for delete_feature_view()."""

    def test_delete_feature_view(self, mock_feature_store):
        """Should call fs.delete_feature_view with name and version."""
        delete_feature_view(mock_feature_store, "RFM_FV", "V1")
        mock_feature_store.delete_feature_view.assert_called_once_with("RFM_FV", "V1")
