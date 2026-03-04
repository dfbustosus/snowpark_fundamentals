"""Tests for Feature Store entity management."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from snowflake.ml.feature_store import CreationMode

from snowpark_fundamentals.feature_store.entities import (
    create_customer_entity,
    delete_entity,
    list_entities,
    register_entity,
    setup_feature_store,
)


class TestSetupFeatureStore:
    """Tests for setup_feature_store()."""

    @patch("snowpark_fundamentals.feature_store.entities.FeatureStore")
    def test_setup_with_defaults(self, mock_fs_class, mock_session):
        """Should use session context and CREATE_IF_NOT_EXIST by default."""
        mock_session.get_current_database.return_value = '"TEST_DB"'
        mock_session.get_current_schema.return_value = '"TEST_SCHEMA"'
        mock_session.get_current_warehouse.return_value = '"TEST_WH"'

        setup_feature_store(mock_session)

        mock_fs_class.assert_called_once_with(
            session=mock_session,
            database="TEST_DB",
            name="TEST_SCHEMA",
            default_warehouse="TEST_WH",
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )

    @patch("snowpark_fundamentals.feature_store.entities.FeatureStore")
    def test_setup_with_explicit_params(self, mock_fs_class, mock_session):
        """Should use explicit params when provided."""
        setup_feature_store(
            mock_session,
            database="MY_DB",
            schema="MY_SCHEMA",
            default_warehouse="MY_WH",
        )

        mock_fs_class.assert_called_once_with(
            session=mock_session,
            database="MY_DB",
            name="MY_SCHEMA",
            default_warehouse="MY_WH",
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )


class TestCreateCustomerEntity:
    """Tests for create_customer_entity()."""

    @patch("snowpark_fundamentals.feature_store.entities.Entity")
    def test_default_entity(self, mock_entity_class):
        """Should create entity with default name and CUSTOMER_ID key."""
        create_customer_entity()
        mock_entity_class.assert_called_once_with(
            name="CUSTOMER", join_keys=["CUSTOMER_ID"], desc=""
        )

    @patch("snowpark_fundamentals.feature_store.entities.Entity")
    def test_custom_name(self, mock_entity_class):
        """Should accept custom entity name."""
        create_customer_entity(name="MY_CUSTOMER")
        mock_entity_class.assert_called_once_with(
            name="MY_CUSTOMER", join_keys=["CUSTOMER_ID"], desc=""
        )

    @patch("snowpark_fundamentals.feature_store.entities.Entity")
    def test_with_description(self, mock_entity_class):
        """Should pass desc to Entity constructor."""
        create_customer_entity(desc="Customer entity for churn")
        mock_entity_class.assert_called_once_with(
            name="CUSTOMER",
            join_keys=["CUSTOMER_ID"],
            desc="Customer entity for churn",
        )


class TestRegisterEntity:
    """Tests for register_entity()."""

    def test_register_calls_fs(self, mock_feature_store, mock_entity):
        """Should call fs.register_entity with the entity."""
        register_entity(mock_feature_store, mock_entity)
        mock_feature_store.register_entity.assert_called_once_with(mock_entity)

    def test_register_returns_result(self, mock_feature_store, mock_entity):
        """Should return the registered entity reference."""
        expected = MagicMock()
        mock_feature_store.register_entity.return_value = expected

        result = register_entity(mock_feature_store, mock_entity)
        assert result == expected


class TestListEntities:
    """Tests for list_entities()."""

    def test_list_entities(self, mock_feature_store):
        """Should call fs.list_entities."""
        list_entities(mock_feature_store)
        mock_feature_store.list_entities.assert_called_once()


class TestDeleteEntity:
    """Tests for delete_entity()."""

    def test_delete_entity(self, mock_feature_store):
        """Should call fs.delete_entity with the name."""
        delete_entity(mock_feature_store, "CUSTOMER")
        mock_feature_store.delete_entity.assert_called_once_with("CUSTOMER")
