"""Feature Store entity management.

Demonstrates how to create and register Entities in the
Snowflake Feature Store — the key building blocks that define
the grain (e.g., CUSTOMER_ID) for feature lookups.
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store import CreationMode, Entity, FeatureStore
from snowflake.snowpark import Session


def setup_feature_store(
    session: Session,
    database: str | None = None,
    schema: str | None = None,
    default_warehouse: str | None = None,
    creation_mode: CreationMode = CreationMode.CREATE_IF_NOT_EXIST,
) -> FeatureStore:
    """Initialize a Snowflake Feature Store instance.

    Creates the FeatureStore object that serves as the entry point
    for all entity and feature view operations.

    Args:
        session: Active Snowpark session.
        database: Database name. Defaults to session's current database.
        schema: Schema name. Defaults to session's current schema.
        default_warehouse: Warehouse for feature materialization.
            Defaults to session's current warehouse.
        creation_mode: Schema creation behavior. Defaults to
            CREATE_IF_NOT_EXIST (creates schema and tags if needed).

    Returns:
        Initialized FeatureStore instance.
    """
    db = database or (session.get_current_database() or "").replace('"', "")
    sch = schema or (session.get_current_schema() or "").replace('"', "")
    wh = default_warehouse or (session.get_current_warehouse() or "").replace('"', "")

    fs: FeatureStore = FeatureStore(
        session=session,
        database=db,
        name=sch,
        default_warehouse=wh,
        creation_mode=creation_mode,
    )
    return fs


def create_customer_entity(name: str = "CUSTOMER", desc: str = "") -> Entity:
    """Create a Customer entity for feature lookups.

    An Entity defines the join key(s) for a set of features.
    This creates a simple single-key entity on CUSTOMER_ID.

    Args:
        name: Entity name in the Feature Store.
        desc: Human-readable description.

    Returns:
        Entity object (not yet registered).
    """
    entity: Entity = Entity(name=name, join_keys=["CUSTOMER_ID"], desc=desc)
    return entity


def register_entity(fs: FeatureStore, entity: Entity) -> Entity:
    """Register an entity in the Feature Store.

    If the entity already exists, it will be retrieved instead
    of raising an error.

    Args:
        fs: Initialized FeatureStore instance.
        entity: Entity object to register.

    Returns:
        Registered Entity reference.
    """
    registered: Entity = fs.register_entity(entity)
    return registered


def list_entities(fs: FeatureStore) -> Any:
    """List all entities registered in the Feature Store.

    Args:
        fs: Initialized FeatureStore instance.

    Returns:
        DataFrame with entity metadata.
    """
    return fs.list_entities()


def delete_entity(fs: FeatureStore, name: str) -> None:
    """Delete an entity from the Feature Store.

    The entity must not be referenced by any feature view.

    Args:
        fs: Initialized FeatureStore instance.
        name: Entity name to delete.
    """
    fs.delete_entity(name)
