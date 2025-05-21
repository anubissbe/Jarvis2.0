try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - neo4j not installed in testing env
    GraphDatabase = None

from ..config import settings


def get_driver():
    """Return a Neo4j driver if available."""
    if GraphDatabase is None:
        return None
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def save_interaction(driver, user_message: str, assistant_message: str) -> None:
    """Persist the interaction if a driver is available."""
    if driver is None:
        return
    with driver.session() as session:  # pragma: no cover - requires DB
        session.execute_write(
            lambda tx: tx.run(
                "CREATE (:Message {user: $u, assistant: $a})",
                u=user_message,
                a=assistant_message,
            )
        )



