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


def save_interaction(driver, user_message: str, ai_message: str):
    """Log an interaction to Neo4j."""
    query = (
        "CREATE (:Interaction {user_message: $user_message, ai_message: $ai_message})"
    )
    with driver.session() as session:
        session.run(query, user_message=user_message, ai_message=ai_message)
