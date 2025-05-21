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


def save_interaction(driver, user_message: str, ai_message: str) -> None:
    """Persist a user/assistant message pair to Neo4j."""
    with driver.session() as session:
        session.run(
            "MERGE (u:User {id: 1}) "
            "CREATE (u)-[:SENT]->(:Message {text: $user_msg}) "
            "CREATE (u)-[:RECEIVED]->(:Message {text: $ai_msg})",
            user_msg=user_message,
            ai_msg=ai_message,
        )
