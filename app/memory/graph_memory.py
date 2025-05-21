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


def save_interaction(driver, user_message: str, bot_response: str) -> None:
    """Persist a single interaction to Neo4j."""
    query = (
        "MERGE (u:User {id: 1}) "
        "CREATE (i:Interaction {user_message: $user_message, bot_response: $bot_response, timestamp: timestamp()}) "
        "MERGE (u)-[:MADE]->(i)"
    )
    with driver.session() as session:
        session.run(query, user_message=user_message, bot_response=bot_response)

