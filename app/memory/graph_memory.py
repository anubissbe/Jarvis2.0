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


def save_interaction(driver: GraphDatabase.driver, user_message: str, assistant_message: str) -> None:
    """Persist a conversation turn in Neo4j."""
    query = (
        "MERGE (c:Conversation {id: 1}) "
        "CREATE (u:Message {text: $user, role: 'user'})-[:PART_OF]->(c) "
        "CREATE (a:Message {text: $assistant, role: 'assistant'})-[:PART_OF]->(c)"
    )
    with driver.session() as session:
        session.execute_write(lambda tx: tx.run(query, user=user_message, assistant=assistant_message))
