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


def save_interaction(driver, user_message: str, response: str) -> None:
    """Persist a conversation turn to Neo4j."""

    def _write(tx, user_msg: str, resp: str) -> None:
        tx.run(
            """
            MERGE (u:Message {text: $user_msg, role: 'user'})
            MERGE (a:Message {text: $resp, role: 'assistant'})
            MERGE (u)-[:ANSWERED_BY]->(a)
            """,
            user_msg=user_msg,
            resp=resp,
        )

    with driver.session() as session:
        session.execute_write(_write, user_message, response)

