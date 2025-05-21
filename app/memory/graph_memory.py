from neo4j import GraphDatabase

from ..config import settings


def get_driver():
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

