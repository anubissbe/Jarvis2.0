from neo4j import GraphDatabase

from ..config import settings


def get_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def save_interaction(driver, user_text: str, assistant_text: str) -> None:
    """Persist a conversation turn as nodes linked by a relationship."""

    def _write(tx, u, a):
        tx.run(
            "MERGE (usr:Message {text: $u, role: 'user'}) "
            "MERGE (bot:Message {text: $a, role: 'assistant'}) "
            "MERGE (usr)-[:REPLIED_WITH]->(bot)",
            u=u,
            a=a,
        )

    with driver.session() as session:
        session.execute_write(_write, user_text, assistant_text)
