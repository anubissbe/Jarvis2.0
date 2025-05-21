from neo4j import GraphDatabase

from ..config import settings


def get_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def save_interaction(driver, user_input: str, response: str) -> None:
    def run(tx):
        tx.run(
            "CREATE (:Interaction {user_input: $u, response: $r, timestamp: timestamp()})",
            u=user_input,
            r=response,
        )

    with driver.session() as session:
        session.execute_write(run)
