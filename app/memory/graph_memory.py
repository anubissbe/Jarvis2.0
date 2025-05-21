from neo4j import GraphDatabase

from ..config import settings


def get_driver():
    """Create a Neo4j driver using configuration settings."""
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def save_interaction(driver, user_message: str, response: str) -> None:
    """Persist a user interaction to Neo4j."""
    query = (
        "CREATE (:Interaction {user_message: $user, response: $resp, timestamp: "
        "datetime()})"
    )
    with driver.session() as session:
        session.run(query, user=user_message, resp=response)
