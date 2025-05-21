from neo4j import GraphDatabase

from ..config import settings


def get_driver():
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
