from neo4j import GraphDatabase

from ..config import settings


def get_driver():
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

