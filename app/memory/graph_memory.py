from neo4j import GraphDatabase

from ..config import settings


def get_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def save_interaction(driver, user_message: str, ai_response: str) -> None:
    """Persist the interaction in Neo4j for long-term memory."""
    query = (
        "MERGE (u:User {id: 'default'}) "
        "CREATE (q:Message {text: $q})-[:FROM]->(u) "
        "CREATE (a:Message {text: $a})-[:FROM]->(u) "
        "CREATE (q)-[:REPLIED_WITH]->(a)"
    )
    with driver.session() as session:
        session.run(query, q=user_message, a=ai_response)
