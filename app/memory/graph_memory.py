from neo4j import GraphDatabase

from ..config import settings


def get_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def save_interaction(driver, user_text: str, assistant_text: str) -> None:
    with driver.session() as session:
        session.run(
            "CREATE (u:UserMessage {text: $u})\n"
            "CREATE (a:AssistantMessage {text: $a})\n"
            "CREATE (u)-[:REPLIED_WITH]->(a)",
            {"u": user_text, "a": assistant_text},
        )
