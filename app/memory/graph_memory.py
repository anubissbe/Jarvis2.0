try:
    from neo4j import AsyncGraphDatabase
except ImportError:  # pragma: no cover - neo4j not installed in testing env
    AsyncGraphDatabase = None

from ..config import settings


def get_driver():
    """Return a Neo4j async driver if available."""
    if AsyncGraphDatabase is None:
        return None
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


async def save_interaction(driver, user_message: str, ai_message: str) -> None:
    """Persist a user/assistant message pair to Neo4j."""
    if driver is None:
        return
    async with driver.session() as session:
        await session.run(
            "MERGE (u:User {id: 1}) "
            "CREATE (u)-[:SENT]->(:Message {text: $user_msg}) "
            "CREATE (u)-[:RECEIVED]->(:Message {text: $ai_msg})",
            user_msg=user_message,
            ai_msg=ai_message,
        )
