from backend.database.connect_to_database import ConnManager

import bcrypt
from typing import Optional
from psycopg2.extras import execute_values


class PostgresService:
    def __init__(self, host: str, port: int, dbname: str, user: str, password: str):
        self.connection_manager = ConnManager(host=host, port=port, dbname=dbname, user=user, password=password)


    def insert_chunk_embedding(self, chunk_id: str, source: str, chunk_text: str, embedding: list[float]):
        with self.connection_manager.get_cursor(commit=True) as cur:
            cur.execute(
                """
                    INSERT INTO chunk_embeddings (chunk_id, source, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO NOTHING
                """, (chunk_id, source, chunk_text, embedding)
            )


    def insert_chunk_embedding_batch(self, payload: list[tuple[str, str, str, list[float]]]):
        if not payload:
            return

        with self.connection_manager.get_cursor(commit=True) as cur:
            execute_values(
                cur=cur,
                sql="""
                    INSERT INTO chunk_embeddings (chunk_id, source, chunk_text, embedding)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO NOTHING
                """,
                argslist=payload
            )


    def search_similar_chunks(self, query_embedding: list[float], top_k: int = 10, source: Optional[str] = None):
        with self.connection_manager.get_cursor(commit=True) as cur:
            if source:
                cur.execute(
                    """
                        SELECT chunk_id, source, chunk_text, embedding <=> %s::vector AS distance
                        FROM chunk_embeddings
                        WHERE source = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, source, query_embedding, top_k)
                )
            else:
                cur.execute(
                    """
                        SELECT chunk_id, source, chunk_text, embedding <=> %s::vector AS distance
                        FROM chunk_embeddings
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, query_embedding, top_k)
                )

            return cur.fetchall()


    def insert_chat_message(self, session_id: str, role: str, content: str, embedding: Optional[list[float]] = None, user_id: Optional[str] = None) -> int:
        assert role in ['user', 'assistant']

        with self.connection_manager.get_cursor(commit=True) as cur:
            cur.execute(
                """
                    INSERT INTO chat_history (user_id, session_id, role, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (user_id, session_id, role, content, embedding)
            )

            return cur.fetchone()['id']


    def get_session_history(self, session_id: str, limit: int = 25):
        with self.connection_manager.get_cursor(commit=False) as cur:
            cur.execute(
                """
                    SELECT id, user_id, role, content, created_at
                    FROM chat_history
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                    LIMIT %s
                """, (session_id, limit)
            )

            return cur.fetchall()


    def search_similar_history(self, query_embedding: list[float], user_id: Optional[str] = None, top_k: int = 5):
        with self.connection_manager.get_cursor(commit=True) as cur:
            if user_id:
                cur.execute(
                    """
                        SELECT id, session_id, role, content, embedding <=> %s::vector AS distance
                        FROM chat_history
                        WHERE user_id = %s AND embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, user_id, query_embedding, top_k)
                )
            else:
                cur.execute(
                    """
                        SELECT id, session_id, role, content, embedding <=> %s::vector AS distance
                        FROM chat_history
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, query_embedding, top_k)
                )

            return cur.fetchall()


    # TODO create the user function for create, verify, get_user_by_id, and update_password