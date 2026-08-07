import psycopg2
import psycopg2.extras
import psycopg2.pool
from contextlib import contextmanager

from pgvector.psycopg2 import register_vector


class ConnManager:
    def __init__(self, host: str, port: int, dbname: str, user: str, password: str, min_conn: int = 1, max_conn: int = 16):
        self._pool_conn = psycopg2.pool.ThreadedConnectionPool(min_conn, max_conn, host=host, port=port, dbname=dbname, user=user, password=password)

    def close(self):
        self._pool_conn.closeall()

    @contextmanager
    def get_cursor(self, commit: bool = False):
        conn = self._pool_conn.getconn()

        try:
            register_vector(conn)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            try:
                yield cursor
                if commit:
                    conn.commit()

            except Exception:
                conn.rollback()
                raise

            finally:
                cursor.close()

        finally:
            self._pool_conn.putconn(conn)