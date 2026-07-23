import psycopg2
from contextlib import contextmanager


class ConnManager:
    def __init__(self, host: str, port: int, dbname: str, user: str, password: str, min_conn: int = 1, max_conn: int = 16):
        self._pool_conn = psycopg2.pool.ThreadedConnectionPool(min_conn, max_conn, host=host, port=port, dbname=dbname, user=user, password=password)

    def close(self):
        self._pool_conn.closeall()

    @contextmanager
    def cursor(self, commit: bool = False):
        conn = self._pool_conn.getconn()

        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            yield cursor

            if commit:
                conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            cursor.close()
            self._pool_conn.putconn(conn)


    def get_cursor(self, commit: bool = False):
        return self.cursor(commit=False)