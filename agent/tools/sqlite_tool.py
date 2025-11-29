import sqlite3
from typing import List, Dict, Any

class SQLiteTool:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        
        cur = self.conn.cursor()
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = [dict(row) for row in cur.fetchall()]
        return rows , columns
