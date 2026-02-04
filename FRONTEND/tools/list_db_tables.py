import sqlite3
import os

db = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'db.sqlite3')
if not os.path.exists(db):
    print('DB file not found:', db)
    raise SystemExit(1)

conn = sqlite3.connect(db)
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
print('Tables in', db)
for t in tables:
    print('-', t[0])
conn.close()
