#database.py
import sqlite3

connection = sqlite3.connect('LoginData.db')
cursor = connection.cursor()

cmd = """DROP TABLE IF EXISTS USERS"""
cursor.execute(cmd)

cmd1 = """CREATE TABLE IF NOT EXISTS USERS(first_name varchar(50),
                                          last_name varchar(50),
                                          email varchar(50) primary key,
                                          password_hash varchar(50) not null)"""

cursor.execute(cmd1)

cmd2 = """INSERT INTO USERS(first_name, last_name, email, password_hash) values
                ('tester','tester','tester@gmail.com','tester')"""
#cursor.execute(cmd2)

connection.commit()

ans = cursor.execute("select * from USERS").fetchall()

for i in ans:
    print(i)

cmd = """CREATE TABLE IF NOT EXISTS USEROTP(email varchar(50) primary key references USERS(email), otp varchar(6))"""
cursor.execute(cmd)
connection.commit()

cmd3 = """
CREATE TABLE IF NOT EXISTS FALL_EVENTS(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email varchar(50) references USERS(email),
    timestamp TEXT NOT NULL,
    location TEXT,
    image_path TEXT
)
"""

try:
    cursor.execute("ALTER TABLE FALL_EVENTS ADD COLUMN status TEXT DEFAULT 'new'")
    connection.commit()
except sqlite3.OperationalError:
    pass

try:
    cursor.execute("ALTER TABLE FALL_EVENTS ADD COLUMN false_positive INTEGER DEFAULT 0")
    connection.commit()
except sqlite3.OperationalError:
    pass

# If table already existed before, ensure image_path column exists
try:
    cursor.execute("ALTER TABLE FALL_EVENTS ADD COLUMN image_path TEXT")
    connection.commit()
except sqlite3.OperationalError:
    # column already exists (or table missing) -> ignore
    pass

cursor.execute(cmd3)
connection.commit()

connection.close()