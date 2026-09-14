import sqlite3
import os


# Database file location
DATABASE_PATH = "database/detections.db"


def create_database():
    # Create database folder if it doesn't exist
    os.makedirs("database", exist_ok=True)

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            target_class TEXT,
            confidence REAL,
            x1 REAL,
            y1 REAL,
            x2 REAL,
            y2 REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_detection(
    image_name,
    target_class,
    confidence,
    x1,
    y1,
    x2,
    y2
):
    os.makedirs("database", exist_ok=True)

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO detections
        (
            image_name,
            target_class,
            confidence,
            x1,
            y1,
            x2,
            y2
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        image_name,
        target_class,
        confidence,
        x1,
        y1,
        x2,
        y2
    ))

    conn.commit()
    conn.close()


# Create database when this file is run directly
if __name__ == "__main__":
    create_database()
    print("Database created successfully!")