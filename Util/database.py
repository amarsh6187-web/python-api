import pickle
import sqlite3
import numpy as np
class DatabaseManager:
    """Handles all SQLite database operations."""
    def __init__(self, db_path):
        self.db_path = db_path
        self._initialize_db()

    def _get_connection(self):
        """Returns a new database connection."""
        return sqlite3.connect(self.db_path)

    def _initialize_db(self):
        """Initializes the database with the required tables and indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Table for perceptual hashes
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                filename TEXT NOT NULL PRIMARY KEY,
                p_hash TEXT NOT NULL
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_p_hash ON images (p_hash)")

            # Tables for face embeddings
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                top INTEGER, right INTEGER, bottom INTEGER, left INTEGER,
                embedding BLOB NOT NULL,
                UNIQUE(image_name, top, right, bottom, left)
            )
            ''')
            conn.commit()

    def add_phash(self, filename: str, p_hash: str):
        """Adds a new perceptual hash and filename to the database."""
        with self._get_connection() as conn:
            # The first value in the tuple should be the filename, then the hash
            conn.execute("INSERT OR IGNORE INTO images (filename, p_hash) VALUES (?, ?)", (filename, p_hash))
            conn.commit()

    def is_phash_duplicate(self, p_hash: str) -> bool:
        """Checks if a similar perceptual hash already exists in the database."""
        # This can be extended to check for similar (not just identical) hashes
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM images WHERE p_hash = ? LIMIT 1", (p_hash,))
            return cursor.fetchone() is not None

    def save_face(self, image_name: str, face_location: tuple, embedding: np.ndarray) -> int:
        """Saves a single face's data to the database."""
        top, right, bottom, left = [int(coord) for coord in face_location]
        encoded_embedding = sqlite3.Binary(pickle.dumps(embedding))
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO faces (image_name, top, right, bottom, left, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (image_name, top, right, bottom, left, encoded_embedding))
            conn.commit()
            return cursor.lastrowid

    def get_all_faces(self) -> list:
        """Retrieves all face records from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, image_name, top, right, bottom, left, embedding FROM faces")
            all_faces = []
            for row in cursor.fetchall():
                face_id, image_name, top, right, bottom, left, embedding_blob = row
                try:
                    embedding = pickle.loads(embedding_blob)
                    all_faces.append({
                        'id': face_id,
                        'image_name': image_name,
                        'face_location': (top, right, bottom, left),
                        'embedding': embedding
                    })
                except pickle.UnpicklingError:
                    print(f"Warning: Could not unpickle embedding for face ID {face_id}. Skipping.")
            return all_faces

    def is_image_processed_for_faces(self, filename: str) -> bool:
        """Checks if faces from a given image are already in the DB."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM faces WHERE image_name = ? LIMIT 1", (filename,))
            return cursor.fetchone() is not None
