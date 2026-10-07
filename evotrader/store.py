"""Persistence: an SQLite log of every life and snapshot, plus world checkpoints."""

import json
import os
import pickle
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY, parent INTEGER, root INTEGER, generation INTEGER,
    timeframe INTEGER, control INTEGER, kind TEXT, born_t REAL, died_t REAL,
    cause TEXT, growth REAL, trades INTEGER, fees REAL, funding REAL,
    liquidations INTEGER, children INTEGER, leverage REAL, margin_frac REAL,
    deadband REAL, sigma REAL, genome BLOB
);
CREATE INDEX IF NOT EXISTS agents_root ON agents(root);
CREATE TABLE IF NOT EXISTS snapshots (t REAL PRIMARY KEY, data TEXT);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class Store:
    def __init__(self, run_dir):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.db = sqlite3.connect(os.path.join(run_dir, "ecosystem.sqlite"))
        self.db.executescript(SCHEMA)

    @property
    def checkpoint_path(self):
        return os.path.join(self.run_dir, "world.pkl")

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    def drain(self, world):
        """Write the world's pending events to the database."""
        for kind, e in world.events:
            if kind == "birth":
                self.db.execute(
                    "INSERT OR REPLACE INTO agents (id, parent, root, generation, timeframe,"
                    " control, kind, born_t, leverage, margin_frac, deadband, sigma, genome)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (e["id"], e["parent"], e["root"], e["generation"], e["timeframe"],
                     int(e["control"]), e["kind"], e["t"], e["leverage"], e["margin_frac"],
                     e["deadband"], e["sigma"], e["genome"]))
            elif kind == "death":
                self.db.execute(
                    "UPDATE agents SET died_t=?, cause=?, growth=?, trades=?, fees=?,"
                    " funding=?, liquidations=?, children=? WHERE id=?",
                    (e["t"], e["cause"], e["growth"], e["trades"], e["fees"],
                     e["funding"], e["liquidations"], e["children"], e["id"]))
            elif kind == "snapshot":
                self.db.execute("INSERT OR REPLACE INTO snapshots VALUES (?, ?)",
                                (e["t"], json.dumps(e)))
        world.events.clear()

    def checkpoint(self, world):
        self.drain(world)
        self.db.commit()
        tmp = self.checkpoint_path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(world, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.checkpoint_path)

    def snapshots(self):
        rows = self.db.execute("SELECT data FROM snapshots ORDER BY t").fetchall()
        return [json.loads(r[0]) for r in rows]

    def close(self):
        self.db.commit()
        self.db.close()


def load_world(path):
    if os.path.isdir(path):
        path = os.path.join(path, "world.pkl")
    with open(path, "rb") as fh:
        return pickle.load(fh)
