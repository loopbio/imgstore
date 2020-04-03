import os.path
import sqlite3
import logging
import operator

import yaml
import numpy as np

from .constants import FRAME_MD


def _load_index(path_without_extension):
    for extension in ('.npz', '.yaml'):
        path = path_without_extension + extension
        if os.path.exists(path):
            if extension == '.yaml':
                with open(path, 'rt') as f:
                    dat = yaml.safe_load(f)
                    return {k: dat[k] for k in FRAME_MD}
            elif extension == '.npz':
                with open(path, 'rb') as f:
                    dat = np.load(f)
                    return {k: dat[k].tolist() for k in FRAME_MD}

    raise IOError('could not find index %s' % path_without_extension)


# noinspection SqlNoDataSourceInspection,SqlDialectInspection,SqlResolve
class ImgStoreIndex(object):

    VERSION = '1'

    log = logging.getLogger('imgstore.index')

    def __init__(self, db=None):
        self._conn = db

        cur = self._conn.cursor()

        cur.execute('SELECT value FROM index_information WHERE name = ?', ('version', ))
        v, = cur.fetchone()
        if v != self.VERSION:
            raise IOError('incorrect index version: %s vs %s' % (v, self.VERSION))

        cur.execute('SELECT COUNT(1) FROM frames')
        self.frame_count, = cur.fetchone()

        def _summary(_what):
            cur.execute('SELECT value FROM summary WHERE name = ?', (_what,))
            return cur.fetchone()[0]

        if self.frame_count:
            self.frame_time_max = _summary('frame_time_max')
            self.frame_time_min = _summary('frame_time_min')
            self.frame_max = _summary('frame_max')
            self.frame_min = _summary('frame_min')

            # keep back compat for nan as types (inf -> nan)
            if not np.isreal(self.frame_max):
                self.frame_max = np.nan
            if not np.isreal(self.frame_min):
                self.frame_min = np.nan
        else:
            self.frame_max = self.frame_min = np.nan
            self.frame_time_max = self.frame_time_min = 0.0

        self.log.debug('frame range %f -> %f' % (self.frame_min, self.frame_max))

        # # all chunks in the store [0,1,2, ... ]
        cur.execute('SELECT chunk FROM chunks ORDER BY chunk;')
        self._chunks = tuple(row[0] for row in cur)

    @classmethod
    def create_database(cls, conn):
        c = conn.cursor()
        # Create tables
        c.execute('CREATE TABLE frames '
                  '(chunk INTEGER, frame_idx INTEGER, frame_number INTEGER, frame_time REAL)')
        c.execute('CREATE TABLE chunks '
                  '(chunk INTEGER, chunk_path TEXT)')
        c.execute('CREATE TABLE index_information '
                  '(name TEXT, value TEXT)')
        c.execute('CREATE TABLE summary '
                  '(name TEXT, value REAL)')
        c.execute('INSERT into index_information VALUES (?, ?)', ('version', cls.VERSION))
        conn.commit()

    @classmethod
    def new_from_chunks(cls, chunk_n_and_chunk_paths):
        db = sqlite3.connect(':memory:', check_same_thread=False)
        cls.create_database(db)

        frame_count = 0
        frame_max = -np.inf
        frame_min = np.inf
        frame_time_max = -np.inf
        frame_time_min = np.inf

        cur = db.cursor()

        for chunk_n, chunk_path in sorted(chunk_n_and_chunk_paths, key=operator.itemgetter(0)):
            try:
                idx = _load_index(chunk_path)
            except IOError:
                cls.log.warn('missing index for chunk %s' % chunk_n)
                continue

            if not idx['frame_number']:
                # empty chunk
                continue

            frame_count += len(idx['frame_number'])
            frame_time_min = min(frame_time_min, np.min(idx['frame_time']))
            frame_time_max = max(frame_time_max, np.max(idx['frame_time']))
            frame_min = min(frame_min, np.min(idx['frame_number']))
            frame_max = max(frame_max, np.max(idx['frame_number']))

            records = [(chunk_n, i, fn, ft) for i, (fn, ft) in enumerate(zip(idx['frame_number'],
                                                                             idx['frame_time']))]
            cur.executemany('INSERT INTO frames VALUES (?,?,?,?)', records)
            cur.execute('INSERT INTO chunks VALUES (?, ?)', (chunk_n, chunk_path))

            db.commit()

        cur.execute('INSERT INTO summary VALUES (?,?)', ('frame_time_min', float(frame_time_min)))
        cur.execute('INSERT INTO summary VALUES (?,?)', ('frame_time_max', float(frame_time_max)))
        cur.execute('INSERT INTO summary VALUES (?,?)', ('frame_min', float(frame_min)))
        cur.execute('INSERT INTO summary VALUES (?,?)', ('frame_max', float(frame_max)))

        db.commit()

        return cls(db)

    @classmethod
    def new_from_file(cls, path):
        db = sqlite3.connect(path, check_same_thread=False)
        return cls(db)

    @staticmethod
    def _get_metadata(cur):
        md = {'frame_number': [], 'frame_time': []}
        for row in cur:
            md['frame_number'].append(row[0])
            md['frame_time'].append(row[1])
        return md

    @property
    def chunks(self):
        """ the number of non-empty chunks that contain images """
        return self._chunks

    def to_file(self, path):
        db = sqlite3.connect(path)
        with db:
            for line in self._conn.iterdump():
                # let python handle the transactions
                if line not in ('BEGIN;', 'COMMIT;'):
                    db.execute(line)
        db.commit()
        db.close()

    def get_all_metadata(self):
        cur = self._conn.cursor()
        cur.execute("SELECT frame_number, frame_time FROM frames ORDER BY rowid;")
        return self._get_metadata(cur)

    def get_chunk_metadata(self, chunk_n):
        cur = self._conn.cursor()
        cur.execute("SELECT frame_number, frame_time FROM frames WHERE chunk = ? ORDER BY rowid;", (chunk_n, ))
        return self._get_metadata(cur)

    def find_chunk(self, what, value):
        assert what in ('frame_number', 'frame_time', 'index')
        cur = self._conn.cursor()

        if what == 'index':
            cur.execute("SELECT chunk, frame_idx FROM frames ORDER BY rowid LIMIT 1 OFFSET {};".format(int(value)))
        else:
            cur.execute("SELECT chunk, frame_idx FROM frames WHERE {} = ?;".format(what), (value, ))

        try:
            chunk_n, frame_idx = cur.fetchone()
        except TypeError:  # no result
            return -1, -1

        return chunk_n, frame_idx

    def find_chunk_nearest(self, what, value):
        assert what in ('frame_number', 'frame_time')
        cur = self._conn.cursor()
        cur.execute("SELECT chunk, frame_idx FROM frames ORDER BY ABS(? - {}) LIMIT 1;".format(what), (value, ))
        chunk_n, frame_idx = cur.fetchone()
        return chunk_n, frame_idx