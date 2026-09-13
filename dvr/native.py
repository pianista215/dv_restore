"""Acelerador en C del parser de bitstream, cargado con ctypes.

Se compila solo la primera vez y se cachea. Si gcc no esta disponible, todo
sigue funcionando con la version de Python de dvr/bitstream.py, solo que mas
despacio. La de Python es la version de referencia: es la que se verifica con
el round-trip byte a byte, y la de C se comprueba contra ella.
"""

import ctypes
import os
import subprocess
import numpy as np

from . import bitstream as bs

MAXTOK = 72
NDCT = 6
SEG_MB = 5
SEG_BYTES = 400

_SRC = os.path.join(os.path.dirname(__file__), "_native", "dvbits.c")
_LIB = None


class DvBlk(ctypes.Structure):
    _fields_ = [("dc", ctypes.c_int16), ("mode", ctypes.c_uint8),
                ("cls", ctypes.c_uint8), ("pos", ctypes.c_int16),
                ("done", ctypes.c_uint8), ("ntok", ctypes.c_int16),
                ("ntok_area", ctypes.c_int16),
                ("tok", ctypes.c_uint16 * MAXTOK)]


class DvMB(ctypes.Structure):
    _fields_ = [("b", DvBlk * NDCT), ("qno", ctypes.c_uint8),
                ("sta", ctypes.c_uint8), ("id", ctypes.c_uint8 * 3)]


class DvSeg(ctypes.Structure):
    _fields_ = [("mb", DvMB * SEG_MB), ("raw", ctypes.c_uint8 * SEG_BYTES),
                ("ok", ctypes.c_int32), ("bad_mb", ctypes.c_int32),
                ("bad_blk", ctypes.c_int32)]


def _build(so):
    os.makedirs(os.path.dirname(so), exist_ok=True)
    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", so, _SRC],
                   check=True, capture_output=True)


def lib():
    """Devuelve la biblioteca cargada, compilandola si hace falta."""
    global _LIB
    if _LIB is not None:
        return _LIB
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    so = os.path.join(root, "work", "dvbits.so")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(_SRC):
        _build(so)
    L = ctypes.CDLL(so)
    L.dv_set_tables.argtypes = [
        np.ctypeslib.ndpointer(np.uint16, flags="C"),
        np.ctypeslib.ndpointer(np.uint8, flags="C"),
        np.ctypeslib.ndpointer(np.int16, flags="C"),
        np.ctypeslib.ndpointer(np.int16, flags="C"), ctypes.c_int]
    L.dv_parse_segment.argtypes = [ctypes.c_char_p, ctypes.POINTER(DvSeg)]
    L.dv_pack_segment.argtypes = [ctypes.POINTER(DvSeg), ctypes.c_char_p,
                                  ctypes.c_char_p]
    L.dv_scan_frame.argtypes = [
        np.ctypeslib.ndpointer(np.uint8, flags="C"),
        np.ctypeslib.ndpointer(np.int32, flags="C"), ctypes.c_int,
        np.ctypeslib.ndpointer(np.uint8, flags="C"),
        np.ctypeslib.ndpointer(np.int16, flags="C"),
        np.ctypeslib.ndpointer(np.int32, flags="C"),
        np.ctypeslib.ndpointer(np.int16, flags="C")]
    L.dv_seg_size.restype = ctypes.c_int
    if L.dv_seg_size() != ctypes.sizeof(DvSeg):
        raise RuntimeError(
            f"la estructura DvSeg no coincide: C={L.dv_seg_size()} "
            f"ctypes={ctypes.sizeof(DvSeg)}")
    L.dv_set_tables(np.ascontiguousarray(bs.VLC_CODE, np.uint16),
                    np.ascontiguousarray(bs.VLC_LEN, np.uint8),
                    np.ascontiguousarray(bs.VLC_RUN, np.int16),
                    np.ascontiguousarray(bs.VLC_LEVEL, np.int16),
                    len(bs.VLC_LEN))
    _LIB = L
    return L


def parse(buf):
    seg = DvSeg()
    lib().dv_parse_segment(bytes(buf), ctypes.byref(seg))
    return seg


def pack(seg, pad_from):
    out = ctypes.create_string_buffer(SEG_BYTES)
    lost = lib().dv_pack_segment(ctypes.byref(seg), bytes(pad_from), out)
    return out.raw[:SEG_BYTES], lost


class FrameScan:
    """Resultado de barrer un frame: validez de cada segmento y rasgos DCT."""

    __slots__ = ("seg_ok", "nnz", "energy", "maxpos", "n_bad")

    def __init__(self, n_seg):
        self.seg_ok = np.zeros(n_seg, np.uint8)
        self.nnz = np.zeros(n_seg * SEG_MB * NDCT, np.int16)
        self.energy = np.zeros(n_seg * SEG_MB * NDCT, np.int32)
        self.maxpos = np.zeros(n_seg * SEG_MB * NDCT, np.int16)
        self.n_bad = 0

    def reshape(self, prof):
        self.nnz = self.nnz.reshape(prof.n_seg * SEG_MB, NDCT)
        self.energy = self.energy.reshape(prof.n_seg * SEG_MB, NDCT)
        self.maxpos = self.maxpos.reshape(prof.n_seg * SEG_MB, NDCT)
        return self


def scan_frame(frame, prof):
    """Barre los 324 segmentos de un frame. Devuelve un FrameScan."""
    r = FrameScan(prof.n_seg)
    off = np.ascontiguousarray(prof.seg[:, 0], np.int32)
    f = np.ascontiguousarray(frame, np.uint8)
    r.n_bad = lib().dv_scan_frame(f, off, prof.n_seg, r.seg_ok, r.nnz,
                                  r.energy, r.maxpos)
    return r.reshape(prof)
