"""Pruebas de las invariantes que sostienen todo lo demas.

Se ejecutan con:  python3 -m unittest discover -s tests -v
Las que necesitan material real se saltan solas si no hay ningun .dv a mano.
"""

import glob
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dvr import bitstream as bs
from dvr.fixup import fit_segment, sanitize_block, segment_bits, SEG_CAPACITY_BITS
from dvr.layout import PAL, NTSC, AREA_BITS, AREA_OFF, BLOCK


def any_dv():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for pat in ("clips/*.dv", "*.dv", "out/*.dv"):
        got = sorted(glob.glob(os.path.join(root, pat)))
        if got:
            return got[0]
    return None


class TestLayout(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(PAL.frame_size, 144000)
        self.assertEqual(PAL.n_video, 1620)
        self.assertEqual(PAL.n_audio, 108)
        self.assertEqual(PAL.n_seg, 324)
        self.assertEqual(PAL.mb_cols * PAL.mb_rows, PAL.n_video)
        self.assertEqual(NTSC.frame_size, 120000)
        self.assertEqual(NTSC.n_video, 1350)

    def test_areas(self):
        # las areas fijas de los 6 bloques DCT llenan justo el payload
        self.assertEqual(sum(a // 8 for a in AREA_BITS), BLOCK - 4)
        for j in range(1, 6):
            self.assertEqual(AREA_OFF[j], AREA_OFF[j - 1] + AREA_BITS[j - 1] // 8)

    def test_segments_dont_straddle_audio(self):
        # los 5 bloques de un segmento tienen que ir seguidos en el flujo
        for s in range(PAL.n_seg):
            off = PAL.seg[s]
            self.assertTrue(np.all(np.diff(off) == BLOCK),
                            f"el segmento {s} no es contiguo")


class TestVLC(unittest.TestCase):
    def test_prefix_code_is_complete(self):
        kraft = sum(2.0 ** -int(l) for l in bs.VLC_LEN)
        self.assertAlmostEqual(kraft, 1.0, places=9)

    def test_codes_unique_and_in_range(self):
        seen = set()
        for c, l in zip(bs.VLC_CODE, bs.VLC_LEN):
            self.assertLess(int(c), 1 << int(l))
            self.assertNotIn((int(l), int(c)), seen)
            seen.add((int(l), int(c)))

    def test_lookup_table_matches(self):
        for i in range(0, bs.NB_VLC, 7):
            l = int(bs.VLC_LEN[i])
            word = int(bs.VLC_CODE[i]) << (bs.MAX_LEN - l)
            self.assertEqual(int(bs._LUT[word]), i)

    def test_eob_terminates(self):
        self.assertEqual(int(bs.VLC_LEVEL[bs.EOB]), 0)
        self.assertGreaterEqual(int(bs.VLC_RUN[bs.EOB]) + 1, 64)


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.path = any_dv()
        if not self.path:
            self.skipTest("no hay ningun .dv con el que probar")

    def test_parse_pack_is_exact(self):
        from dvr import native, dcplane
        n = os.path.getsize(self.path) // PAL.frame_size
        M = np.memmap(self.path, dtype=np.uint8, mode="r", shape=(n, PAL.frame_size))
        tot = ok = 0
        for fi in np.linspace(0, n - 1, min(6, n)).astype(int):
            f = np.array(M[fi])
            sta = dcplane.sta(f, PAL).reshape(PAL.n_seg, 5)
            for s in range(PAL.n_seg):
                if sta[s].any():
                    continue
                off = int(PAL.seg[s, 0])
                buf = f[off:off + 400].tobytes()
                seg = native.parse(buf)
                if not seg.ok:          # falso sano: no es cosa del parser
                    continue
                out, lost = native.pack(seg, buf)
                tot += 1
                ok += (out == buf and lost == 0)
        self.assertGreater(tot, 100, "muy pocos segmentos sanos para la prueba")
        self.assertEqual(ok, tot, f"{tot - ok} de {tot} segmentos no salen iguales")

    def test_c_matches_python(self):
        from dvr import native, dcplane
        n = os.path.getsize(self.path) // PAL.frame_size
        M = np.memmap(self.path, dtype=np.uint8, mode="r", shape=(n, PAL.frame_size))
        f = np.array(M[n // 2])
        sta = dcplane.sta(f, PAL).reshape(PAL.n_seg, 5)
        checked = 0
        for s in range(PAL.n_seg):
            if sta[s].any() or checked >= 20:
                continue
            off = int(PAL.seg[s, 0])
            buf = f[off:off + 400].tobytes()
            c = native.parse(buf)
            p = bs.parse_segment(buf)
            if not c.ok:
                continue
            checked += 1
            for m in range(5):
                for j in range(6):
                    cb, pb = c.mb[m].b[j], p.mb[m][j]
                    self.assertEqual(cb.dc, pb.dc)
                    self.assertEqual(cb.mode, pb.mode)
                    self.assertEqual(cb.cls, pb.cls)
                    self.assertEqual(list(cb.tok[:cb.ntok]), list(pb.tokens))
        self.assertGreater(checked, 0)


class TestFixup(unittest.TestCase):
    def setUp(self):
        self.path = any_dv()
        if not self.path:
            self.skipTest("no hay ningun .dv con el que probar")

    def _a_segment(self):
        from dvr import native, dcplane
        n = os.path.getsize(self.path) // PAL.frame_size
        M = np.memmap(self.path, dtype=np.uint8, mode="r", shape=(n, PAL.frame_size))
        for fi in range(min(n, 40)):
            f = np.array(M[fi])
            sta = dcplane.sta(f, PAL).reshape(PAL.n_seg, 5)
            for s in range(PAL.n_seg):
                if sta[s].any():
                    continue
                off = int(PAL.seg[s, 0])
                seg = native.parse(bytes(f[off:off + 400]))
                if seg.ok:
                    return seg, bytes(f[off:off + 400])
        self.skipTest("no se ha encontrado ningun segmento sano")

    def test_fit_is_a_noop_when_it_already_fits(self):
        seg, _ = self._a_segment()
        self.assertLessEqual(segment_bits(seg), SEG_CAPACITY_BITS)
        self.assertEqual(fit_segment(seg), 0)

    def test_fit_trims_and_keeps_eob(self):
        seg, raw = self._a_segment()
        # se fuerza un segmento que no cabe duplicando los coeficientes
        for m in range(5):
            for j in range(6):
                b = seg.mb[m].b[j]
                n = int(b.ntok)
                for i in range(n - 1):
                    if n + i >= 70:
                        break
                    b.tok[n + i] = b.tok[i]
                b.ntok = min(70, 2 * n - 1)
        self.assertGreater(segment_bits(seg), SEG_CAPACITY_BITS)
        dropped = fit_segment(seg)
        self.assertGreater(dropped, 0)
        self.assertLessEqual(segment_bits(seg), SEG_CAPACITY_BITS)
        # y lo que salga tiene que volver a parsear
        from dvr import native
        buf, lost = native.pack(seg, raw)
        self.assertEqual(lost, 0)
        self.assertTrue(native.parse(buf).ok, "el segmento recortado no parsea")

    def test_sanitize_closes_an_unfinished_block(self):
        seg, raw = self._a_segment()
        b = seg.mb[0].b[0]
        b.done = 0
        b.pos = 10
        b.ntok_area = min(2, int(b.ntok))
        self.assertTrue(sanitize_block(b, False))
        self.assertTrue(b.done)
        self.assertEqual(int(b.tok[int(b.ntok) - 1]), bs.EOB)


class TestOrder(unittest.TestCase):
    def test_topo_order_with_gaps(self):
        from dvr.match import _topo_order

        class Cap:
            def __init__(self, n):
                self.n = n

        # captura 0: frames 0,1,2   captura 1: frames 0,1,2,3
        # el frame 1 de la captura 1 es un hueco que la 0 no tiene
        clusters = [[(0, 0), (1, 0)], [(1, 1)], [(0, 1), (1, 2)],
                    [(0, 2), (1, 3)]]
        cid = {}
        for k, g in enumerate(clusters):
            for r in g:
                cid[r] = k
        order = _topo_order(clusters, cid, [Cap(3), Cap(4)])
        self.assertEqual(len(order), 4)
        self.assertEqual(order, [0, 1, 2, 3])

    def test_chains_are_not_interleaved(self):
        """Dos capturas enlazadas por pocas anclas no deben intercalarse.

        Es el fallo que metia frames de momentos distintos alternados: al
        desempatar por el numero de frame CRUDO, el frame 1 de una captura se
        colaba junto al frame 1 de otra aunque correspondieran a puntos de
        cinta lejanos. El desempate tiene que ser la posicion estimada.
        """
        from dvr.match import _topo_order, estimate_position

        class Cap:
            def __init__(self, n):
                self.n = n

        # captura 0 (columna vertebral): frames 0,1,2,3
        # captura 1: su frame 0 casa con el 1 de la base, y su frame 3 con el 2
        #            -> sus frames 1 y 2 van ENTRE medias, no al principio
        clusters = [[(0, 0)], [(0, 1), (1, 0)], [(1, 1)], [(1, 2)],
                    [(0, 2), (1, 3)], [(0, 3)]]
        cid = {}
        for k, g in enumerate(clusters):
            for r in g:
                cid[r] = k
        caps = [Cap(4), Cap(4)]
        pos = estimate_position(clusters, cid, caps, spine=0)
        self.assertTrue(pos[1] < pos[2] < pos[4], f"posiciones raras: {pos}")
        self.assertTrue(pos[2] < pos[3] < pos[4], f"posiciones raras: {pos}")
        order = _topo_order(clusters, cid, caps)
        rank = {c: i for i, c in enumerate(order)}
        self.assertLess(rank[1], rank[2])
        self.assertLess(rank[3], rank[4])
        self.assertLess(rank[4], rank[5])
        # y el frame 0 de la base sigue siendo el primero
        self.assertEqual(order[0], 0)

    def test_union_refuses_same_capture(self):
        """Una captura no puede leer dos veces el mismo frame de cinta.

        Aceptar esa union convertia la cadena que separa los dos frames en un
        CICLO, y con el orden topologico un ciclo temprano bloquea todo lo que
        va detras: en el arranque de la cinta dejaba el 84% de los grupos sin
        ordenar por topologia.
        """
        from dvr.match import Union
        # 5 lecturas: 0 y 1 de la captura A, 2 y 3 de la B, 4 de la C
        cap = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
        uf = Union(5, cap_of=lambda i: cap[i])
        self.assertIsNotNone(uf.union(0, 2))     # A0 con B0: bien
        self.assertIsNone(uf.union(0, 1))        # A0 con A1: misma captura
        self.assertIsNone(uf.union(1, 2))        # A1 con el grupo que ya tiene A
        self.assertEqual(uf.refused, 2)
        self.assertIsNotNone(uf.union(0, 4))     # C cabe
        self.assertEqual(uf.find(4), uf.find(2))
        self.assertNotEqual(uf.find(1), uf.find(0))

    def test_topo_order_breaks_cycles_locally(self):
        """Un ciclo no debe arrastrar a todo lo que va detras."""
        from dvr.match import _topo_order

        class Cap:
            def __init__(self, n):
                self.n = n

        # cadena 0->1->2->3->4 con un ciclo entre 1 y 2 (dos capturas que se
        # contradicen), y una cola larga detras que NO debe desordenarse
        clusters = [[(0, 0)], [(0, 1), (1, 1)], [(0, 2), (1, 0)],
                    [(0, 3)], [(0, 4)]]
        cid = {}
        for k, g in enumerate(clusters):
            for r in g:
                cid[r] = k
        order = _topo_order(clusters, cid, [Cap(5), Cap(2)])
        self.assertEqual(len(order), 5)
        self.assertEqual(len(set(order)), 5)
        rank = {c: i for i, c in enumerate(order)}
        # la cola detras del ciclo sigue en orden
        self.assertLess(rank[3], rank[4])
        self.assertEqual(order[0], 0)

    def test_topo_order_survives_a_cycle(self):
        from dvr.match import _topo_order

        class Cap:
            def __init__(self, n):
                self.n = n

        # emparejamiento falso: el frame 2 de la captura 0 cae en el mismo
        # grupo que el 0, lo que crea un ciclo
        clusters = [[(0, 0), (0, 2)], [(0, 1)]]
        cid = {(0, 0): 0, (0, 2): 0, (0, 1): 1}
        order = _topo_order(clusters, cid, [Cap(3)])
        self.assertEqual(sorted(order), [0, 1])


if __name__ == "__main__":
    unittest.main()
