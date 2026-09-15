"""Video de referencia (un DVD) como ayuda para la ocultacion.

La cinta manda: esto solo aporta luma para los macrobloques que no tienen dato
real en ninguna pasada. Todo lo que sabe de MPEG-2 vive aqui; el resto del repo
solo ve una funcion  ref(i, k) -> 16x16 float | None.

Geometria, medida sobre los ficheros reales (no supuesta):

  columna x del VOB  <->  columna x+8 del DV      (recorte limpio, sin escalar)
  fila    y del VOB  <->  fila    y+1 del DV

El recorte de 8 columnas por lado es el de siempre entre los 720 muestreos de
BT.601 y los 704 "activos" que usa el DVD; misma frecuencia de muestreo, asi
que no hay reescalado que deshacer.
"""

import json
import os
import subprocess

import numpy as np

REF_W, REF_H = 704, 576
REF_DX, REF_DY = 8, 1

# Celdas de 16x16 del DV que el VOB cubre entera: sobran la fila 0 y las
# columnas 0 y 44, que se salen del recorte. Son 35*43 = 1505 de 1620 (92,9%).
CELL_R0, CELL_R1 = 1, 36
CELL_C0, CELL_C1 = 1, 44


class FieldMap:
    """De que campo del VOB sale cada campo del DV.

    Para la paridad p del DV (0 = filas pares), la terna (dframe, q, line)
    significa:  fila 2i+p del DV  <-  fila 2*(i+line)+q del frame v+dframe.

    El valor por defecto es el medido: los dos campos salen del MISMO frame del
    VOB y el mapeo se reduce a un desplazamiento uniforme de una linea
    (fila r del DV <- fila r-1 del VOB). La paridad queda intercambiada, que es
    lo correcto: el DV PAL es bottom-field-first y el VOB top-field-first, o sea
    que el campo que va primero en el tiempo en uno cae en el que va primero en
    el otro.

    Se guarda como parametro, y no como constante, porque la alternativa
    (campos separados por medio frame) solo se distingue con movimiento y
    cambiaria de donde se leen los pixeles sin cambiar nada mas.
    """

    __slots__ = ("even", "odd")

    def __init__(self, even=(0, 1, -1), odd=(0, 0, 0)):
        self.even = tuple(even)
        self.odd = tuple(odd)

    @property
    def uniform_shift(self):
        """El desplazamiento de filas si los dos campos salen del mismo frame.

        Devuelve None si el mapa no se puede reducir a un desplazamiento unico
        (campos de frames distintos, o paridades incoherentes).
        """
        offs = set()
        for p, (df, q, line) in ((0, self.even), (1, self.odd)):
            if df != 0:
                return None
            offs.add((2 * line + q) - p)
        return offs.pop() if len(offs) == 1 else None

    def to_json(self):
        return {"even": list(self.even), "odd": list(self.odd)}

    @classmethod
    def from_json(cls, d):
        return cls(tuple(d["even"]), tuple(d["odd"]))

    def __repr__(self):
        u = self.uniform_shift
        extra = f", desplazamiento uniforme {u:+d}" if u is not None else ""
        return f"<FieldMap par={self.even} impar={self.odd}{extra}>"


def _probe_frames(path):
    # por nombre, no por posicion: ffprobe los saca en el orden en que estan
    # declarados en el stream, no en el que se piden
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,width,height",
         "-of", "default=nw=0:nk=0", path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout
    got = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            if v.strip().isdigit():
                got[k.strip()] = int(v)
    try:
        return got["nb_read_frames"], got["width"], got["height"]
    except KeyError:
        raise RuntimeError(f"ffprobe no sabe decir cuantos frames tiene {path}")


def check_range_safe(path):
    """Comprueba que sacar la luma no expande el rango de estudio.

    Pedir 'gray' a ffmpeg expande 16-235 a 0-255 (medido: gray = 1,1348*Y-14,95).
    'extractplanes=y' entrega el plano Y tal cual. Esto lo verifica en vez de
    darlo por supuesto, porque la misma trampa ya costo una hora una vez.
    """
    def dec(vf, pix):
        return subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", path, "-vf", f"select='lt(n\\,1)',{vf}",
             "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", pix, "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    n = REF_W * REF_H
    plane = dec("null", "yuv420p")[:n]
    return dec("extractplanes=y", "gray")[:n] == plane


def extract(vob_path, out_dir, verbose=True):
    """Decodifica el video entero a luma cruda, en UNA pasada secuencial.

    Deja en <out_dir>/refvideo/:
        luma.raw   (n, 576, 704) uint8   ~9,2 GB para 15 minutos
        sig.npy    (n, 35, 43)  float32  firma por frame, en rejilla del DV
        meta.json  n, tamano y mtime del origen, y el mapa de campos

    Nada de buscar por tiempo: el contenedor de un VOB miente (ffprobe da
    duration=0,79 en un fichero de 15 minutos) y un salto mal hecho desplaza un
    frame en silencio, lo que es invisible en un plano fijo y demoledor en un
    barrido. Decodificando desde el frame 0, el numero de frame es cierto por
    construccion.

    Resumible: si meta.json cuadra con el origen, no hace nada.
    """
    d = os.path.join(out_dir, "refvideo")
    os.makedirs(d, exist_ok=True)
    meta_p = os.path.join(d, "meta.json")
    raw_p = os.path.join(d, "luma.raw")
    sig_p = os.path.join(d, "sig.npy")

    st = os.stat(vob_path)
    want = {"source": os.path.abspath(vob_path),
            "size": st.st_size, "mtime": int(st.st_mtime)}
    if os.path.exists(meta_p) and os.path.exists(raw_p) and os.path.exists(sig_p):
        have = json.load(open(meta_p))
        if all(have.get(k) == v for k, v in want.items()):
            if verbose:
                print(f"ya extraido: {have['n']} frames en {d}")
            return RefStore(out_dir)

    if not check_range_safe(vob_path):
        raise RuntimeError(
            "extractplanes=y no da el plano Y tal cual en este ffmpeg; "
            "la luma saldria con el rango expandido")

    n, w, h = _probe_frames(vob_path)
    if (w, h) != (REF_W, REF_H):
        raise RuntimeError(f"esperaba {REF_W}x{REF_H}, el video es {w}x{h}")
    if verbose:
        print(f"extrayendo {n} frames ({n/25:.0f}s) de {os.path.basename(vob_path)}")
        print(f"  -> {raw_p}  ({n*w*h/1e9:.1f} GB)")

    fs = w * h
    sig = np.zeros((n, CELL_R1 - CELL_R0, CELL_C1 - CELL_C0), np.float32)
    part = raw_p + ".part"
    p = subprocess.Popen(
        ["ffmpeg", "-v", "quiet", "-i", vob_path, "-map", "0:v:0",
         "-vf", "extractplanes=y", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=fs * 4)
    got = 0
    with open(part, "wb") as fh:
        while got < n:
            buf = p.stdout.read(fs)
            if len(buf) < fs:
                break
            fh.write(buf)
            sig[got] = _signature(np.frombuffer(buf, np.uint8).reshape(h, w))
            got += 1
            if verbose and got % 2000 == 0:
                print(f"  {got}/{n} frames...")
    p.stdout.close()
    p.wait()
    if got < n:
        if verbose:
            print(f"  el video se acabo en {got} frames (ffprobe decia {n})")
        n = got
        sig = sig[:n]
    os.replace(part, raw_p)
    np.save(sig_p, sig)
    want["n"] = n
    want["fields"] = FieldMap().to_json()
    json.dump(want, open(meta_p, "w"), indent=1)
    if verbose:
        print(f"listo: {n} frames")
    return RefStore(out_dir)


def _signature(luma):
    """Medias de 16x16 sobre las celdas del DV que el VOB cubre enteras."""
    r0 = 16 * CELL_R0 - REF_DY
    c0 = 16 * CELL_C0 - REF_DX
    nr, nc = CELL_R1 - CELL_R0, CELL_C1 - CELL_C0
    sub = luma[r0:r0 + 16 * nr, c0:c0 + 16 * nc]
    return sub.reshape(nr, 16, nc, 16).mean(axis=(1, 3)).astype(np.float32)


class RefStore:
    """Acceso por frame a la luma ya extraida. Sin ffmpeg, sin seek."""

    def __init__(self, out_dir):
        d = os.path.join(out_dir, "refvideo")
        self.dir = d
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.n = self.meta["n"]
        self.fields = FieldMap.from_json(self.meta["fields"])
        self.data = np.memmap(os.path.join(d, "luma.raw"), dtype=np.uint8,
                              mode="r", shape=(self.n, REF_H, REF_W))
        self.sigs = np.load(os.path.join(d, "sig.npy"), mmap_mode="r")

    def luma(self, v):
        return self.data[v]

    def field(self, v, q):
        """Campo q del frame v: (288, 704). Primitiva de lectura."""
        return self.data[v][q::2]

    def sig(self, v):
        return np.asarray(self.sigs[v], np.float32)

    def frame_dv(self, v, fm=None):
        """El frame v puesto en la geometria del DV: (576, 720) float32.

        Fuera de la cobertura va NaN, para que nada lo use por accidente.
        Se compone campo a campo aunque el mapa sea un desplazamiento uniforme:
        asi, si algun dia resulta que los campos vienen separados, cambia el
        FieldMap y no el codigo.
        """
        fm = fm or self.fields
        out = np.full((REF_H, 720), np.nan, np.float32)
        for p, (df, q, line) in ((0, fm.even), (1, fm.odd)):
            w = v + df
            if not (0 <= w < self.n):
                continue
            src = self.field(w, q).astype(np.float32)      # (288, 704)
            nf = REF_H // 2
            i = np.arange(nf)
            j = i + line
            ok = (j >= 0) & (j < nf)
            out[2 * i[ok] + p, REF_DX:REF_DX + REF_W] = src[j[ok]]
        return out

    def covers(self, r, c):
        """Si la celda (fila, columna) de macrobloque esta cubierta entera."""
        return CELL_R0 <= r < CELL_R1 and CELL_C0 <= c < CELL_C1


# --------------------------------------------------------------------------
# Alineacion
# --------------------------------------------------------------------------

def dv_signatures(cap, table=None, chunk=512, progress=None):
    """Firma por frame de una captura DV, sin decodificar y sin ffmpeg.

    Devuelve (sig, health):
        sig    (n, 35, 43) float32, en las mismas celdas que la firma del VOB
        health (n,) float32, fraccion de macrobloques que la camara da por sanos

    El DC de cada macrobloque esta en posicion fija (ver dcplane), asi que la
    miniatura sale gratis. Es la misma magnitud que usa FrameInfo.grid_dc.
    """
    from . import dcplane, shuffle
    prof = cap.prof
    table = shuffle.load(prof) if table is None else table
    nr, nc = CELL_R1 - CELL_R0, CELL_C1 - CELL_C0
    sig = np.empty((cap.n, nr, nc), np.float32)
    health = np.empty(cap.n, np.float32)
    npos = prof.mb_rows * prof.mb_cols
    for a in range(0, cap.n, chunk):
        b = min(a + chunk, cap.n)
        blk = cap.data[a:b]
        dc = dcplane.dc_raw(blk, prof)[:, :, :4].mean(axis=2)     # (m, n_video)
        g = np.zeros((b - a, npos), np.float32)
        g[:, table] = dc
        sig[a:b] = g.reshape(-1, prof.mb_rows, prof.mb_cols)[
            :, CELL_R0:CELL_R1, CELL_C0:CELL_C1]
        health[a:b] = 1.0 - (dcplane.sta(blk, prof) != 0).mean(axis=1)
        if progress:
            progress(b, cap.n)
    return sig, health


def _normalize(sig):
    """Aplana, quita la media y normaliza: correlacion = producto escalar.

    Asi la comparacion es invariante a la afin fotometrica entre el DVD y el
    DV, que es justo lo que no queremos tener que calibrar para alinear.
    """
    f = sig.reshape(len(sig), -1).astype(np.float32)
    f = f - f.mean(axis=1, keepdims=True)
    n = np.linalg.norm(f, axis=1, keepdims=True)
    return f / np.maximum(n, 1e-6)


def _anchors(R, D, health, every=25, min_corr=0.90, min_health=0.30):
    """Puntos de anclaje fiables (f, v), ya filtrados y hechos monotonos."""
    fs = [f for f in range(0, len(D), every) if health[f] >= min_health]
    if not fs:
        return []
    c = R @ D[fs].T                                  # (n_ref, n_anchor)
    best = c.argmax(axis=0)
    val = c[best, np.arange(len(fs))]
    cand = [(f, int(v), float(s)) for f, v, s in zip(fs, best, val)
            if s >= min_corr]
    if not cand:
        return []
    # Subsecuencia no decreciente de peso maximo: una sola ancla disparatada
    # arrastra toda la banda, y con escenas estaticas las hay seguro.
    n = len(cand)
    w = [c[2] for c in cand]
    bestw = list(w)
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if cand[j][1] <= cand[i][1] and bestw[j] + w[i] > bestw[i]:
                bestw[i] = bestw[j] + w[i]
                prev[i] = j
    i = int(np.argmax(bestw))
    keep = []
    while i >= 0:
        keep.append(cand[i])
        i = prev[i]
    return keep[::-1]


def align(sig_ref, sig_dv, health, band=60, max_skip=24, lam=0.010, mu=0.060,
          every=25, verbose=True):
    """Mapa frame del DV -> frame del video de referencia.

    Programacion dinamica monotona y bandeada. Las transiciones son:
        v -> v+1     coste 0     (lo normal: los dos van a 25 fps)
        v -> v+1+g   lam*g       (la captura se salto g frames de cinta)
        v -> v       mu          (la captura tiene un frame que el DVD no)

    Se hace por programacion dinamica y no por argmax porque en los planos
    fijos la correlacion empata (medido: 0,6343 contra 0,6212 en el segundo
    candidato). El argmax se la juega a cara o cruz; el camino queda sujeto por
    los vecinos de los dos lados.

    Devuelve (v_of_f, cost_of_f, margin_of_f), los tres de longitud n_dv.
    margin_of_f es cuanto mejor es el mejor candidato de la banda frente al
    mejor NO adyacente: por debajo de ~0,03 el frame no se distingue solo.
    """
    R = _normalize(sig_ref)
    D = _normalize(sig_dv)
    nref, ndv = len(R), len(D)

    anc = _anchors(R, D, health, every=every)
    if len(anc) < 2:
        raise RuntimeError("no hay anclas fiables: el video de referencia no "
                           "parece contener este material")
    if verbose:
        print(f"  anclas fiables: {len(anc)} "
              f"(de {ndv // every} probadas), desfase de {anc[0][1]-anc[0][0]:+d} "
              f"a {anc[-1][1]-anc[-1][0]:+d}")

    af = np.array([a[0] for a in anc])
    av = np.array([a[1] for a in anc])
    centre = np.interp(np.arange(ndv), af, av)
    base = np.clip(np.rint(centre).astype(np.int64) - band, 0,
                   max(0, nref - (2 * band + 1)))
    B = 2 * band + 1

    # coste por celda, winsorizado por fila: un frame estropeado del VOB tiene
    # que costar una cantidad acotada, no arrastrar el camino
    cost = np.empty((ndv, B), np.float32)
    for f in range(ndv):
        v0 = base[f]
        cost[f] = 1.0 - (R[v0:v0 + B] @ D[f])
    hi = np.percentile(cost, 90, axis=1, keepdims=True)
    cost = np.minimum(cost, hi)
    cost[health < 0.05] = 0.0        # frame casi todo invencion: que decida el prior

    margin = np.empty(ndv, np.float32)
    for f in range(ndv):
        row = cost[f]
        j = int(row.argmin())
        far = np.abs(np.arange(B) - j) > 3
        margin[f] = float(row[far].min() - row[j]) if far.any() else 0.0

    NEG = np.float32(1e9)
    dp = cost[0].copy()                       # inicio libre
    bp = np.full((ndv, B), -1, np.int16)
    for f in range(1, ndv):
        sh = int(base[f - 1] - base[f])
        best = np.full(B, NEG, np.float32)
        arg = np.full(B, -1, np.int16)
        for delta in range(0, max_skip + 2):
            pen = np.float32(mu if delta == 0 else lam * (delta - 1))
            off = sh + delta
            # j' = j + off   ->   dp[j] va a best[j+off]
            lo_j = max(0, -off)
            hi_j = min(B, B - off)
            if lo_j >= hi_j:
                continue
            src = dp[lo_j:hi_j] + pen
            dst = best[lo_j + off:hi_j + off]
            better = src < dst
            if better.any():
                idx = np.nonzero(better)[0]
                dst[idx] = src[idx]
                arg[lo_j + off + idx] = (lo_j + idx).astype(np.int16)
        dp = best + cost[f]
        bp[f] = arg

    j = int(dp.argmin())                      # fin libre
    v_of_f = np.empty(ndv, np.int64)
    cost_of_f = np.empty(ndv, np.float32)
    for f in range(ndv - 1, -1, -1):
        v_of_f[f] = base[f] + j
        cost_of_f[f] = cost[f, j]
        if f:
            j = int(bp[f, j])
            if j < 0:
                raise RuntimeError(f"camino roto en el frame {f}")
    return v_of_f, cost_of_f, margin


# --------------------------------------------------------------------------
# Parches: de un frame de la referencia a la luma de un macrobloque
# --------------------------------------------------------------------------

def photometric(dv, ref, ok=None, trim=0.10):
    """Ajusta  dv ~= ganancia*ref + offset  sobre las celdas sanas.

    Robusto: descarta el 'trim' de residuos mas extremos y reajusta. Las
    celdas rotas del DV llevan la estimacion de la camara, que puede estar muy
    lejos, y un ajuste por minimos cuadrados crudo se iria detras de ellas.
    """
    m = np.isfinite(ref) & np.isfinite(dv)
    if ok is not None:
        m &= ok
    x, y = ref[m], dv[m]
    if x.size < 400:
        return None
    g = np.polyfit(x, y, 1)
    res = np.abs(np.polyval(g, x) - y)
    keep = res <= np.quantile(res, 1.0 - trim)
    if keep.sum() >= 400:
        g = np.polyfit(x[keep], y[keep], 1)
    return float(g[0]), float(g[1])


class RefPatches:
    """Lo que ve la ocultacion: ref(i, k) -> 16x16 de luma, o None.

    No sabe de VOB ni de ffmpeg. Se le da el mapa de frames ya resuelto y una
    forma de conseguir la luma del DV, y devuelve parches ya puestos en la
    geometria y los niveles del DV.
    """

    GAIN_LO, GAIN_HI = 0.80, 1.20
    OFF_LO, OFF_HI = -30.0, 30.0

    def __init__(self, store, v_of_i, prof, table, luma=None, ok_of=None,
                 covered=None, fm=None, limit=8):
        self.store = store
        self.v = np.asarray(v_of_i, np.int64)
        self.prof = prof
        self.luma = luma                       # luma(i) -> (h, w) float, rango DV
        self.ok_of = ok_of                     # ok_of(i) -> (mb_rows, mb_cols) bool
        self.covered = covered
        self.fm = fm or store.fields
        self.rows = np.asarray(table) // prof.mb_cols
        self.cols = np.asarray(table) % prof.mb_cols
        self._cache = {}
        self._limit = limit
        self.fits = {}

    def bind(self, luma, ok_of):
        """La ocultacion presta su cache de luma y su mapa de sanos.

        Asi no se decodifican dos veces los mismos frames, que es el coste
        dominante de todo esto.
        """
        self.luma = luma
        self.ok_of = ok_of
        self._cache.clear()

    def frame(self, i):
        """El frame de referencia de i, en geometria y niveles del DV."""
        if i in self._cache:
            return self._cache[i]
        v = int(self.v[i])
        if v < 0 or v >= self.store.n or (
                self.covered is not None and not self.covered[i]):
            img = None
        else:
            img = self.store.frame_dv(v, self.fm)
            dv = self.luma(i)
            ok = None
            if self.ok_of is not None:
                ok = np.repeat(np.repeat(self.ok_of(i), 16, 0), 16, 1)
            fit = photometric(dv, img, ok)
            if fit is None or not (self.GAIN_LO <= fit[0] <= self.GAIN_HI
                                   and self.OFF_LO <= fit[1] <= self.OFF_HI):
                img = None
            else:
                img = img * fit[0] + fit[1]
                self.fits[i] = fit
        if len(self._cache) >= self._limit:
            self._cache.pop(next(iter(self._cache)))
        self._cache[i] = img
        return img

    def __call__(self, i, k):
        img = self.frame(i)
        if img is None:
            return None
        r, c = int(self.rows[k]), int(self.cols[k])
        if not self.store.covers(r, c):
            return None
        p = img[16 * r:16 * r + 16, 16 * c:16 * c + 16]
        return None if not np.isfinite(p).all() else p

    WINDOWS = (1, 2, 4, 8)
    MIN_PIX = 4 * 256

    def local_rmse(self, i, k, ok_grid):
        """Desacuerdo con el DV SANO que rodea al macrobloque.

        Es la puerta de confianza. Donde la referencia trae rayas de la
        conexion analogica, o esta desalineada, o sencillamente no tiene este
        material, discrepa de la cinta sana de alrededor y se ve aqui, por
        macrobloque y usando solo dato real.

        La ventana crece por escalones por la misma razon que la de la puerta
        de movimiento: el dano viene en manchas, y con radio 1 la mitad de los
        macrobloques rotos no tienen NI UN vecino sano con el que medir
        (medido: 49% en el tramo mas danado). Sin escalonado, la puerta no
        rechaza: sencillamente no opina, y se traga o descarta a ciegas.
        """
        img = self.frame(i)
        if img is None:
            return np.inf
        dv = self.luma(i)
        r, c = int(self.rows[k]), int(self.cols[k])
        for win in self.WINDOWS:
            num = den = 0.0
            for rr in range(r - win, r + win + 1):
                for cc in range(c - win, c + win + 1):
                    if (rr == r and cc == c) or not self.store.covers(rr, cc):
                        continue
                    if not ok_grid[rr, cc]:
                        continue
                    a = dv[16 * rr:16 * rr + 16, 16 * cc:16 * cc + 16]
                    b = img[16 * rr:16 * rr + 16, 16 * cc:16 * cc + 16]
                    if not np.isfinite(b).all():
                        continue
                    num += float(((a - b) ** 2).sum())
                    den += a.size
            if den >= self.MIN_PIX:
                return float(np.sqrt(num / den))
        return np.inf if den < 256 else float(np.sqrt(num / den))


class RefPlan:
    """El mapa global de alineacion, listo para repartir por ventanas.

    Se calcula una vez con `refalign` sobre la captura base y se guarda; aqui
    solo se lee y se traduce a la escala de cada ventana.
    """

    def __init__(self, work_dir, align_path=None):
        self.store = RefStore(work_dir)
        p = align_path or os.path.join(work_dir, "refvideo", "align.json")
        d = json.load(open(p))
        self.map = np.array(d["map"], np.int64)
        self.covered = np.array(d["covered"], bool)
        self.base_n = int(d["base"]["n"])
        self.base_name = d["base"]["name"]

    def for_window(self, idx, base_ci, lo, prof, table, drop=()):
        """RefPatches para una ventana ya indexada.

        idx        indice de frames de cinta de la ventana
        base_ci    cual de las capturas del indice es el recorte de la base
        lo         frame de la base al que corresponde el frame 0 del recorte
        drop       posiciones de cinta descartadas por estar fuera de sitio

        Los frames de cinta que solo tienen lecturas de donantes no tienen
        numero de base propio; se rellenan interpolando entre los que si lo
        tienen, que es exacto mientras la referencia tenga los frames de en
        medio (los dos van a 25 fps). Donde no los tenga, lo caza la puerta de
        confianza, que mira macrobloque a macrobloque.
        """
        ks, vs, cs = [], [], []
        for k in range(len(idx)):
            mine = [f for c, f in idx.reads(k) if c == base_ci]
            if not mine:
                continue
            g = lo + int(mine[0])
            if 0 <= g < len(self.map):
                ks.append(k)
                vs.append(int(self.map[g]))
                cs.append(bool(self.covered[g]))
        if len(ks) < 2:
            return None
        all_k = np.arange(len(idx))
        v_of_k = np.rint(np.interp(all_k, ks, vs)).astype(np.int64)
        # cobertura: solo donde la hay a los dos lados, que es lo prudente
        cov = np.interp(all_k, ks, np.array(cs, float)) >= 0.999

        drop = set(int(x) for x in drop)
        if drop:
            keep = [k for k in range(len(idx)) if k not in drop]
            v_of_k = v_of_k[keep]
            cov = cov[keep]
        return RefPatches(self.store, v_of_k, prof, table, covered=cov)
