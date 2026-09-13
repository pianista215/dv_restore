/* dvbits.c - parser y repaquetizador de segmentos de video DV.
 *
 * Misma logica que dvr/bitstream.py, que es la version de referencia y la que
 * se verifica con el round-trip byte a byte. Esta existe solo por velocidad:
 * el Python tarda ~1,9 ms por segmento, que son 0,6 s por frame.
 *
 * Se compila solo, sin dependencias, desde dvr/native.py.
 */

#include <stdint.h>
#include <string.h>

#define SEG_MB    5
#define NDCT      6
#define SEG_BYTES 400
#define SEG_BITS  3200
#define HDR_BITS  12
#define MAXTOK    72
#define MAXFRAG   48

static const int AREA_OFF[NDCT]  = { 4, 18, 32, 46, 60, 70 };
static const int AREA_BITS[NDCT] = { 112, 112, 112, 112, 80, 80 };

/* tabla VLC, rellenada desde Python en la carga */
static uint16_t VLC_CODE[1024];
static uint8_t  VLC_LEN[1024];
static int16_t  VLC_RUN[1024];
static int16_t  VLC_LEVEL[1024];
static uint16_t LUT[1 << 16];
static int NB_VLC = 0;

void dv_set_tables(const uint16_t *code, const uint8_t *len,
                   const int16_t *run, const int16_t *level, int n)
{
    NB_VLC = n;
    memcpy(VLC_CODE, code, n * sizeof(uint16_t));
    memcpy(VLC_LEN, len, n * sizeof(uint8_t));
    memcpy(VLC_RUN, run, n * sizeof(int16_t));
    memcpy(VLC_LEVEL, level, n * sizeof(int16_t));
    for (int i = 0; i < n; i++) {
        int l = len[i];
        int c = (int) code[i] << (16 - l);
        for (int k = 0; k < (1 << (16 - l)); k++)
            LUT[c + k] = (uint16_t) i;
    }
}

/* ---- lector sobre una lista de trozos de bits del propio segmento ---- */
typedef struct {
    const uint8_t *buf;
    int off[MAXFRAG], len[MAXFRAG], n;
    int fi, bi;
} Rd;

static void rd_init(Rd *r, const uint8_t *buf) { r->buf = buf; r->n = r->fi = r->bi = 0; }

static void rd_add(Rd *r, int off, int len)
{
    if (len > 0 && r->n < MAXFRAG) { r->off[r->n] = off; r->len[r->n] = len; r->n++; }
}

static int rd_bit(Rd *r)
{
    while (r->fi < r->n && r->bi >= r->len[r->fi]) { r->fi++; r->bi = 0; }
    if (r->fi >= r->n) return -1;
    int p = r->off[r->fi] + r->bi;
    r->bi++;
    return (r->buf[p >> 3] >> (7 - (p & 7))) & 1;
}

static int rd_avail(const Rd *r)
{
    int s = 0;
    for (int i = r->fi; i < r->n; i++) s += (i == r->fi ? r->len[i] - r->bi : r->len[i]);
    return s;
}

static uint32_t rd_read(Rd *r, int k)
{
    uint32_t v = 0;
    for (int i = 0; i < k; i++) { int b = rd_bit(r); v = (v << 1) | (b < 0 ? 0 : (uint32_t) b); }
    return v;
}

static uint32_t rd_peek16(Rd *r)
{
    int fi = r->fi, bi = r->bi;
    uint32_t v = 0;
    for (int i = 0; i < 16; i++) { int b = rd_bit(r); v = (v << 1) | (b < 0 ? 0 : (uint32_t) b); }
    r->fi = fi; r->bi = bi;
    return v;
}

/* lo que queda sin leer, como lista de trozos: es el deposito para el
 * siguiente bloque DCT */
static void rd_rest(const Rd *r, Rd *out)
{
    out->buf = r->buf; out->n = out->fi = out->bi = 0;
    for (int i = r->fi; i < r->n; i++) {
        if (i == r->fi) rd_add(out, r->off[i] + r->bi, r->len[i] - r->bi);
        else            rd_add(out, r->off[i], r->len[i]);
    }
}

static void rd_concat(Rd *dst, const Rd *a, const Rd *b)
{
    dst->buf = a->buf ? a->buf : b->buf;
    dst->n = dst->fi = dst->bi = 0;
    for (int i = a->fi; i < a->n; i++)
        rd_add(dst, a->off[i] + (i == a->fi ? a->bi : 0), a->len[i] - (i == a->fi ? a->bi : 0));
    for (int i = b->fi; i < b->n; i++)
        rd_add(dst, b->off[i] + (i == b->fi ? b->bi : 0), b->len[i] - (i == b->fi ? b->bi : 0));
}

/* ---- estructuras de salida ---- */
typedef struct {
    int16_t dc;
    uint8_t mode, cls;
    int16_t pos;
    uint8_t done;
    int16_t ntok;
    /* palabras decodificadas dentro del area fija del propio bloque. Si el
     * segmento resulta invalido, solo estas son de fiar: lo que venga del
     * desbordamiento puede ser basura de otro macrobloque. */
    int16_t ntok_area;
    uint16_t tok[MAXTOK];
} DvBlk;

typedef struct {
    DvBlk b[NDCT];
    uint8_t qno, sta, id[3];
} DvMB;

typedef struct {
    DvMB mb[SEG_MB];
    uint8_t raw[SEG_BYTES];
    int32_t ok;          /* 1 = todos los bloques DCT terminan */
    int32_t bad_mb;      /* primer macrobloque que falla, -1 si ninguno */
    int32_t bad_blk;
} DvSeg;

static int decode_into(Rd *r, DvBlk *blk)
{
    for (;;) {
        int avail = rd_avail(r);
        if (avail <= 0) return 0;
        int t = LUT[rd_peek16(r)];
        int l = VLC_LEN[t];
        if (l > avail) return 0;
        rd_read(r, l);
        if (blk->ntok < MAXTOK) blk->tok[blk->ntok++] = (uint16_t) t;
        else return 0;
        blk->pos += VLC_RUN[t] + 1;
        if (blk->pos >= 64) { blk->done = 1; return 1; }
    }
}

int dv_parse_segment(const uint8_t *buf, DvSeg *seg)
{
    memset(seg, 0, sizeof(*seg));
    memcpy(seg->raw, buf, SEG_BYTES);
    seg->bad_mb = seg->bad_blk = -1;
    const uint8_t *b = seg->raw;

    Rd mb_free[SEG_MB], pend[SEG_MB][NDCT];
    for (int m = 0; m < SEG_MB; m++) {
        rd_init(&mb_free[m], b);
        for (int j = 0; j < NDCT; j++) rd_init(&pend[m][j], b);
    }

    /* pasada 1: el area fija de cada bloque DCT */
    for (int m = 0; m < SEG_MB; m++) {
        int base = m * 80 * 8;
        seg->mb[m].id[0] = b[m * 80]; seg->mb[m].id[1] = b[m * 80 + 1];
        seg->mb[m].id[2] = b[m * 80 + 2];
        seg->mb[m].sta = b[m * 80 + 3] >> 4;
        seg->mb[m].qno = b[m * 80 + 3] & 0x0F;
        for (int j = 0; j < NDCT; j++) {
            Rd r; rd_init(&r, b);
            rd_add(&r, base + AREA_OFF[j] * 8, AREA_BITS[j]);
            DvBlk *blk = &seg->mb[m].b[j];
            int dc = (int) rd_read(&r, 9);
            blk->dc = (int16_t) (dc >= 256 ? dc - 512 : dc);
            blk->mode = (uint8_t) rd_read(&r, 1);
            blk->cls = (uint8_t) rd_read(&r, 2);
            decode_into(&r, blk);
            blk->ntok_area = blk->ntok;
            if (blk->done) {
                Rd rest; rd_rest(&r, &rest);
                /* el area que sobra pasa al deposito del macrobloque */
                for (int i = 0; i < rest.n; i++) rd_add(&mb_free[m], rest.off[i], rest.len[i]);
            } else {
                rd_rest(&r, &pend[m][j]);
            }
        }
    }

    /* pasada 2: deposito del macrobloque */
    Rd vs_free; rd_init(&vs_free, b);
    for (int m = 0; m < SEG_MB; m++) {
        Rd pool = mb_free[m];
        int all_done = 1;
        for (int j = 0; j < NDCT; j++) {
            DvBlk *blk = &seg->mb[m].b[j];
            if (blk->done) continue;
            Rd r; rd_concat(&r, &pend[m][j], &pool);
            decode_into(&r, blk);
            if (!blk->done) { rd_rest(&r, &pend[m][j]); all_done = 0; break; }
            rd_init(&pend[m][j], b);
            rd_rest(&r, &pool);
        }
        if (all_done)
            for (int i = pool.fi; i < pool.n; i++)
                rd_add(&vs_free, pool.off[i] + (i == pool.fi ? pool.bi : 0),
                       pool.len[i] - (i == pool.fi ? pool.bi : 0));
    }

    /* pasada 3: deposito del segmento */
    Rd pool = vs_free;
    for (int m = 0; m < SEG_MB; m++) {
        for (int j = 0; j < NDCT; j++) {
            DvBlk *blk = &seg->mb[m].b[j];
            if (blk->done) continue;
            Rd r; rd_concat(&r, &pend[m][j], &pool);
            decode_into(&r, blk);
            if (!blk->done) {
                seg->ok = 0; seg->bad_mb = m; seg->bad_blk = j;
                return 0;
            }
            rd_init(&pend[m][j], b);
            rd_rest(&r, &pool);
        }
    }
    seg->ok = 1;
    return 1;
}

/* ---- escritura ---- */
typedef struct { int off, len; } Hole;

typedef struct {
    uint8_t *out;
    uint8_t *touched;
} Wr;

static void put_bits(Wr *w, int at, uint32_t v, int k)
{
    for (int i = 0; i < k; i++) {
        int p = at + i, bit = (v >> (k - 1 - i)) & 1;
        if (bit) w->out[p >> 3] |= (uint8_t) (1 << (7 - (p & 7)));
        else     w->out[p >> 3] &= (uint8_t) ~(1 << (7 - (p & 7)));
        w->touched[p] = 1;
    }
}

/* Escribe las palabras pendientes en la lista de huecos. Si se agotan a mitad
 * de una palabra, deja apuntado el TROZO QUE FALTA: los bits ya escritos se
 * quedan donde estan y el siguiente deposito continua desde ahi. */
static int drain(Wr *w, Hole *holes, int *phi, int nh,
                 uint32_t *tv, uint8_t *tl, int *pti, int nt)
{
    int hi = *phi, ti = *pti;
    while (ti < nt) {
        uint32_t v = tv[ti];
        int need = tl[ti];
        while (need && hi < nh) {
            int take = need < holes[hi].len ? need : holes[hi].len;
            put_bits(w, holes[hi].off, (v >> (need - take)) & ((1u << take) - 1), take);
            need -= take;
            if (take == holes[hi].len) hi++;
            else { holes[hi].off += take; holes[hi].len -= take; }
        }
        if (need) {
            tv[ti] = v & ((1u << need) - 1);
            tl[ti] = (uint8_t) need;
            *phi = hi; *pti = ti;
            return 1;               /* quedan sobras */
        }
        ti++;
    }
    *phi = hi; *pti = ti;
    return 0;
}

int dv_pack_segment(const DvSeg *seg, const uint8_t *pad_src, uint8_t *out)
{
    uint8_t touched[SEG_BITS];
    memset(touched, 0, sizeof(touched));
    memcpy(out, pad_src, SEG_BYTES);
    Wr w = { out, touched };

    uint32_t tv[SEG_MB][NDCT][MAXTOK + 1];
    uint8_t  tl[SEG_MB][NDCT][MAXTOK + 1];
    int      tn[SEG_MB][NDCT], ti[SEG_MB][NDCT];
    Hole holes[SEG_MB][MAXFRAG]; int nh[SEG_MB];
    Hole own[SEG_MB][NDCT]; int has_own[SEG_MB][NDCT];
    int lost = 0;

    for (int m = 0; m < SEG_MB; m++) {
        nh[m] = 0;
        int base = m * 80 * 8;
        for (int j = 0; j < NDCT; j++) {
            const DvBlk *blk = &seg->mb[m].b[j];
            int n = 0;
            tv[m][j][n] = (uint32_t) (((blk->dc & 0x1FF) << 3) |
                                      ((blk->mode & 1) << 2) | (blk->cls & 3));
            tl[m][j][n++] = HDR_BITS;
            for (int k = 0; k < blk->ntok; k++) {
                tv[m][j][n] = VLC_CODE[blk->tok[k]];
                tl[m][j][n++] = VLC_LEN[blk->tok[k]];
            }
            tn[m][j] = n; ti[m][j] = 0; has_own[m][j] = 0;

            int a = base + AREA_OFF[j] * 8, cap = AREA_BITS[j], used = 0, k = 0;
            while (k < n && used + tl[m][j][k] <= cap) {
                put_bits(&w, a + used, tv[m][j][k], tl[m][j][k]);
                used += tl[m][j][k]; k++;
            }
            ti[m][j] = k;
            if (k == n) {
                if (cap - used) { holes[m][nh[m]].off = a + used; holes[m][nh[m]].len = cap - used; nh[m]++; }
            } else if (cap - used) {
                own[m][j].off = a + used; own[m][j].len = cap - used; has_own[m][j] = 1;
            }
        }
    }

    Hole vs[SEG_MB * MAXFRAG]; int nvs = 0;
    for (int m = 0; m < SEG_MB; m++) {
        Hole pool[MAXFRAG + 1]; int np = 0, hi = 0;
        for (int i = 0; i < nh[m]; i++) pool[np++] = holes[m][i];
        int all_done = 1;
        for (int j = 0; j < NDCT; j++) {
            if (ti[m][j] >= tn[m][j]) continue;
            Hole mine[MAXFRAG + 2]; int nm = 0;
            if (has_own[m][j]) { mine[nm++] = own[m][j]; has_own[m][j] = 0; }
            for (int i = hi; i < np; i++) mine[nm++] = pool[i];
            int mhi = 0;
            int rest = drain(&w, mine, &mhi, nm, tv[m][j], tl[m][j], &ti[m][j], tn[m][j]);
            np = 0; hi = 0;
            for (int i = mhi; i < nm; i++) pool[np++] = mine[i];
            if (rest) { all_done = 0; break; }
        }
        if (all_done) for (int i = hi; i < np; i++) vs[nvs++] = pool[i];
    }

    int vhi = 0;
    for (int m = 0; m < SEG_MB; m++) {
        for (int j = 0; j < NDCT; j++) {
            if (ti[m][j] >= tn[m][j]) continue;
            Hole mine[SEG_MB * MAXFRAG + 2]; int nm = 0;
            if (has_own[m][j]) { mine[nm++] = own[m][j]; has_own[m][j] = 0; }
            for (int i = vhi; i < nvs; i++) mine[nm++] = vs[i];
            int mhi = 0;
            int rest = drain(&w, mine, &mhi, nm, tv[m][j], tl[m][j], &ti[m][j], tn[m][j]);
            nvs = 0; vhi = 0;
            for (int i = mhi; i < nm; i++) vs[nvs++] = mine[i];
            if (rest) { lost += tn[m][j] - ti[m][j]; ti[m][j] = tn[m][j]; }
        }
    }

    for (int m = 0; m < SEG_MB; m++) {
        out[m * 80]     = seg->mb[m].id[0];
        out[m * 80 + 1] = seg->mb[m].id[1];
        out[m * 80 + 2] = seg->mb[m].id[2];
        out[m * 80 + 3] = (uint8_t) (((seg->mb[m].sta & 0x0F) << 4) | (seg->mb[m].qno & 0x0F));
    }
    return lost;
}

/* ---- barrido de un frame entero ---- */
/* Para cada macrobloque saca: validez del bitstream de su segmento, numero de
 * coeficientes AC no nulos y energia AC. Es lo que alimenta la deteccion de
 * bloques falsamente sanos. */
int dv_scan_frame(const uint8_t *frame, const int32_t *seg_off, int n_seg,
                  uint8_t *seg_ok, int16_t *nnz, int32_t *energy,
                  int16_t *maxpos)
{
    DvSeg seg;
    int bad = 0;
    for (int s = 0; s < n_seg; s++) {
        int ok = dv_parse_segment(frame + seg_off[s], &seg);
        seg_ok[s] = (uint8_t) ok;
        if (!ok) bad++;
        for (int m = 0; m < SEG_MB; m++) {
            for (int j = 0; j < NDCT; j++) {
                const DvBlk *b = &seg.mb[m].b[j];
                int idx = (s * SEG_MB + m) * NDCT + j;
                int nz = 0, en = 0, pos = 0, mp = 0;
                for (int k = 0; k < b->ntok; k++) {
                    pos += VLC_RUN[b->tok[k]] + 1;
                    if (pos >= 64) break;
                    int lv = VLC_LEVEL[b->tok[k]];
                    if (lv) { nz++; en += lv < 0 ? -lv : lv; mp = pos; }
                }
                nnz[idx] = (int16_t) nz;
                energy[idx] = en;
                maxpos[idx] = (int16_t) mp;
            }
        }
    }
    return bad;
}

int dv_seg_size(void) { return (int) sizeof(DvSeg); }
