# dv_restore — restauración de MiniDV combinando varias capturas

Herramientas para combinar varias pasadas de captura de la misma cinta MiniDV
deteriorada y quedarse con lo mejor de cada una.

El punto de partida: **dos lecturas correctas del mismo frame de cinta dan
bloques de vídeo byte a byte idénticos**. Eso permite emparejar frames entre
capturas sin depender del timecode (inválido en estas cintas) ni del `abst`.

## Lo que se ha averiguado sobre este material

Todo lo de aquí abajo está medido sobre los ficheros reales, no supuesto.

**La cámara oculta los errores antes de sacar el vídeo.** Un bloque marcado con
`STA = 0xE` no contiene basura: contiene la estimación que hizo la cámara,
normalmente el mismo bloque del frame anterior. Por eso un frame con 1606 de
1620 macrobloques marcados se ve bien. Lo que hay que medir no es "cuántos
bloques están rotos" sino **cuántos llevan dato real y cuántos una
estimación**.

Comparando contra la otra captura, que sirve de verdad de campo, el error de la
ocultación de la cámara es: mediana 14 unidades de DC, p90 106, y un **23 % de
los bloques se desvía más de 50** (unos 29 niveles de luma). Eso es daño bien
visible: se queda con un fotograma viejo y se pierde el movimiento.

**El daño de esta cinta tiene periodo 5.** Dentro de cada segmento de vídeo
sobreviven casi siempre los macrobloques 1 y 3, y mueren el 0, el 2 y el 4.
Consecuencia práctica: elegir las sondas de emparejamiento a paso constante
múltiplo de 5 hace que todas caigan en una posición siempre rota y no se
obtenga **ni un voto** aunque los dos frames compartan cientos de bloques
idénticos. Las sondas se eligen por sorteo fijo.

**El desbordamiento VLC obliga a trabajar por segmentos.** Los 5 macrobloques
de un segmento comparten el espacio sobrante: sustituir un bloque DIF de 80
bytes puede romper los otros cuatro. Por eso hay un parser y un repaquetizador
de segmento completos.

**Los ID DIF difieren entre capturas** (los 1620, byte 0: 147 frente a 148)
aunque el contenido sea idéntico. Al copiar hay que reponerlos.

## Lo que aporta cada pieza, medido

Sobre un recorte de 10 s de la zona más dañada (296 frames de cinta):

| | |
|---|---|
| Fusión: macrobloques que pasan de estimación a dato real | **18,9 %** |
| Reconstrucción temporal: frames que el base no capturó | **103** (+4,1 s de 12) |
| Fusión: macrobloques sin resolver | 8,15 %, que es **exactamente el mínimo teórico** |
| Ocultación propia frente a la de la cámara | +15,8 % de error medio, solo donde actúa |

La fusión es **óptima**: deja sin resolver exactamente los macrobloques que
están ocultados en las dos capturas, ni uno más.

La ocultación propia, en cambio, aporta poco: donde puede actuar con garantías
(zonas quietas), la cámara ya había hecho lo mismo y bien. Donde la cámara se
equivoca de verdad es en las zonas con movimiento, y ahí copiar del frame
vecino sería igual de erróneo. Está implementada y calibrada, pero **el valor
está en tener más pasadas**, no en inventar.

## Cómo lanzarlo

Requisitos: Python 3 con numpy y Pillow, ffmpeg y gcc. Nada más; el acelerador
en C se compila solo la primera vez.

### El caso normal: restaurar una captura con todas las demás

```bash
# 1. que donantes solapan de verdad, y en que tramo (solo lectura, no escribe)
python3 dvr.py triage MALA.dv OTRAS*.dv --out work/triage.json

# 2. proceso completo. Es resumible: si se corta, relanza el mismo comando
python3 dvr.py runall MALA.dv OTRAS*.dv \
        --out /ruta/con/espacio/trabajo \
        --final /ruta/con/espacio/restaurada.dv

# 3. comparativa en video: el original estirado a la linea temporal real de la
#    cinta (congela donde perdio frames) junto al resultado
python3 dvr.py preview restaurada.dv --index work/w.json --out comparativa.mp4
```

`runall` trocea la base en ventanas de 750 frames con paso 600 y cose por el
centro: una sola ventana con todos los donantes no cabe en memoria ni en
tiempo. Cada ventana deja su tramo en `<out>/cores/`, así que lo ya hecho no se
repite. Con `--budget SEGUNDOS` para limpiamente para poder trocearlo.

### Para iterar y depurar

```bash
# verificar el parser y calibrar el barajado (una sola vez por perfil)
python3 dvr.py calib captura.dv --shuffle

# recortar una ventana con todos sus donantes, para iterar en segundos
python3 dvr.py window MALA.dv OTRAS*.dv --start 1550 --count 750 --out clips

# ver el estado
python3 dvr.py scan clips/*.dv
python3 dvr.py map clips/w_MALA.dv --frames 21   # mapa de daño sobre la imagen

# paso a paso
python3 dvr.py index clips/w_*.dv --out work/w.json
python3 dvr.py merge --index work/w.json --out out/merged.dv
python3 dvr.py calconceal out/merged.dv          # elegir el umbral con datos
python3 dvr.py conceal out/merged.dv --out out/final.dv --threshold 8
```

**Todo lo grande va a `--out`.** Los `.dv` de origen se abren solo lectura y no
se tocan nunca.

## Invariantes que se comprueban

- `parse` + `pack` de un segmento devuelve **los 400 bytes exactos**. Verificado
  sobre 19 542 segmentos de las dos capturas: 100 % de los válidos.
- La tabla de barajado se deduce del propio decodificador con planos de bits
  sobre un frame gris sintético, y se verifica bloque a bloque.
- Todo segmento que se escribe **vuelve a parsear**. Si al mezclar no caben los
  coeficientes, se recortan las frecuencias más altas y se conserva el EOB;
  saltarse esto deja franjas negras en la imagen.

## Estructura

- `dvr/layout.py` — geometría del frame DV.
- `dvr/dcplane.py` — DC, modo DCT, clase, STA y QNO de forma vectorizada, sin
  decodificar el bitstream. El DC está en posición fija, así que sale gratis.
- `dvr/bitstream.py` — parser y repaquetizador de segmento (referencia, Python).
- `dvr/_native/dvbits.c` + `dvr/native.py` — lo mismo en C, 85× más rápido
  (7 ms por frame). Se compila solo; si falla gcc, se usa el Python.
- `dvr/shuffle.py` — bloque del flujo → macrobloque de la pantalla.
- `dvr/match.py` — emparejamiento N-way e índice de frames de cinta.
- `dvr/merge.py` — fusión por segmento y por macrobloque.
- `dvr/conceal.py` — ocultación temporal con puerta de movimiento.
- `dvr/fixup.py` — saneado y recorte para que el segmento siempre sea válido.
- `dvr/maps.py`, `dvr/sheet.py`, `dvr/render.py` — diagnóstico visual.

Los scripts `dv_*.py` de la raíz son la versión anterior, de una sola pareja de
ficheros y a nivel de bloque de 80 bytes. Se conservan como referencia.
