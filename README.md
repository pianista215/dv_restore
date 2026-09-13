# dv_restore — restauración de MiniDV a partir de varias capturas

Herramientas para combinar varias pasadas de captura de la misma cinta MiniDV
deteriorada y quedarse con lo mejor de cada una.

El punto de partida: dos lecturas correctas del mismo frame de cinta dan
bloques de vídeo byte a byte idénticos. Eso permite emparejar frames entre
capturas sin depender del timecode (inválido en estas cintas) ni del abst.

## Estado

- `dvr/layout.py` — geometría del frame DV (bloques, segmentos, audio).
- `dvr/dcplane.py` — lectura vectorizada del DC, modo DCT, clase, STA y QNO
  sin decodificar el bitstream.
- `dvr/bitstream.py` — parser y repaquetizador de segmento de vídeo.
  Verificado con round-trip byte a byte sobre 19 542 segmentos.
- `dvr/render.py` — decodificación a imagen con ffmpeg.

Los scripts `dv_*.py` de la raíz son la versión anterior, de una sola pareja de
ficheros y a nivel de bloque de 80 bytes. Se conservan como referencia.
