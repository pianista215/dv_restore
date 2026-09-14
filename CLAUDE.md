# CLAUDE.md

Guía para trabajar en este repositorio.

## Qué es

Herramientas para restaurar cintas MiniDV deterioradas combinando varias
pasadas de captura del mismo material. No es un reproductor ni un
transcodificador: opera sobre el flujo DV crudo, a nivel de macrobloque.

## Reglas que no se saltan

- **Los `.dv` de origen son sagrados.** Se abren siempre con
  `np.memmap(..., mode="r")`. Nunca se escriben, se mueven ni se borran.
- **Nada de material de vídeo al repositorio.** El `.gitignore` excluye `*.dv`,
  `*.mp4`, `*.png`, y los directorios `clips/`, `work/`, `out/`. Si añades una
  salida nueva, añádela también al `.gitignore`.
- **Todo segmento que se escriba tiene que volver a parsear.** Es la invariante
  que evita las franjas negras. Si al mezclar no caben los coeficientes, se
  recortan las frecuencias altas conservando el EOB (`dvr/fixup.py`); nunca se
  deja que el empaquetador pierda el final.
- **Medir antes de afirmar.** Cada decisión de este repo salió de una medición,
  y varias hipótesis razonables resultaron falsas al medirlas. Si cambias un
  umbral, enseña el número que lo justifica.

## Arquitectura, en una frase cada pieza

- `dvr/layout.py` — geometría del frame DV. La unidad de trabajo es el
  **segmento de vídeo** (5 macrobloques, 400 bytes), no el bloque DIF de 80:
  los 5 comparten el desbordamiento VLC y tocar uno rompe los otros cuatro.
- `dvr/bitstream.py` — parser y repaquetizador de segmento. Versión de
  referencia en Python, verificada con round-trip byte a byte.
- `dvr/_native/dvbits.c` + `dvr/native.py` — lo mismo en C, 85× más rápido. Se
  compila solo con gcc la primera vez y cachea el `.so` en `work/`.
- `dvr/dcplane.py` — DC, modo DCT, clase, STA y QNO de forma vectorizada, sin
  decodificar. El DC está en posición fija, así que sale gratis.
- `dvr/shuffle.py` — bloque del flujo → macrobloque de la pantalla, deducido
  del propio decodificador.
- `dvr/match.py` — emparejado N-way e índice de frames de cinta.
- `dvr/merge.py` — fusión por macrobloque. `dvr/conceal.py` — ocultación.
- `dvr/pipeline.py` — proceso completo por ventanas, resumible.

## Cosas del material que conviene saber

- **La cámara ya oculta los errores** antes de sacar el vídeo, y marca lo
  ocultado con `STA = 0xE`. Un bloque marcado no es basura: es la estimación de
  la cámara. Lo que se mide no es "cuántos bloques están rotos" sino **cuántos
  llevan dato real y cuántos una estimación**.
- **El daño tiene periodo 5.** Las sondas de emparejado se eligen por sorteo
  fijo, nunca a paso constante: un paso múltiplo de 5 hace que todas caigan en
  una posición siempre rota.
- **Los ID DIF difieren entre capturas** aunque el contenido sea idéntico. Al
  copiar hay que reponerlos.
- **Una captura no puede leer dos veces el mismo frame de cinta.** El union-find
  rechaza esas uniones: aceptarlas crea ciclos en el grafo de orden y un ciclo
  temprano desordena todo lo que va detrás.

## Cosas que parecían buena idea y NO funcionan

Medidas y descartadas. No las reintentes sin leer esto.

- **Elegir el origen de la copia por la estructura del ECC.** El daño de estas
  cintas golpea siempre la misma posición dentro del segmento de vídeo, y el
  barajado convierte eso en una banda de 9 columnas que cruza la pantalla: m=3
  son las columnas 0-8, m=1 las 9-17, m=0 las 18-26, m=2 las 27-35 y m=4 las
  36-44. Cuando la posición que falla deriva, la banda nace en el centro y se va
  a la derecha hasta salirse. Explica perfectamente el artefacto, pero preferir
  orígenes con esa banda fría **empeora**: error mediano 2,2 -> 3,0 donde cambia
  la elección, peor en el 52% de los casos. Los orígenes estructuralmente
  limpios están más lejos en el tiempo y eso cuesta más de lo que gana.
  Desempatar solo entre candidatos a igual distancia es todavía peor (media 4,6
  -> 9,3). **La cercanía temporal domina.**

- **Estabilizar el lienzo entre frames consecutivos.** El lienzo saltaba de
  captura en el 31% de las transiciones y la diferencia media era mayor cuando
  cambiaba (11,3 frente a 7,5). Parecía causal y no lo era: bajarlo al 8% no
  movió ni un salto. El lienzo cambia donde hay más daño, y hay más daño donde
  la imagen se mueve. Se dejó puesto porque mantiene coherentes el subcódigo y
  el audio, no porque arregle nada.

- **Votar por mayoría entre lecturas sanas que discrepan.** Con ocho pasadas
  solo discrepan 391 bloques de 792 087 (0,049%), todos con mayoría clara: 0,3
  bloques por frame. Invisible. La basura que queda es idéntica en las ocho
  pasadas, o sea que está escrita en la cinta.

## Pruebas

```bash
python3 -m unittest discover -s tests -v
```

Son rápidas y cubren las invariantes: tabla VLC, round-trip del parser, que el
C dé lo mismo que el Python, el recorte por falta de sitio, y el orden
(huecos, ciclos, cadenas sin enlazar, uniones de la misma captura).

Antes de dar por bueno un cambio en el orden o la fusión, además de las
pruebas: saca una tira de frames consecutivos y **míralos**. Los números
globales tapan los desórdenes locales; las dos veces que este repo estuvo mal
lo detectó el ojo, no la métrica.
