### Conceptos Clave de Granulometría

- En minería, la granulometría es la medición de la distribución de tamaños de las partículas (rocas).

- D20, D50, D80: Son los "puntos de corte" de la Curva Granulométrica Acumulada. Por ejemplo, el D80 es el tamaño de partícula tal que el 80% del material (en masa o volumen) es más pequeño que ese tamaño. Es el indicador más crítico para ajustar las máquinas chancadoras.

- Pasante Acumulado: El porcentaje de material que "pasa" por una malla de cierto tamaño.

- Excentricidad/Deformidad: Mide qué tan "alargada" es una roca. Una roca muy excéntrica puede pasar por una malla de una forma pero no de otra, lo que afecta la eficiencia.

### Run develop framework

To run the develop framework, simply compile Develop Docker File (Based on ubuntu 22) and then you can run it using the following command:

```
docker run --rm -it --gpus all -v "C:\Users\ignac\Escritorio\Delyrium\Lithos_Analytics_Challenge\images:/home/images" ubuntu_22_cuda
```

* It is addapted to Windows Power Shell, using other environments might be a little different.