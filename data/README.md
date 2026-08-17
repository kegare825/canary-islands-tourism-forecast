# Datos — turismo en Canarias (ISTAC + INE)

A diferencia del proyecto #1 (Kaggle), aquí las fuentes son **datos abiertos oficiales**,
lo que significa más trabajo de descubrimiento/armonización pero cero solapamiento con
portfolios ajenos. Verificado por búsqueda directa antes de documentarlo — no son URLs
inventadas.

## 1. ISTAC — fuente principal (más directa, descarga inmediata)

El ISTAC publica "cubos estadísticos" descargables directamente en CSV/JSON/XLSX sin
necesidad de credenciales, vía su API de datos abiertos (`datos.canarias.es`).

### Cubo `C00065A_000003` — Tarifa media diaria (ADR), RevPAR, ingresos y empleo
Mensual y anual desde 2009, por islas y municipios de Canarias, desagregado por
categoría de establecimiento. Es el más valioso de los tres porque ya trae ADR y RevPAR
calculados — listos para usar sin cálculo adicional.

```
https://datos.canarias.es/api/estadisticas/statistical-resources/v1.0/datasets/ISTAC/C00065A_000003/1.60.csv
```

(Cambiar `.csv` por `.json`, `.jsonstat`, `.tsv` o `.xlsx` según el formato que se prefiera
trabajar. El número de versión `1.60` puede haber avanzado — si el enlace da error,
buscar "C00065A_000003" en <https://datos.canarias.es> para la versión vigente.)

### Cubo `C00065A_000040` — Pernoctaciones, viajeros y estancia media
Anual desde 2009, por isla y microdestino. Buena serie larga, pero anual — pierde la
estacionalidad mensual, que es la parte más interesante de un forecast turístico.

### Cubo `C00065A_000060` — Pernoctaciones, viajeros y estancia media (mensual)
Mensual, pero solo desde 2023 — demasiado corta todavía para modelar estacionalidad
mes-a-mes con SARIMA (se necesitan mínimo 3-4 ciclos anuales completos). Útil como
serie complementaria/de validación reciente, no como serie principal de entrenamiento.

**Decisión recomendada para el proyecto**: usar `C00065A_000003` (mensual, 2009+) como
serie principal — es la más larga en frecuencia mensual y la que mejor conecta con RevPAR.

## 2. INE — fuente complementaria (contraste nacional)

**Encuesta de Ocupación Hotelera (EOH)**: serie mensual nacional desde finales de los 90,
desagregable por provincia (Las Palmas / Santa Cruz de Tenerife = Canarias). Vía la
API Tempus3:

```
https://servicios.ine.es/wstempus/js/es/DATOS_SERIE/{IdSERIE}
```

El `IdSERIE` no es un número fijo documentado aquí a propósito — hay que navegar
[INEbase → Encuestas de turismo → Encuesta de ocupación hotelera](https://www.ine.es/dynt3/inebase/index.htm?capsel=239&padre=1701),
filtrar por "Viajeros y pernoctaciones por comunidades autónomas y provincias" y
Canarias/sus provincias, y copiar el identificador numérico de la URL de la tabla
resultante. Usar esta fuente solo como contraste secundario (¿Canarias se mueve distinto
de la media nacional?), no como serie principal.

## Esquema armonizado esperado (`data/processed/`)

Las fuentes crudas vienen en formato "cubo" (columnas de dimensión + indicador + valor,
estilo SDMX/JSON-stat) — **inspeccionar el CSV real tras descargarlo** antes de escribir
el parser definitivo en `notebooks/01_eda.ipynb`; los nombres de columna exactos del
export no se han verificado byte a byte, solo la existencia y contenido general del cubo.

El objetivo es transformarlo a este formato largo, que es el que espera `src/data.py`:

| Columna | Tipo | Descripción |
|---|---|---|
| `fecha` | date (YYYY-MM-01) | Mes de referencia |
| `isla` | string | Tenerife, Gran Canaria, Lanzarote, Fuerteventura, La Palma, La Gomera, El Hierro |
| `pernoctaciones` | int | Noches de estancia registradas |
| `viajeros` | int | Viajeros alojados |
| `adr_eur` | float | Tarifa media diaria |
| `revpar_eur` | float | Ingresos por habitación disponible |
| `grado_ocupacion` | float | % de ocupación por plazas |

No todas las columnas van a estar en todas las fuentes — `src/data.py` tolera nulos en
las que no vengan de un cubo dado y las combina por `fecha` + `isla`.
