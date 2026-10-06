# Datos del curso

Todos los datos del curso están en esta carpeta. **No tiene que bajarlos a mano**: cada notebook los descarga solo en la primera celda (y, si no puede, la celda *Plan B* le pide subir el archivo). Esta página sirve para mirarlos por su cuenta, abrirlos en Excel o usarlos en otro proyecto.

| Archivo | Sesiones | Tamaño | Qué es | Fuente y licencia |
|---|---|---|---|---|
| [`incidentes_medellin_muestra_5000.csv`](incidentes_medellin_muestra_5000.csv) | Notebook 0 | 5 000 × 13 | Muestra al azar del archivo siguiente | ídem |
| [`incidentes_medellin_2019_2025.csv`](incidentes_medellin_2019_2025.csv) | S1–S3 | 222 923 × 13 | Incidentes viales en Medellín, 2019–2025: fecha, hora, clase, gravedad, dirección, comuna, barrio y coordenadas | Secretaría de Movilidad de Medellín; Sánchez Corredor, Arango Uribe y Correa Álvarez, *Data* (MDPI, 2026); Mendeley Data, DOI 10.17632/r6g5dfnpgh.1 · CC BY 4.0 |
| [`insurance.csv`](insurance.csv) | S4 | 1 338 × 7 | Costo anual de seguro médico según edad, sexo, IMC, hijos, fumador y región (EE. UU.) | Lantz, *Machine Learning with R*; copia de los datasets de PyCaret |
| [`saber11_valle_2022.csv.gz`](saber11_valle_2022.csv.gz) | S4 | 78 720 × 25 | Resultados Saber 11 del Valle del Cauca (calendario A, 2022): colegio, hogar y puntajes. Comprimido: pandas lo lee directo | ICFES, *Resultados únicos Saber 11*, datos.gov.co (`kgxf-xxbe`) · datos abiertos |
| [`bank.csv`](bank.csv) | S5, S6, S10–S12 | 45 211 × 17 | Campañas telefónicas de un banco portugués: datos del cliente y si abrió un depósito a término (`deposit`) | Moro, Cortez y Rita (2014), UCI Machine Learning Repository · CC BY 4.0 |
| [`penguins.csv`](penguins.csv) | S7, S8 | 344 × 7 | Medidas de pingüinos de tres especies en la Antártida | Gorman, Williams y Fraser (2014); paquete `palmerpenguins` · CC0 |
| [`municipios_ipm_2018.csv`](municipios_ipm_2018.csv) | S9 | 1 122 × 25 | Índice de Pobreza Multidimensional (IPM) y sus 15 privaciones por municipio, con departamento y coordenadas | DANE, IPM municipal del Censo 2018 + DIVIPOLA (datos.gov.co `gdxc-w37w`) |

La Sesión 8 usa además los dígitos escritos a mano que vienen incluidos en scikit-learn (`load_digits`, 1 797 imágenes de 8 × 8), y S7 genera datos de laboratorio con `make_blobs` y `make_moons`.

**Cómo mirarlos:**
- **En GitHub:** haga clic en el nombre del archivo. Los pequeños se ven como tabla; los grandes (Medellín, Saber 11) GitHub no los muestra: use *Download raw file* (ícono de descarga, arriba a la derecha).
- **En Colab, fuera de las sesiones:**
  ```python
  import pandas as pd
  df = pd.read_csv("https://raw.githubusercontent.com/Nitobest/curso-aad-python/main/datos/bank.csv")
  df.head()
  ```
- **En Excel:** descargue el CSV y ábralo con *Datos → Desde texto/CSV*, codificación UTF-8 (si no, las tildes se ven mal).

Si usa alguno de estos datos en su proyecto o en otro trabajo, cite la fuente de la tabla.
