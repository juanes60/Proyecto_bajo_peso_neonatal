# Riesgo de bajo peso al nacer (< 2.500 g) — regresión logística

Modelo de apoyo para priorizar seguimiento durante la gestación. **No es un diagnóstico.**
Proyecto de Machine Learning I · Universidad Externado de Colombia · Pregrado en Ciencia de Datos.

## Archivos
| archivo | qué es |
|---|---|
| `modelo.joblib` | el pipeline entrenado (imputación → one-hot → escalado → selección bivariable → regresión logística) |
| `transformadores.py` | clases del pipeline; `joblib` las necesita para cargar `modelo.joblib` |
| `app.py` | la aplicación (Streamlit) |
| `requirements.txt` | las librerías con su versión exacta |
| `README.md` | este archivo |

## Cómo correrla
```bash
pip install -r requirements.txt
streamlit run app.py
```
Python 3.13.18. Todos los archivos deben estar en la misma carpeta.
Para usar el modelo sin la app:
```python
import joblib
modelo = joblib.load("modelo.joblib")           # requiere transformadores.py en la carpeta
p = modelo.predict_proba(df)[:, 1]              # df con las columnas de la tabla de abajo
alerta = p >= 0.14                      # umbral de decisión
```

## Qué columnas espera
El pipeline recibe un `DataFrame` con estas 18 columnas (en cualquier orden). El modelo solo usa las marcadas con «sí»;
las demás deben estar presentes pero pueden ir en `NaN`.

| columna | tipo | valores / unidades | la usa el modelo |
|---|---|---|---|
| `SEXO` | categórica | texto, uno de: MASCULINO, FEMENINO | sí |
| `TIEMPO DE GESTACION` | numérica | 37 a 41 (semanas completas) | sí |
| `NUMERO CONSULTAS PRENATALES` | numérica | 0 a 19 (número de consultas) | sí |
| `TIPO PARTO` | categórica | texto, uno de: Espontáneo, Cesárea, Instrumentado | sí |
| `GRUPO SANGUINEO` | categórica | (no la usa el modelo; si falta se imputa) | no |
| `FACTOR RH` | categórica | (no la usa el modelo; si falta se imputa) | no |
| `PERTENENCIA ETNICA` | categórica | (no la usa el modelo; si falta se imputa) | no |
| `EDAD MADRE` | numérica | 13 a 45 (años) | sí |
| `ESTADO CONYUGAL MADRE` | categórica | texto, uno de: Union libre, Soltera, Casada | sí |
| `NIVEL EDUCATIVO MADRE` | categórica | texto, uno de: Media, Secundaria, Superior, Primaria, Ninguno | sí |
| `DEPARTAMENTO RESIDENCIA` | categórica | (no la usa el modelo; si falta se imputa) | no |
| `AREA RESIDENCIA` | categórica | texto, uno de: Cabecera municipal, Rural disperso, Centro poblado | sí |
| `NUMERO HIJOS NACIDOS VIVOS` | numérica | 1 a 9 (número de hijos) | sí |
| `NUMERO EMBARAZOS` | numérica | 1 a 9 (número de embarazos) | sí |
| `REGIMEN SEGURIDAD` | categórica | (no la usa el modelo; si falta se imputa) | no |
| `EDAD PADRE` | numérica | 15 a 64 (años) | sí |
| `NIVEL EDUCATIVO PADRE` | categórica | (no la usa el modelo; si falta se imputa) | no |
| `depresion` | numérica | (no la usa el modelo; si falta se imputa) | no |

Los textos deben escribirse exactamente como aparecen arriba (mayúsculas y tildes incluidas). Una categoría que el modelo no conoce
se trata como «Otras» o como la categoría de referencia.

## Con qué datos se entrenó
- Fuente: «Nacimientos ocurridos en el Hospital General de Medellín» (Datos Abiertos Colombia, descarga del 18-ago-2026), 20.873 registros iniciales (2022–2026).
- Se excluyeron partos múltiples y nacimientos antes de la semana 37. Quedaron 17.092 nacimientos (bajo peso: 8,66 %).
- Partición estratificada (semilla 73): 11.964 entrenamiento · 2.564 validación · 2.564 prueba.
- Variable objetivo: peso al nacer < 2.500 g. Se retiraron talla, APGAR y localidad (fuga de datos / demasiadas categorías).
- La depresión perinatal es una variable **simulada** a partir de PPD_DATA; no superó el filtro bivariable y el modelo no la usa.

## Modelo
- Regresión logística con regularización L2, C = 0,1059. Sin ponderar clases: el desbalance se maneja con el umbral.
- **Umbral de decisión: 0,14** (elegido por validación cruzada en entrenamiento, maximizando F2). No es 0,5.
- Variables seleccionadas por el filtro bivariable (11): TIEMPO DE GESTACION, NUMERO CONSULTAS PRENATALES, EDAD MADRE, NUMERO HIJOS NACIDOS VIVOS, NUMERO EMBARAZOS, EDAD PADRE, SEXO, TIPO PARTO, ESTADO CONYUGAL MADRE, NIVEL EDUCATIVO MADRE, AREA RESIDENCIA.

| conjunto | AUC-ROC | AUC-PR | recall | precisión | F2 | % de alarmas |
|---|---|---|---|---|---|---|
| validación | 0,846 | 0,320 | 0,698 | 0,277 | 0,536 | 21,8 % |
| prueba | 0,844 | 0,359 | 0,743 | 0,297 | 0,572 | 21,6 % |

## Cuándo se entrenó
18-11-2026 · scikit-learn 1.6.1 · pandas 2.2.3.

## Limitaciones
- Datos de un solo hospital de tercer nivel; no se debe generalizar a toda Colombia.
- No hay mediciones de MUAC, estrés, hemoglobina ni consejería nutricional.
- Solo nacimientos de 37 semanas o más: el modelo no sirve para estimar riesgo en prematuros.
- Resultados de asociación, no de causalidad. En prueba, cerca de 3 de cada 10 alertas son correctas y ~26 % de los casos de bajo peso no se detecta.
