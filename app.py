# app.py -- Apoyo para priorizar seguimiento prenatal: riesgo de bajo peso al nacer (< 2.500 g)
# Uso:  streamlit run app.py
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RUTA_MODELO = Path(__file__).parent / "modelo.joblib"
UMBRAL = 0.14            # umbral de decisión fijado en entrenamiento (no es 0,5)
COLUMNAS = ['SEXO', 'TIEMPO DE GESTACION', 'NUMERO CONSULTAS PRENATALES', 'TIPO PARTO', 'GRUPO SANGUINEO', 'FACTOR RH', 'PERTENENCIA ETNICA', 'EDAD MADRE', 'ESTADO CONYUGAL MADRE', 'NIVEL EDUCATIVO MADRE', 'DEPARTAMENTO RESIDENCIA', 'AREA RESIDENCIA', 'NUMERO HIJOS NACIDOS VIVOS', 'NUMERO EMBARAZOS', 'REGIMEN SEGURIDAD', 'EDAD PADRE', 'NIVEL EDUCATIVO PADRE', 'depresion']        # columnas que espera el pipeline, en este orden
CATEGORICAS = ['SEXO', 'TIPO PARTO', 'GRUPO SANGUINEO', 'FACTOR RH', 'PERTENENCIA ETNICA', 'ESTADO CONYUGAL MADRE', 'NIVEL EDUCATIVO MADRE', 'DEPARTAMENTO RESIDENCIA', 'AREA RESIDENCIA', 'REGIMEN SEGURIDAD', 'NIVEL EDUCATIVO PADRE']
ESPECIFICACION = {'TIEMPO DE GESTACION': {'tipo': 'num', 'min': 37.0, 'max': 41.0, 'valor': 39.0, 'entero': True}, 'NUMERO CONSULTAS PRENATALES': {'tipo': 'num', 'min': 0.0, 'max': 19.0, 'valor': 7.0, 'entero': True}, 'EDAD MADRE': {'tipo': 'num', 'min': 13.0, 'max': 45.0, 'valor': 24.0, 'entero': True}, 'NUMERO HIJOS NACIDOS VIVOS': {'tipo': 'num', 'min': 1.0, 'max': 9.0, 'valor': 2.0, 'entero': True}, 'NUMERO EMBARAZOS': {'tipo': 'num', 'min': 1.0, 'max': 9.0, 'valor': 2.0, 'entero': True}, 'EDAD PADRE': {'tipo': 'num', 'min': 15.0, 'max': 64.0, 'valor': 28.0, 'entero': True}, 'SEXO': {'tipo': 'cat', 'opciones': ['MASCULINO', 'FEMENINO']}, 'TIPO PARTO': {'tipo': 'cat', 'opciones': ['Espontáneo', 'Cesárea', 'Instrumentado']}, 'ESTADO CONYUGAL MADRE': {'tipo': 'cat', 'opciones': ['Union libre', 'Soltera', 'Casada']}, 'NIVEL EDUCATIVO MADRE': {'tipo': 'cat', 'opciones': ['Media', 'Secundaria', 'Superior', 'Primaria', 'Ninguno']}, 'AREA RESIDENCIA': {'tipo': 'cat', 'opciones': ['Cabecera municipal', 'Rural disperso', 'Centro poblado']}}     # variables que usa el modelo y sus valores permitidos
UNIDADES = {'TIEMPO DE GESTACION': 'semanas completas', 'NUMERO CONSULTAS PRENATALES': 'número de consultas', 'EDAD MADRE': 'años', 'EDAD PADRE': 'años', 'NUMERO HIJOS NACIDOS VIVOS': 'número de hijos', 'NUMERO EMBARAZOS': 'número de embarazos'}


@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


def construir_fila(valores):
    # Una fila con TODAS las columnas. Las que no se piden al usuario (el modelo no las usa)
    # quedan en NaN y el imputador del pipeline las rellena con lo aprendido en entrenamiento.
    fila = {}
    for c in COLUMNAS:
        v = valores.get(c, np.nan)
        fila[c] = pd.Series([v], dtype=object) if c in CATEGORICAS else pd.Series([v], dtype=float)
    return pd.DataFrame(fila, columns=COLUMNAS)


def predecir(valores):
    p = float(cargar_modelo().predict_proba(construir_fila(valores))[0, 1])
    return p, p >= UMBRAL


def main():
    st.set_page_config(page_title="Riesgo de bajo peso al nacer", page_icon="👶")
    st.title("Riesgo de bajo peso al nacer")
    st.caption("Herramienta de apoyo para priorizar seguimiento prenatal. No es un diagnóstico.")

    valores = {}
    izq, der = st.columns(2)
    for i, (var, spec) in enumerate(ESPECIFICACION.items()):
        etiqueta = var.capitalize() + (f" ({UNIDADES[var]})" if var in UNIDADES else "")
        with (izq if i % 2 == 0 else der):
            if spec["tipo"] == "num":
                if spec["entero"]:
                    valores[var] = st.number_input(etiqueta, min_value=int(spec["min"]), max_value=int(spec["max"]),
                                                   value=int(spec["valor"]), step=1)
                else:
                    valores[var] = st.number_input(etiqueta, min_value=spec["min"], max_value=spec["max"],
                                                   value=spec["valor"], step=1.0)
            else:
                valores[var] = st.selectbox(etiqueta, spec["opciones"])

    if st.button("Calcular riesgo", type="primary"):
        p, alerta = predecir(valores)
        st.metric("Probabilidad estimada de bajo peso", f"{p:.1%}")
        if alerta:
            st.warning(f"ALERTA: la probabilidad supera el umbral de decisión ({UMBRAL:.2f}). "
                       "Conviene priorizar el seguimiento de esta gestación.")
        else:
            st.success(f"Por debajo del umbral de decisión ({UMBRAL:.2f}). No se genera alerta.")
        st.caption("El umbral es bajo a propósito: se prefiere detectar más casos de riesgo aunque haya falsas alarmas. "
                   "El modelo muestra asociaciones, no causalidad, y se entrenó con datos de un solo hospital.")


if __name__ == "__main__":
    main()
