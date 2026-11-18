# transformadores.py -- clases del pipeline de preprocesamiento (necesarias para cargar modelo.joblib)
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLS_MODA = ["GRUPO SANGUINEO", "FACTOR RH", "ESTADO CONYUGAL MADRE", "NIVEL EDUCATIVO MADRE"]


class ImputadorProyecto(BaseEstimator, TransformerMixin):
    # Moda para las columnas indicadas; edad y nivel educativo del padre por grupo
    # (madre con pareja / sin pareja). Todo se aprende SOLO con el train.
    def __init__(self, cols_moda=(), col_pareja="ESTADO CONYUGAL MADRE",
                 col_edad_padre="EDAD PADRE", col_edu_padre="NIVEL EDUCATIVO PADRE"):
        self.cols_moda = cols_moda; self.col_pareja = col_pareja
        self.col_edad_padre = col_edad_padre; self.col_edu_padre = col_edu_padre

    def _grupo(self, X):
        return np.where(X[self.col_pareja] == "Soltera", "sin_pareja", "con_pareja")

    def _moda(self, s):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else np.nan

    def fit(self, X, y=None):
        self.modas_ = {c: self._moda(X[c]) for c in self.cols_moda}
        Xc = X.fillna(self.modas_)
        g = pd.Series(self._grupo(Xc), index=Xc.index)
        self.med_edad_ = Xc[self.col_edad_padre].groupby(g).median().to_dict()
        self.med_edad_global_ = Xc[self.col_edad_padre].median()
        self.moda_edu_ = Xc[self.col_edu_padre].groupby(g).agg(self._moda).to_dict()
        self.moda_edu_global_ = self._moda(Xc[self.col_edu_padre])
        self.respaldo_ = {c: (X[c].median() if X[c].dtype.kind in "biuf" else self._moda(X[c]))
                          for c in X.columns}
        return self

    def transform(self, X):
        Xc = X.copy().fillna(self.modas_)
        g = pd.Series(self._grupo(Xc), index=Xc.index)
        e, d = self.col_edad_padre, self.col_edu_padre
        Xc[e] = Xc[e].fillna(g.map(self.med_edad_)).fillna(self.med_edad_global_)
        Xc[d] = Xc[d].fillna(g.map(self.moda_edu_)).fillna(self.moda_edu_global_)
        return Xc.fillna(self.respaldo_)


class OneHotReferencia(BaseEstimator, TransformerMixin):
    # One-hot con categoria de referencia (la mas frecuente en train).
    # Las categorias con menos de min_freq (1 %) se agrupan en "Otras".
    def __init__(self, min_freq=0.01):
        self.min_freq = min_freq

    def fit(self, X, y=None):
        self.cols_cat_ = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        self.cols_num_ = [c for c in X.columns if c not in self.cols_cat_]
        self.ref_, self.cats_, self.otras_ = {}, {}, {}
        for c in self.cols_cat_:
            fr = X[c].value_counts(normalize=True)
            comunes = fr[fr >= self.min_freq].index.tolist()
            self.ref_[c] = comunes[0]
            self.cats_[c] = comunes[1:]
            self.otras_[c] = bool((fr < self.min_freq).any())
        return self

    def transform(self, X):
        out = X[self.cols_num_].copy()
        for c in self.cols_cat_:
            s = X[c]
            for k in self.cats_[c]:
                out[f"{c}={k}"] = (s == k).astype(float)
            if self.otras_[c]:
                comunes = [self.ref_[c]] + self.cats_[c]
                out[f"{c}=Otras"] = (~s.isin(comunes)).astype(float)
        return out.astype(float)


class SeleccionBivariada(BaseEstimator, TransformerMixin):
    # Filtro bivariado por VARIABLE ORIGINAL (razon de verosimilitud):
    # todas las dummies de una categorica se conservan o se descartan juntas.
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def fit(self, X, y):
        import statsmodels.api as sm          # solo hace falta para ENTRENAR, no para predecir
        y = np.asarray(y); grupos = {}
        for c in X.columns:
            grupos.setdefault(c.split("=")[0], []).append(c)
        filas = []
        for v, cols in grupos.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = sm.Logit(y, sm.add_constant(X[cols].astype(float))).fit(disp=0)
                p = res.llr_pvalue
            except Exception:
                p = np.nan
            filas.append({"variable": v, "n_columnas": len(cols), "p_valor": p})
        self.resumen_ = pd.DataFrame(filas).sort_values("p_valor").reset_index(drop=True)
        self.resumen_["seleccionada"] = self.resumen_["p_valor"] < self.alpha
        sel = set(self.resumen_.loc[self.resumen_["seleccionada"], "variable"])
        self.variables_ = [v for v in grupos if v in sel]
        self.seleccionadas_ = [c for v in self.variables_ for c in grupos[v]]
        return self

    def transform(self, X):
        return X[self.seleccionadas_]


class EscaladorNumericas(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.cols_ = [c for c in X.columns if X[c].nunique() > 2]
        self.sc_ = StandardScaler().fit(X[self.cols_])
        return self

    def transform(self, X):
        Xc = X.copy()
        Xc[self.cols_] = self.sc_.transform(X[self.cols_])
        return Xc


def armar_pipeline(C=1.0, random_state=73):
    return Pipeline([
        ("imputar",   ImputadorProyecto(cols_moda=COLS_MODA)),
        ("onehot",    OneHotReferencia(min_freq=0.01)),
        ("escalar",   EscaladorNumericas()),
        ("seleccion", SeleccionBivariada(alpha=0.05)),
        ("modelo",    LogisticRegression(C=C, max_iter=2000, random_state=random_state)),
    ])
