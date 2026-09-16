"""
stationarity.py
================
Kombinovany ADF + KPSS postup pre stacionarizaciu casovych radov.

FINALNA LOGIKA PRE TENTO PROJEKT:
    - Premenna sa testuje v urovniach pomocou ADF a KPSS.
    - Ak nie je stacionarna, aplikuje sa VZDY LEN PRVA diferencia.
    - Druha diferencia sa iba TESTUJE diagnosticky ako upozornenie;
      NIKDY sa neaplikuje do finalneho datasetu.
    - Finalny dataset preto vzdy odstrani PRESNE JEDNO pociatocne
      pozorovanie (za predpokladu, ze aspon jedna premenna dostala
      prvu diferenciu).

To chrani pred overdifferencingom a zachovava jednoduchu, rovnaku
transformaciu I(1) premennych pri ARIMAX, Elastic Net a XGBoost.
"""

import warnings
from typing import Literal
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tools.sm_exceptions import InterpolationWarning


# ----------------------------------------------------------------------
# 1. Kombinovany ADF + KPSS test
# ----------------------------------------------------------------------

def _run_tests(y, alpha=0.05, kpss_regression: Literal["c", "ct"]="c"):
    """Vrati ADF a KPSS statistiky/p-hodnoty bez KPSS warningov v konzole."""
    y = pd.Series(y).dropna()
    if len(y) < 20:
        raise ValueError(f"Na test stacionarity treba aspon 20 pozorovani, mam {len(y)}.")

    adf_stat, adf_pval, *_ = adfuller(y, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_stat, kpss_pval, *_ = kpss(y, regression=kpss_regression, nlags="auto")

    is_stationary = (adf_pval < alpha) and (kpss_pval >= alpha)
    return adf_stat, adf_pval, kpss_stat, kpss_pval, is_stationary


def check_stationarity(df, alpha=0.05, kpss_regression: Literal["c", "ct"]="c"):
    """
    Spusti ADF aj KPSS pre kazdy numericky stlpec.

    Pravidlo:
        stacionarna <=> ADF p < alpha a KPSS p >= alpha.

    Poznamka: vysledok je diagnosticky. Pri konflikte ADF/KPSS treba
    pozriet ACF, graf rady a ekonomicku logiku; KPSS je citlivy na
    strukturalne zlomy a silnu perzistenciu.
    """
    rows = []
    for col in df.select_dtypes(include="number").columns:
        adf_stat, adf_pval, kpss_stat, kpss_pval, is_stationary = _run_tests(
            df[col], alpha=alpha, kpss_regression=kpss_regression
        )

        if is_stationary:
            verdict = "stacionarna (zhoda ADF+KPSS)"
        elif adf_pval >= alpha and kpss_pval < alpha:
            verdict = "nestacionarna (zhoda ADF+KPSS)"
        else:
            verdict = "nejasna / konflikt ADF+KPSS"

        rows.append({
            "premenna": col,
            "adf_stat": adf_stat,
            "adf_pval": adf_pval,
            "kpss_stat": kpss_stat,
            "kpss_pval": kpss_pval,
            "is_stationary": is_stationary,
            "verdict": verdict,
        })

    return pd.DataFrame(rows).set_index("premenna")


# ----------------------------------------------------------------------
# 2. Rozhodnutie: maximalne prva diferencia + diagnostika druhej
# ----------------------------------------------------------------------

def determine_diff_orders(df, alpha=0.05, overdiff_acf_threshold=-0.4,
                          verbose=True):
    """
    Urci rad transformacie pre finalny dataset.

    FINALNE rozhodnutie je vzdy iba 0 alebo 1:
        0 = premenna ostane v urovniach
        1 = premenna dostane prvu diferenciu

    Pre premennu, ktora ani po 1. diferencii nesplna prisne ADF+KPSS
    pravidlo, sa vypocita aj 2. diferencia IBA DIAGNOSTICKY. Vysledok
    sa vytlaci ako upozornenie; do vystupneho slovnika sa nikdy
    nezapise hodnota 2. Tym sa zabrani tomu, aby build_combined_dataset
    odstranil prve dve pozorovania.

    Pri konflikte ADF/KPSS po 1. diferencii kod vrati 1 a upozorni,
    aby si sa pozrel na ACF/graf/ekonomicku interpretaciu. Je to vhodne
    pre tvoje makro data, kde vyssie diferencie casto znamenaju
    overdifferencing.
    """
    final_orders = {}

    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()

        # Test v urovniach
        _, adf_p0, _, kpss_p0, is_stat_0 = _run_tests(series, alpha=alpha)

        if is_stat_0:
            final_orders[col] = 0
            if verbose:
                print(f"  {col}: I(0) - stacionarna v urovniach, bez diferencie.")
            continue

        # Test po PRVEJ diferencii
        y_diff1 = series.diff().dropna()
        _, adf_p1, _, kpss_p1, is_stat_1 = _run_tests(y_diff1, alpha=alpha)
        acf1_diff1 = acf(y_diff1, nlags=1, fft=True)[1]

        # Finalny rozhodovaci krok: maximalne prva diferencia.
        final_orders[col] = 1

        if is_stat_1:
            if verbose:
                print(f"  {col}: I(1) - 1. diferencia je stacionarna "
                      f"(ADF p={adf_p1:.4f}, KPSS p={kpss_p1:.4f}, ACF(1)={acf1_diff1:.3f}).")
        else:
            # 2. diferencia sa iba diagnosticky kontroluje, NIE pre finalne data.
            y_diff2 = y_diff1.diff().dropna()
            _, adf_p2, _, kpss_p2, is_stat_2 = _run_tests(y_diff2, alpha=alpha)
            acf1_diff2 = acf(y_diff2, nlags=1, fft=True)[1]

            msg = (f"  UPOZORNENIE - {col}: po 1. diferencii konflikt/nesplnenie "
                   f"ADF+KPSS (ADF p={adf_p1:.4f}, KPSS p={kpss_p1:.4f}, "
                   f"ACF(1)={acf1_diff1:.3f}). "
                   f"2. diferencia je iba diagnosticka: ADF p={adf_p2:.4f}, "
                   f"KPSS p={kpss_p2:.4f}, ACF(1)={acf1_diff2:.3f}. "
                   f"Finalne ostava PRVA diferencia (rad 1).")
            if acf1_diff2 < overdiff_acf_threshold:
                msg += " ACF(1) po 2. diferencii je silne zaporne -> znak overdifferencingu."
            if verbose:
                print(msg)

    return final_orders


# ----------------------------------------------------------------------
# 3. Finalny dataset - maximalne jedna diferencia, teda max. 1 riadok prec
# ----------------------------------------------------------------------

def build_combined_dataset(df, diff_orders, manual_diff_order=None):
    """
    Vytvori finalny dataset pre modelovanie.

    Garantuje maximalne jednu diferenciu:
        - 0: premenna zostane v urovniach
        - 1: premenna dostane prvu diferenciu

    Funkcia zamerne NEDOVOLI rad 2 alebo vyssi. Druha diferencia v tejto
    metodike sluzi len ako diagnosticky test, nie ako transformacia.

    Ak existuje aspon jedna premenna s radom 1, odstrani presne prvy
    riadok, pretoze `Series.diff()` pre prve pozorovanie nema hodnotu.
    Stacionarne premenne sa tym iba casovo zarovnaju s diferencovanymi.
    """
    if not isinstance(diff_orders, dict):
        raise TypeError(
            "diff_orders musi byt dict, napriklad vystup z determine_diff_orders(df)."
        )

    manual_diff_order = manual_diff_order or {}
    final_orders = {**diff_orders, **manual_diff_order}

    missing = set(df.columns) - set(final_orders)
    if missing:
        raise ValueError(f"Chybaju rady diferencií pre stlpce: {sorted(missing)}")

    invalid_orders = {col: order for col, order in final_orders.items() if order not in (0, 1)}
    if invalid_orders:
        raise ValueError(
            "V tejto verzii su povolene len rady 0 a 1. "
            f"Najdene nepovolene hodnoty: {invalid_orders}. "
            "Druha diferencia je iba diagnosticka a nema vstupovat do df_final."
        )

    pieces = {}
    for col in df.columns:
        pieces[col] = df[col].diff() if final_orders[col] == 1 else df[col].copy()

    df_final = pd.DataFrame(pieces)

    # Ak bola pouzita aspon jedna prva diferencia, prvy riadok obsahuje NaN.
    # Odstranime presne JEDNO pozorovanie, nikdy nie dve.
    if any(order == 1 for order in final_orders.values()):
        df_final = df_final.iloc[1:]

    # Toto odstrani iba povodne chybajuce hodnoty (ak existuju), nie dalsie
    # riadky vytvorene nasou diferenciou.
    if df_final.isna().any().any():
        n_before = len(df_final)
        df_final = df_final.dropna()
        n_dropped = n_before - len(df_final)
        if n_dropped:
            print(f"Pozor: odstranilo sa dalsich {n_dropped} riadkov kvoli NaN, "
                  "ktore uz boli v povodnych datach.")

    return df_final