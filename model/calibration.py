"""
Calibration of the Lake Constance whitefish model (3 stocks: zooplankton, young and adult whitefish).

This script contains a Python implementation of exactly the same equations as the Vensim model
(whitefish_model.mdl, Euler integration, time step 0.0625 years). It estimates nine parameters by
weighted least squares against three datasets at the same time:
  - whitefish catch 1950-2023      (IGKB, read from chart; 2023 from Frei & Kugler, 2026)
  - whitefish survey index 1997-2022 (IGKB / IBKF)
  - crustacean zooplankton 1961-2025 (IGKB / BOWIS), in 1000 individuals per m2
The resulting values are printed and written to calibrated_parameters.json; they were then
entered into the Vensim model.

For transperancy: This script was created with help of the AI claude
"""
import json
import numpy as np
from scipy.optimize import least_squares

# ----------------------------------------------------------------------------------- data
# Observed data (calibration targets) and model inputs (phosphorus, deep-water oxygen).
# Phosphorus 1950 = 7 ugP/L and oxygen 1950 = 8 mgO2/L are assumptions (before the first measurements).
CATCH = {1950: 263, 1951: 411, 1952: 337, 1953: 232, 1954: 189, 1955: 432, 1956: 884, 1957: 747, 1958: 505, 1959: 716, 1960: 726, 1961: 474, 1962: 368, 1963: 147, 1964: 126, 1965: 379, 1966: 137, 1967: 126, 1968: 242, 1969: 379, 1970: 495, 1971: 347, 1972: 116, 1973: 579, 1974: 274, 1975: 200, 1976: 863, 1977: 1211, 1978: 453, 1979: 442, 1980: 295, 1981: 379, 1982: 526, 1983: 495, 1984: 526, 1985: 695, 1986: 358, 1987: 168, 1988: 463, 1989: 768, 1990: 905, 1991: 1000, 1992: 811, 1993: 621, 1994: 326, 1995: 547, 1996: 779, 1997: 958, 1998: 958, 1999: 853, 2000: 674, 2001: 611, 2002: 726, 2003: 726, 2004: 800, 2005: 632, 2006: 442, 2007: 411, 2008: 537, 2009: 547, 2010: 600, 2011: 621, 2012: 337, 2013: 284, 2014: 295, 2015: 147, 2016: 200, 2017: 179, 2018: 116, 2019: 53, 2020: 179, 2021: 95, 2022: 21, 2023: 9.8}
P_OBS = {1950: 7, 1960: 15, 1961: 19, 1962: 24, 1963: 30, 1964: 34, 1965: 36, 1966: 35, 1967: 45.1, 1968: 43.7, 1969: 47, 1970: 52.5, 1971: 52.6, 1972: 69.6, 1973: 76.5, 1974: 78.9, 1975: 80.1, 1976: 83.6, 1977: 83.9, 1978: 77.8, 1979: 81.4, 1980: 78.7, 1981: 78.5, 1982: 73.8, 1983: 70.9, 1984: 64.7, 1985: 62.1, 1986: 58.7, 1987: 52.4, 1988: 42.1, 1989: 40.1, 1990: 36.3, 1991: 32.5, 1992: 29.9, 1993: 26.2, 1994: 24.8, 1995: 22.7, 1996: 19, 1997: 16.7, 1998: 15.4, 1999: 14, 2000: 12.8, 2001: 11.6, 2002: 10.7, 2003: 10, 2004: 8.9, 2005: 8.5, 2006: 7.6, 2007: 7.5, 2008: 7.6, 2009: 6.4, 2010: 5.8, 2011: 5.9, 2012: 6.3, 2013: 6.7, 2014: 6.9, 2015: 7.6, 2016: 8, 2017: 7.5, 2018: 7.6, 2019: 6.2, 2020: 6.3, 2021: 6, 2022: 6.2, 2023: 5.4, 2024: 6.1, 2025: 5.3}
O2_OBS = {1950: 8, 1963: 5.2, 1964: 3.5, 1965: 5.1, 1966: 4.8, 1967: 6.1, 1968: 5.6, 1969: 5.8, 1970: 7, 1971: 2.4, 1972: 3.4, 1973: 2.3, 1974: 4.8, 1975: 4.6, 1976: 3.4, 1977: 6.5, 1978: 6.4, 1979: 6.4, 1980: 5.3, 1981: 7, 1982: 5.8, 1983: 6.3, 1984: 6.2, 1985: 5.6, 1986: 5.6, 1987: 6.36, 1988: 4.6, 1989: 4.3, 1990: 4.3, 1991: 5, 1992: 6.3, 1993: 5, 1994: 5.9, 1995: 6.12, 1996: 6.3, 1997: 6.5, 1998: 6.1, 1999: 7, 2000: 7.2, 2001: 6.1, 2002: 6.8, 2003: 6.7, 2004: 7.2, 2005: 7.5, 2006: 7.3, 2007: 7, 2008: 6.6, 2009: 6.6, 2010: 7.9, 2011: 6.3, 2012: 7.2, 2013: 6.2, 2014: 6.5, 2015: 5.9, 2016: 5.2, 2017: 6.1, 2018: 6.4, 2019: 7.5, 2020: 7, 2021: 6.9, 2022: 6.2, 2023: 6.8, 2024: 6.5, 2025: 6}
SURVEY = {1997: 40.66, 1998: 24.29, 1999: 32.86, 2000: 27.11, 2001: 22.09, 2002: 15.58, 2003: 17.7, 2004: 23.13, 2005: 23.19, 2006: 17.71, 2007: 5.21, 2008: 7.06, 2009: 11.29, 2010: 11.72, 2011: 25.52, 2012: 4.34, 2013: 2.07, 2014: 3.49, 2015: 1.52, 2016: 2.11, 2017: 4.42, 2018: 6.82, 2019: 0.43, 2020: 8.51, 2021: 0.42, 2022: 0.19}
ZOO = {1961: 607.893, 1962: 915.8, 1963: 1129.5, 1964: 1074.1, 1965: 821.4, 1966: 1325.1, 1967: 835.4, 1968: 986.8, 1969: 1184.5, 1970: 1470.2, 1971: 778, 1972: 859.8, 1973: 1418, 1974: 1152.11, 1975: 864.5, 1976: 1077.1, 1977: 799.1, 1978: 1129.9, 1979: 551.08, 1980: 885.43, 1981: 1125.05, 1982: 1004.3, 1983: 782, 1984: 873.9, 1985: 754.36, 1986: 907.5, 1987: 629.819, 1988: 932.6, 1989: 917.976, 1990: 995.625, 1991: 1051.25, 1992: 825.317, 1993: 719.284, 1994: 694.464, 1996: 294.058, 1997: 353.985, 1998: 169.656, 1999: 353.945, 2000: 193.013, 2001: 630.581, 2002: 604.81, 2003: 494.407, 2004: 211.443, 2005: 187.557, 2006: 217.555, 2007: 199.061, 2008: 328.815, 2009: 258.681, 2010: 230.509, 2011: 231.258, 2012: 283.287, 2013: 314.665, 2014: 310.793, 2015: 247.703, 2016: 281.155, 2017: 300.725, 2018: 450.999, 2019: 277.512, 2020: 420.152, 2021: 418.637, 2022: 428.761, 2023: 385.017, 2024: 304.334, 2025: 299.6}

P_PTS, O2_PTS = sorted(P_OBS.items()), sorted(O2_OBS.items())


def lookup(pts, t):
    """Linear interpolation between data points, value held constant outside the data range (as in Vensim)."""
    xs, ys = zip(*pts)
    return float(np.interp(t, xs, ys))


# ----------------------------------------------------------------------------------- fixed parameters
# Parameters that are NOT calibrated (literature values, assumptions and scenario settings).
FIXED = dict(
    maturation_time=2.5,        # Year
    young_mortality=0.3,        # 1/Year
    adult_mortality=0.25,       # 1/Year
    fishing_rate=0.5,           # 1/Year
    zoo_mortality=2.0,          # 1/Year
    o2_half=2.0, o2_ref=6.0,    # mgO2/L
    zoo_ref=300.0,              # 1000 ind./m2
    sb_start=2012.0, sb_decline_year=2024.0, sb_decline=0.8, scenario_year=2027.0, sb_after_2027=0.2,
    ban_start=2024.0, ban_end=2027.33, fishing_ban=1, catch_from_2027=25.0,
)

# ----------------------------------------------------------------------------------- model
def simulate(p, t0=1950, t1=2025, dt=0.0625):
    """Euler integration of the three ODEs. Returns yearly values of the stocks and the catch."""
    q = dict(FIXED); q.update(p)
    food = lambda Z: (Z / (Z + q["food_half"])) / (q["zoo_ref"] / (q["zoo_ref"] + q["food_half"]))
    oxy = lambda O: (O / (O + q["o2_half"])) / (q["o2_ref"] / (q["o2_ref"] + q["o2_half"]))
    zgrowth = lambda P: q["zoo_production"] * P / (P + q["zoo_p_half"])

    # start values: adults calibrated, zooplankton and young whitefish in equilibrium with 1950 conditions
    A = q["initial_adults"]
    Z = zgrowth(P_PTS[0][1]) / (q["zoo_mortality"] + q["whitefish_grazing"] * A)
    Y = A * q["max_recruitment"] * food(Z) * oxy(O2_PTS[0][1]) / (1 / q["maturation_time"] + q["young_mortality"])

    out = {}
    for i in range(int(round((t1 - t0) / dt)) + 1):
        t = t0 + i * dt
        P, O = lookup(P_PTS, t), lookup(O2_PTS, t)
        # stickleback pressure index
        if t < q["sb_start"]: S = 0.0
        elif t < q["sb_decline_year"]: S = 1.0
        elif t < q["scenario_year"]: S = 1 - q["sb_decline"]
        else: S = q["sb_after_2027"]
        # flows
        z_growth = zgrowth(P)
        z_mort = q["zoo_mortality"] * Z
        graz_wf = q["whitefish_grazing"] * A * Z
        graz_sb = q["stickleback_grazing"] * S * Z
        recruitment = A * q["max_recruitment"] * food(Z) * oxy(O) * (1 - q["stickleback_egg_impact"] * S)
        maturation = Y / q["maturation_time"]
        young_mort = q["young_mortality"] * Y
        nat_death = q["adult_mortality"] * A
        if t < q["ban_start"]:
            catch = q["fishing_rate"] * A
        elif t < q["ban_end"]:
            catch = (1 - q["fishing_ban"]) * q["fishing_rate"] * A
        else:
            catch = min(q["catch_from_2027"], q["fishing_rate"] * A)
        if abs(t - round(t)) < 1e-9:
            out[int(round(t))] = dict(zoo=Z, young=Y, adults=A, catch=catch)
        # Euler step
        Z += dt * (z_growth - z_mort - graz_wf - graz_sb)
        Y += dt * (recruitment - maturation - young_mort)
        A += dt * (maturation - nat_death - catch)
    return out


# ----------------------------------------------------------------------------------- calibration
NAMES = ["max_recruitment", "zoo_production", "zoo_p_half", "whitefish_grazing",
         "stickleback_grazing", "food_half", "stickleback_egg_impact", "initial_adults"]
LOWER = [0.05, 50, 0.5, 1e-6, 0, 10, 0, 50]
UPPER = [20, 1e5, 300, 0.05, 10, 1e5, 1, 2e4]
STARTS = [[1.5, 3000, 20, 5e-4, .5, 300, .5, 700], [3, 1500, 50, 2e-4, 1, 100, .3, 400],
          [1, 6000, 10, 1e-3, .2, 800, .7, 1200], [2, 2500, 80, 1e-4, 2, 200, .2, 900]]


def survey_catchability(sim):
    """Factor that best scales the simulated adult stock to the survey index (closed-form least squares)."""
    A = np.array([sim[y]["adults"] for y in SURVEY]); s = np.array(list(SURVEY.values()))
    return (A @ s) / (A @ A)


def residuals(x):
    """Weighted differences model - data. Each dataset is scaled to its typical size;
    survey (x0.7) and zooplankton (x0.5) get a slightly lower weight than the catch."""
    sim = simulate(dict(zip(NAMES, x)))
    r_catch = [(sim[y]["catch"] - CATCH[y]) / 300 for y in sorted(CATCH)]
    q = survey_catchability(sim)
    r_survey = [(q * sim[y]["adults"] - SURVEY[y]) / 15 * 0.7 for y in SURVEY]
    r_zoo = [(sim[y]["zoo"] - ZOO[y]) / 300 * 0.5 for y in ZOO]
    return np.array(r_catch + r_survey + r_zoo)


if __name__ == "__main__":
    # several start points to avoid a local minimum; the best result is kept
    fits = [least_squares(residuals, x0, bounds=(LOWER, UPPER), x_scale="jac", max_nfev=4000) for x0 in STARTS]
    best = min(fits, key=lambda r: r.cost)
    params = dict(zip(NAMES, best.x))
    sim = simulate(params)
    params["survey_catchability"] = survey_catchability(sim)

    print("Calibrated parameters")
    for k, v in params.items():
        print(f"  {k:24s} {v:.6g}")
    corr = lambda a, b: np.corrcoef(a, b)[0, 1]
    print("Goodness of fit (Pearson r)")
    print("  catch 1950-2023   ", round(corr([sim[y]["catch"] for y in sorted(CATCH)], [CATCH[y] for y in sorted(CATCH)]), 2))
    print("  survey 1997-2022  ", round(corr([sim[y]["adults"] for y in SURVEY], list(SURVEY.values())), 2))
    print("  zooplankton       ", round(corr([sim[y]["zoo"] for y in ZOO], list(ZOO.values())), 2))
    json.dump(params, open("calibrated_parameters.json", "w"), indent=1)
