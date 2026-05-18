import numpy as np

def Smit2021(weight, height, creatinine):
    # Bedside Schwartz for CrCl (capped at 120 mL/min)
    CrCl = min(0.413 * height / creatinine, 120.0)

    priors = np.array([
        8.9 * weight / 22.1,  # Vc
        12.3 * weight / 22.1,  # Vp
        2.12 * np.power(weight / 22.1, 0.745) * CrCl / 100,  # CL
        1.55 * np.power(weight / 22.1, 0.599)  # Q
    ])

    omega_Vp = np.log(1.1 ** 2 + 1)
    omega_CL = np.log(0.287 ** 2 + 1)

    cov = np.array([
        [0.00, 0.00, 0.00, 0.00],
        [0.00, omega_Vp, -0.085, 0.00],
        [0.00, -0.085, omega_CL, 0.00],
        [0.00, 0.00, 0.00, 0.00]
    ])

    iiv = [False, True, True, False]

    # Error model: Proportional CV
    error_config = {"type": "proportional", "sigma": 0.0789}

    return priors, cov, iiv, error_config

def Lamarre2000(weight, height, age, creatinine):
    # Schwartz CrCl
    if age < 1.0:
        k = 0.45
    if age >= 1.0:
        k = 0.55

    BSA = (4 * weight + 7) / (weight + 90)
    CrCl = k * height / creatinine * BSA / 1.73 * 0.06 # CrCl in L/Hr

    priors = np.array([
        0.27 * weight,                # Vc
        0.16 * weight,                # Vp
        0.46 * CrCl + 0.018 * weight,  # CL
        0.16 * weight                 # Q
    ])

    omega_Vc = np.log(0.42 ** 2 + 1)
    omega_Vp = np.log(0.43 ** 2 + 1)
    omega_CL = np.log(0.45 ** 2 + 1)
    omega_Q  = np.log(0.43 ** 2 + 1)

    cov = np.array([
        [omega_Vc, 0, 0, 0],
        [0, omega_Vp, 0, 0],
        [0, 0, omega_CL, 0],
        [0, 0, 0, omega_Q]
    ])

    iiv = [True, True, True, True]

    error_config = {"type": "fixed", "sigma": np.log(0.065 ** 2 + 1)}