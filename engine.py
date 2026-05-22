import numpy as np
import pandas as pd
from scipy.optimize import minimize

import models


class VancomycinBayesEngine:
    """
    Bayesian forecasting engine for vancomycin dosing.
    Supports continuous infusion and intermittent dosing via MAP estimation
    and Monte Carlo simulations for PTA (Probability of Target Attainment).
    """

    def __init__(self, weight_kg, height_cm, age_total_days, creatinine, model="Le2013",
                 target_auc_min=400.0, target_auc_max=600.0, trough_min=8.0, trough_max=12.0, peak=50.0):

        # Patient Demographics & Targets
        self.weight = weight_kg
        self.height = height_cm
        self.creatinine = creatinine
        self.age_total_days = age_total_days
        self.age_years = age_total_days / 365.25
        self.target_auc_min = target_auc_min
        self.target_auc_max = target_auc_max
        self.trough_min = trough_min
        self.trough_max = trough_max
        self.peak = peak

        # State Management
        self.dose_history = []
        self.level_history = []
        self.map_params = None
        self.calibrated = False
        self.posterior_covariance = None

        # Initialize PK model, priors, and error structure
        self.population_mean, self.full_covariance, self.has_iiv, self.error_config = self._initialize_model(model)

        # Pre-invert active covariance for Mahalanobis penalty during MAP estimation
        active_cov = self.full_covariance[np.ix_(self.has_iiv, self.has_iiv)]
        self.inv_active_cov = np.linalg.inv(active_cov)

    def _initialize_model(self, model_name):
        """Standardizes model selection and maps the respective error term definitions."""
        if model_name == 'Smit2021':
            priors, cov, iiv, error_config = models.Smit2021(self.weight, self.height, self.creatinine)
        elif model_name == 'Lamarre2000':
            priors, cov, iiv, error_config = models.Lamarre2000(self.weight, self.height, self.age_years, self.creatinine)
        elif model_name == 'Le2013':
            priors, cov, iiv, error_config = models.Le2013(self.weight, self.height, self.age_total_days, self.creatinine)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        return priors, cov, np.array(iiv, dtype=bool), error_config

    def _get_distribution_params(self):
        """Returns the active log-mean and covariance matrix for Monte Carlo simulations."""
        if not self.calibrated:
            safe_pop_mean = np.maximum(self.population_mean, 1e-10)
            return np.log(safe_pop_mean), self.full_covariance

        safe_map_params = np.maximum(self.map_params, 1e-10)
        active_cov = self.posterior_covariance if self.posterior_covariance is not None else self.full_covariance
        return np.log(safe_map_params), active_cov

    def _calculate_sigma(self, predictions):
        """Calculates the Standard Deviation of error with a safety floor to prevent division by zero."""
        if self.error_config["type"] == "proportional":
            return np.maximum(predictions * self.error_config["sigma"], 0.001)
        if self.error_config["type"] == "fixed":
            return self.error_config["sigma"]

    def _calculate_pk_constants(self, Vc, Vp, CL, Q):
        """
        Calculates micro and macro rate constants.
        Supports both 1-compartment and 2-compartment models.
        Handles scalar floats and vectorized numpy array inputs.
        """
        k10 = CL / Vc

        # 1-Compartment Bypass (e.g., Le2013 where Vp and Q are 0.0)
        if np.all(Vp <= 1e-6) and np.all(Q <= 1e-6):
            k12 = np.zeros_like(Vp) if isinstance(Vp, np.ndarray) else 0.0
            k21 = np.zeros_like(Vp) if isinstance(Vp, np.ndarray) else 0.0

            alpha = k10
            beta = np.zeros_like(Vp) if isinstance(Vp, np.ndarray) else 0.0

            A = 1.0 / Vc
            B = np.zeros_like(Vp) if isinstance(Vp, np.ndarray) else 0.0

            return alpha, beta, A, B, k10, k12, k21

        # 2-Compartment Logic
        k12 = Q / Vc
        k21 = Q / Vp

        S = k10 + k12 + k21
        P = k10 * k21

        # np.maximum prevents negative roots due to floating point inaccuracies
        discriminant = np.sqrt(np.maximum(S ** 2 - 4 * P, 0.0))

        alpha = (S + discriminant) / 2.0
        beta = (S - discriminant) / 2.0

        # Protect against division by zero in macro constants
        denom = Vc * (alpha - beta)
        denom = np.where(denom == 0, 1e-10, denom) if isinstance(denom, np.ndarray) else (
            1e-10 if denom == 0 else denom)

        A = (alpha - k21) / denom
        B = (k21 - beta) / denom

        return alpha, beta, A, B, k10, k12, k21

    def _solve_trajectory(self, log_samples, target_times, doses_list):
        """Solves the differential equations for concentration over time using RK4."""
        if log_samples.ndim == 1:
            log_samples = log_samples[np.newaxis, :]

        n_sims = log_samples.shape[0]
        n_steps = len(target_times)

        params = np.exp(np.clip(log_samples, -10, 10))
        Vc, Vp, CL, Q = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        _, _, _, _, k10, k12, k21 = self._calculate_pk_constants(Vc, Vp, CL, Q)

        y = np.zeros((2, n_sims))
        results = np.zeros((2, n_sims, n_steps))

        def get_derivatives(state, t):
            central_conc = np.maximum(state[0], 0.0)
            peripheral_conc = np.maximum(state[1], 0.0)

            infusion_rate_array = np.zeros(n_sims)
            for dose in doses_list:
                mask = (t >= dose['start']) & (t <= (dose['start'] + dose['t_inf']))
                if np.any(mask):
                    infusion_rate_array[mask] = dose['dose'] / dose['t_inf']

            dc_dt = (infusion_rate_array / Vc) - (k10 + k12) * central_conc + k21 * peripheral_conc
            dp_dt = k12 * central_conc - k21 * peripheral_conc
            return np.array([dc_dt, dp_dt])

        for i in range(n_steps - 1):
            results[:, :, i] = y
            t = target_times[i]
            dt = target_times[i + 1] - t

            sub_steps = int(np.ceil(dt / 0.1))
            dt_sub = dt / sub_steps

            for _ in range(sub_steps):
                k1 = get_derivatives(y, t)
                k2 = get_derivatives(y + dt_sub / 2 * k1, t + dt_sub / 2)
                k3 = get_derivatives(y + dt_sub / 2 * k2, t + dt_sub / 2)
                k4 = get_derivatives(y + dt_sub * k3, t + dt_sub)
                y = y + (dt_sub / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                t += dt_sub

        results[:, :, -1] = y
        return results[0], results[1]

    def _objective_function(self, active_log_params, doses, labs):
        """Calculates the MAP objective (Log-Likelihood + Prior Penalty)."""
        safe_pop_mean = np.maximum(self.population_mean, 1e-10)
        full_log_params = np.copy(np.log(safe_pop_mean))
        full_log_params[self.has_iiv] = active_log_params

        # 1. Extract lab data
        lab_times = np.array([l['draw_time'] for l in labs])
        lab_vals = np.array([l['val'] for l in labs])

        # 2. Create a stable integration grid ensuring exact lab times are hit
        t_max = np.max(lab_times)
        sim_grid = np.arange(0, t_max + 0.1, 0.1)
        sim_grid = np.unique(np.sort(np.concatenate([sim_grid, lab_times])))

        # 3. Solve trajectory and interpolate predictions
        cc_full, _ = self._solve_trajectory(full_log_params, sim_grid, doses)
        preds = np.array([np.interp(t, sim_grid, cc_full[0]) for t in lab_times])

        # 4. Likelihood Calculation
        bounded_predictions = np.maximum(preds, 0.1)
        sigma = self._calculate_sigma(bounded_predictions)
        log_likelihood = 0.5 * np.sum(((lab_vals - bounded_predictions) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

        # 5. Prior Penalty (Mahalanobis Distance)
        diff = active_log_params - np.log(self.population_mean[self.has_iiv])
        penalty = 0.5 * diff.T @ self.inv_active_cov @ diff

        return log_likelihood + penalty

    def fit_patient(self, clinical_data):
        """Executes the Bayesian fitting routine against patient observations."""
        doses_list = []
        labs_list = []

        for _, row in clinical_data.iterrows():
            if row['Event'] == 'Dose':
                doses_list.append({
                    'dose': float(row['Dose']),
                    'start': float(row['Time_hr']),
                    't_inf': float(row['InfusionTime'])
                })
            elif row['Event'] == 'Level':
                if pd.notna(row['Level']):
                    labs_list.append({
                        'draw_time': float(row['Time_hr']),
                        'val': float(row['Level'])
                    })

        if not labs_list:
            print("No levels provided for fitting. Returning population priors.")
            self.calibrated = False
            return self.population_mean

            # Protect the initial guess
        safe_pop_mean = np.maximum(self.population_mean, 1e-10)
        initial_guess = np.log(safe_pop_mean[self.has_iiv])

        optimization_result = minimize(
            self._objective_function,
            initial_guess,
            args=(doses_list, labs_list),
            method='BFGS',
            options={'disp': False}
        )

        if optimization_result.success:
            # Protect the final parameter assembly
            final_log_params = np.log(safe_pop_mean.copy())
            final_log_params[self.has_iiv] = optimization_result.x
            self.map_params = np.exp(final_log_params)
            self.calibrated = True

            # Extract posterior covariance from inverse Hessian
            if hasattr(optimization_result.hess_inv, "todense"):
                active_posterior_cov = optimization_result.hess_inv.todense()
            else:
                active_posterior_cov = optimization_result.hess_inv

            self.posterior_covariance = np.zeros_like(self.full_covariance)
            self.posterior_covariance[np.ix_(self.has_iiv, self.has_iiv)] = active_posterior_cov
        else:
            print("Optimization failed to converge. Falling back to population parameters.")
            self.calibrated = False
            self.posterior_covariance = None

        return self.map_params if self.calibrated else self.population_mean

    def get_coefficients_of_variation(self):
        """Calculates CV% for both population prior and individual post-fit states."""
        prior_variances = np.diag(self.full_covariance)
        prior_cvs = np.sqrt(np.exp(prior_variances) - 1) * 100
        prior_cvs[~self.has_iiv] = 0.0

        fit_cvs = None
        if self.calibrated and self.posterior_covariance is not None:
            fit_variances = np.diag(self.posterior_covariance)
            fit_cvs = np.sqrt(np.exp(fit_variances) - 1) * 100
            fit_cvs[~self.has_iiv] = 0.0

        return prior_cvs, fit_cvs

    def evaluate_regimen(self, dose_amt, interval, t_inf, n_samples=500000):
        """Simulates steady-state AUC/Peak distributions to calculate PTA."""
        mu_log, cov_matrix = self._get_distribution_params()

        log_samples = np.random.multivariate_normal(mu_log, cov_matrix, n_samples)
        param_samples = np.exp(log_samples)

        Vc = param_samples[:, 0]
        Vp = param_samples[:, 1]
        CL = param_samples[:, 2]
        Q = param_samples[:, 3]

        alpha, beta, A, B, _, _, _ = self._calculate_pk_constants(Vc, Vp, CL, Q)

        # 1. Exact AUC24 calculation
        daily_dose = dose_amt * (24.0 / interval)
        auc_samples = daily_dose / CL

        # Protect against division by zero in 1-compartment Monte Carlo arrays
        beta_safe = np.where(beta == 0, 1e-10, beta)
        denom_beta = 1 - np.exp(-beta_safe * interval)
        denom_beta_safe = np.where(denom_beta == 0, 1e-10, denom_beta)

        # 2. Exact Deterministic Peak calculation
        R = dose_amt / t_inf
        peak_samples = R * (
                (A / alpha) * ((1 - np.exp(-alpha * t_inf)) / (1 - np.exp(-alpha * interval))) +
                (B / beta_safe) * ((1 - np.exp(-beta_safe * t_inf)) / denom_beta_safe)
        )

        metrics = {
            'prob_subtherapeutic': np.mean(auc_samples < self.target_auc_min) * 100,
            'prob_target_attainment': np.mean(
                (auc_samples >= self.target_auc_min) & (auc_samples <= self.target_auc_max)) * 100,
            'prob_supratherapeutic': np.mean(auc_samples > self.target_auc_max) * 100,
            'prob_supratherapeutic_peak': np.mean(peak_samples > self.peak) * 100
        }

        return auc_samples, peak_samples, metrics

    def simulate_profile(self, params, clinical_data, sim_step=(1.0 / 60.0), extra_hours=48.0):
        """Simulates a single concentration-time point-estimate profile."""
        if clinical_data.empty:
            return np.array([0]), np.array([0])

        max_time = clinical_data['Time_hr'].max()
        times = np.arange(0, max_time + extra_hours, sim_step)
        total_conc = self._compute_superposition(params, clinical_data, times)

        return times, total_conc

    def calculate_ci_boundaries(self, params, clinical_data, sim_step=(1.0 / 60.0), extra_hours=48.0, n_samples=500):
        """Executes a Monte Carlo simulation across time to return 95% CI bounds."""
        if clinical_data.empty:
            return None, None

        max_time = clinical_data['Time_hr'].max()
        times = np.arange(0, max_time + extra_hours, sim_step)

        is_prior = np.array_equal(params, self.population_mean)
        if is_prior:
            safe_pop_mean = np.maximum(self.population_mean, 1e-10)
            mu_log = np.log(safe_pop_mean)
            cov_matrix = self.full_covariance
        else:
            mu_log, cov_matrix = self._get_distribution_params()

        log_samples = np.random.multivariate_normal(mu_log, cov_matrix, n_samples)
        param_samples = np.exp(log_samples)

        matrix_profiles = np.zeros((n_samples, len(times)))
        for idx in range(n_samples):
            matrix_profiles[idx, :] = self._compute_superposition(param_samples[idx], clinical_data, times)

        ci_lower = np.percentile(matrix_profiles, 2.5, axis=0)
        ci_upper = np.percentile(matrix_profiles, 97.5, axis=0)

        return ci_lower, ci_upper

    def _compute_superposition(self, params, clinical_data, times):
        """Computes the exact analytical superposition of multiple doses over time."""
        Vc, Vp, CL, Q = params[0], params[1], params[2], params[3]
        alpha, beta, A, B, _, _, _ = self._calculate_pk_constants(Vc, Vp, CL, Q)

        conc_line = np.zeros_like(times)
        doses = clinical_data[clinical_data['Event'] == 'Dose']

        beta_safe = 1e-10 if beta == 0 else beta

        for _, row in doses.iterrows():
            t_start, dose_amt, t_inf = float(row['Time_hr']), float(row['Dose']), float(row['InfusionTime'])
            if pd.isna(dose_amt) or pd.isna(t_inf) or t_inf <= 0:
                continue

            R = dose_amt / t_inf
            t_rel = times - t_start

            during_inf = (t_rel >= 0) & (t_rel <= t_inf)
            post_inf = (t_rel > t_inf)

            conc_line[during_inf] += R * (
                    (A / alpha) * (1 - np.exp(-alpha * t_rel[during_inf])) +
                    (B / beta_safe) * (1 - np.exp(-beta_safe * t_rel[during_inf]))
            )

            t_post = t_rel[post_inf] - t_inf
            conc_line[post_inf] += R * (
                    (A / alpha) * (1 - np.exp(-alpha * t_inf)) * np.exp(-alpha * t_post) +
                    (B / beta_safe) * (1 - np.exp(-beta_safe * t_inf)) * np.exp(-beta_safe * t_post)
            )

        return conc_line

    def suggest_regimens(self, dose_step=25.0, n_samples=50000):
        """Simulates a grid of intervals/doses to return the optimal regimens."""
        mu_log, cov_matrix = self._get_distribution_params()

        log_samples = np.random.multivariate_normal(mu_log, cov_matrix, n_samples)
        param_samples = np.exp(log_samples)

        Vc = param_samples[:, 0]
        Vp = param_samples[:, 1]
        CL = param_samples[:, 2]
        Q = param_samples[:, 3]

        alpha, beta, A, B, _, _, _ = self._calculate_pk_constants(Vc, Vp, CL, Q)

        intervals = [6, 8, 12, 24]
        min_dose = max(dose_step, np.floor((5.0 * self.weight) / dose_step) * dose_step)
        max_dose = np.ceil((35.0 * self.weight) / dose_step) * dose_step
        doses = np.arange(min_dose, max_dose + dose_step, dose_step)

        grid_results = {}

        # Protect against division by zero in 1-compartment arrays
        beta_safe = np.where(beta == 0, 1e-10, beta)

        for interval in intervals:
            regimen_scores = []

            # Precompute denominators that only depend on the interval
            denom_alpha = 1 - np.exp(-alpha * interval)
            denom_beta = 1 - np.exp(-beta_safe * interval)
            denom_beta_safe = np.where(denom_beta == 0, 1e-10, denom_beta)

            for dose in doses:
                t_inf = 1.0 if dose < 1000 else (1.5 if dose < 1500 else 2.0)

                # 1. Deterministic AUC24
                daily_dose = dose * (24.0 / interval)
                auc_samples = daily_dose / CL

                # 2. Deterministic Peak
                R = dose / t_inf
                peak_samples = R * (
                        (A / alpha) * ((1 - np.exp(-alpha * t_inf)) / denom_alpha) +
                        (B / beta_safe) * ((1 - np.exp(-beta_safe * t_inf)) / denom_beta_safe)
                )

                # Probabilities
                prob_target_attainment = np.mean(
                    (auc_samples >= self.target_auc_min) & (auc_samples <= self.target_auc_max)) * 100
                prob_supratherapeutic = np.mean(auc_samples > self.target_auc_max) * 100
                prob_supratherapeutic_peak = np.mean(peak_samples > self.peak) * 100

                regimen_scores.append({
                    'dose': int(dose),
                    'pta': prob_target_attainment,
                    'score': prob_target_attainment,  # Maximize PTA
                    'supra_risk': prob_supratherapeutic,
                    'peak_risk': prob_supratherapeutic_peak
                })

            best = sorted(regimen_scores, key=lambda x: x['score'], reverse=True)[0]

            grid_results[f"q{interval}hr"] = {
                'Optimal Dose': f"{best['dose']} mg",
                '% PTA': f"{best['pta']:.1f}%",
                'AUC > 600 Risk': f"{best['supra_risk']:.1f}%",
                'Peak > 50 mg/L Risk': f"{best['peak_risk']:.1f}%"
            }

        return grid_results

    def get_steady_state_auc(self, clinical_data):
        """Calculates deterministic steady-state AUC based on current interval."""
        if not self.calibrated or clinical_data.empty:
            return None

        doses = clinical_data[clinical_data['Event'] == 'Dose']
        if doses.empty:
            return None

        last_dose = doses.iloc[-1]

        if len(doses) > 1:
            interval = round(doses.iloc[-1]['Time_hr'] - doses.iloc[-2]['Time_hr'])
        else:
            interval = 8.0

        daily_dose = last_dose['Dose'] * (24.0 / interval)
        CL = self.map_params[2]
        return daily_dose / CL

    def estimate_1compartment_pk(self, clinical_data):
        """Basic 1-compartment log-linear regression estimate."""
        from scipy.stats import linregress
        if clinical_data.empty:
            return {"error": "No clinical data."}

        doses = clinical_data[clinical_data['Event'] == 'Dose'].copy()
        levels = clinical_data[clinical_data['Event'] == 'Level'].copy()

        if len(doses) == 0 or len(levels) < 2:
            return {"error": "Need at least 1 dose and 2 levels for estimation."}

        rel_times = []
        log_conc = []

        last_dose_row = doses.iloc[-1]
        last_dose_amt = last_dose_row['Dose']
        t_inf = last_dose_row['InfusionTime']

        if len(doses) > 1:
            intervals = np.diff(doses['Time_hr'])
            interval = round(np.mean(intervals[intervals > 0])) if any(intervals > 0) else 24.0
        else:
            interval = 24.0

        for _, level in levels.iterrows():
            lvl_time = level['Time_hr']
            past_doses = doses[doses['Time_hr'] <= lvl_time]
            if len(past_doses) > 0:
                last_dose = past_doses.iloc[-1]
                t_since_dose = lvl_time - last_dose['Time_hr']
                if level['Level'] > 0:
                    rel_times.append(t_since_dose)
                    log_conc.append(np.log(level['Level']))

        if len(rel_times) < 2:
            return {"error": "Need at least 2 levels with preceding doses."}

        slope, intercept, _, _, _ = linregress(rel_times, log_conc)
        ke = -slope

        if ke <= 0:
            return {"error": "Elimination rate is <= zero. Check peak/trough validity."}

        t_half = np.log(2) / ke
        Cmax = np.exp(intercept - ke * t_inf)
        Cmin = np.exp(intercept - ke * interval)

        Rate = last_dose_amt / t_inf
        V = (Rate / ke) * (1 - np.exp(-ke * t_inf)) / (Cmax * (1 - np.exp(-ke * interval)))
        V_kg = V / self.weight
        CL = ke * V

        AUC = (0.5 * (Cmin + Cmax) * t_inf + (Cmax - Cmin) / ke) * 24 / interval

        return {
            "Half-life (hr)": t_half,
            "Clearance (L/hr)": CL,
            "Volume of Distribution (L/kg)": V_kg,
            "Estimated Cmax (mg/L)": Cmax,
            "Estimated Cmin (mg/L)": Cmin,
            "Estimated AUC24": AUC
        }