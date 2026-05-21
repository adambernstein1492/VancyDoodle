import numpy as np
import pandas as pd
from scipy.optimize import minimize
import models


class VancomycinBayesEngine:
    def __init__(self, weight_kg, height_cm, age_years, creatinine, model="Smit2021",
                 target_auc_min=400.0, target_auc_max=600.0, trough_min=8.0, trough_max=12.0, peak=50.0):
        self.weight = weight_kg
        self.height = height_cm
        self.creatinine = creatinine
        self.age = age_years
        self.target_auc_min = target_auc_min
        self.target_auc_max = target_auc_max
        self.trough_min = trough_min
        self.trough_max = trough_max
        self.peak = peak
        self.dose_history = []
        self.level_history = []
        self.map_params = None
        self.calibrated = False
        self.posterior_covariance = None

        # Initialize PK model and error structure
        self.population_mean, self.full_covariance, self.has_iiv, self.error_config = self._initialize_model(model)

        # Pre-invert active covariance for Mahalanobis penalty
        active_cov = self.full_covariance[np.ix_(self.has_iiv, self.has_iiv)]
        self.inv_active_cov = np.linalg.inv(active_cov)

    def _initialize_model(self, model_name):
        """Standardizes model selection and error term definitions."""
        if model_name == 'Smit2021':
            priors, cov, iiv, error_config = models.Smit2021(self.weight, self.height, self.creatinine)

        if model_name == 'Lamarre2000':
            priors, cov, iiv, error_config = models.Lamarre2000(self.weight, self.height, self.age, self.creatinine)

        if model_name == 'Le2013':
            priors, cov, iiv, error_config = models.Le2013(self.weight, self.height, self.age, self.creatinine)

        return priors, cov, np.array(iiv, dtype=bool), error_config

    def _calculate_sigma(self, predictions):
        """Calculates SD of error with a safety floor to prevent division by zero."""
        if self.error_config["type"] == "proportional":
            # Add a tiny constant (1e-3) so sigma is never 0
            return np.maximum(predictions * self.error_config["sigma"], 0.001)
        if self.error_config["type"] == "fixed":
            return self.error_config["sigma"]

    def _solve_trajectory(self, log_samples, target_times, doses_list):
        if log_samples.ndim == 1:
            log_samples = log_samples[np.newaxis, :]

        n_sims = log_samples.shape[0]
        n_steps = len(target_times)

        params = np.exp(np.clip(log_samples, -10, 10))
        vc, vp, cl, q = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        k10, k12, k21 = cl / vc, q / vc, q / vp

        y = np.zeros((2, n_sims))
        results = np.zeros((2, n_sims, n_steps))

        def get_derivatives(state, t):
            cc = np.maximum(state[0], 0.0)
            cp = np.maximum(state[1], 0.0)

            r = np.zeros(n_sims)
            for dose in doses_list:
                mask = (t >= dose['start']) & (t <= (dose['start'] + dose['t_inf']))
                if np.any(mask):
                    r[mask] = dose['dose'] / dose['t_inf']

            return np.array([
                (r / vc) - (k10 + k12) * cc + k21 * cp,
                k12 * cc - k21 * cp
            ])

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
        """MAP Bayesian objective with stable integration grid."""
        full_log_params = np.copy(np.log(self.population_mean))
        full_log_params[self.has_iiv] = active_log_params

        # 1. Extract lab data
        lab_times = np.array([l['draw_time'] for l in labs])
        lab_vals = np.array([l['val'] for l in labs])

        # 2. Create a stable integration grid (e.g., every 0.1 hours)
        # Ensure it includes the very last lab draw time
        t_max = np.max(lab_times)
        sim_grid = np.arange(0, t_max + 0.1, 0.1)
        # Ensure exact lab times are included in the grid for accuracy
        sim_grid = np.unique(np.sort(np.concatenate([sim_grid, lab_times])))

        # 3. Solve trajectory over the fine grid
        cc_full, _ = self._solve_trajectory(full_log_params, sim_grid, doses)

        # 4. Extract predictions at exact lab times via indexing or interpolation
        # cc_full is [1, len(sim_grid)], we need values at lab_times
        preds = np.array([np.interp(t, sim_grid, cc_full[0]) for t in lab_times])

        # 5. Likelihood Calculation
        safe_preds = np.maximum(preds, 0.1)
        sigma = self._calculate_sigma(safe_preds)
        log_likelihood = 0.5 * np.sum(((lab_vals - safe_preds) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

        # 6. Prior Penalty
        diff = active_log_params - np.log(self.population_mean[self.has_iiv])
        penalty = 0.5 * diff.T @ self.inv_active_cov @ diff

        return log_likelihood + penalty

    def fit_patient(self, clinical_data):
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

        if len(labs_list) == 0:
            print("No levels provided for fitting. Returning population priors.")
            self.calibrated = False
            return self.population_mean

        # Initial guess in log-space for parameters with Inter-Individual Variance (IIV)
        initial_guess = np.log(self.population_mean[self.has_iiv])

        # Run minimize using 'BFGS' to get the inverse Hessian (hess_inv)
        # which acts as our posterior parameter covariance matrix
        res = minimize(
            self._objective_function,
            initial_guess,
            args=(doses_list, labs_list),
            method='BFGS',
            options={'disp': False}
        )

        if res.success:
            final_log_params = np.log(self.population_mean.copy())
            final_log_params[self.has_iiv] = res.x
            self.map_params = np.exp(final_log_params)
            self.calibrated = True

            # --- EXTRACT POSTERIOR COVARIANCE MATRIX ---
            # res.hess_inv contains the covariance matrix for the active parameters in log-space
            # If it's an L-BFGS-B/BFGS operator object, convert it to a dense array via .todense()
            if hasattr(res.hess_inv, "todense"):
                active_posterior_cov = res.hess_inv.todense()
            else:
                active_posterior_cov = res.hess_inv

            # Reconstruct the full 4x4 matrix, filling the non-IIV parameters with zeros
            self.posterior_covariance = np.zeros_like(self.full_covariance)
            self.posterior_covariance[np.ix_(self.has_iiv, self.has_iiv)] = active_posterior_cov

        else:
            print("Optimization failed to converge. Falling back to population parameters.")
            self.calibrated = False
            self.posterior_covariance = None

        return self.map_params if self.calibrated else self.population_mean

    def get_coefficients_of_variation(self):
        """
        Calculates the parameter Coefficient of Variation (%) for both
        the population prior and individual post-fit states.
        """
        # Extract the variance diagonals from log-space
        prior_variances = np.diag(self.full_covariance)

        # Calculate Prior CV% using log-normal translation
        prior_cvs = np.sqrt(np.exp(prior_variances) - 1) * 100
        # Zero out parameters without Inter-Individual Variability (IIV)
        prior_cvs[~self.has_iiv] = 0.0

        fit_cvs = None
        if self.calibrated and self.posterior_covariance is not None:
            fit_variances = np.diag(self.posterior_covariance)
            fit_cvs = np.sqrt(np.exp(fit_variances) - 1) * 100
            fit_cvs[~self.has_iiv] = 0.0

        return prior_cvs, fit_cvs

    def evaluate_regimen(self, dose_amt, interval, t_inf, n_samples=500000):
        """
        Simulates steady-state AUC and Peak distributions and calculates
        all relevant clinical probabilities.
        """
        if not self.calibrated:
            mu_log = np.log(self.population_mean)
            cov_matrix = self.full_covariance
        else:
            mu_log = np.log(self.map_params)
            cov_matrix = self.posterior_covariance if self.posterior_covariance is not None else self.full_covariance

        # Generate sample population
        log_samples = np.random.multivariate_normal(mu_log, cov_matrix, n_samples)
        param_samples = np.exp(log_samples)

        Vc, Vp, CL, Q = param_samples[:, 0], param_samples[:, 1], param_samples[:, 2], param_samples[:, 3]

        # 1. Exact AUC24 calculation
        daily_dose = dose_amt * (24.0 / interval)
        auc_samples = daily_dose / CL

        # 2. Exact Deterministic Peak calculation
        k10, k12, k21 = CL / Vc, Q / Vc, Q / Vp
        S, P = k10 + k12 + k21, k10 * k21

        alpha = (S + np.sqrt(S ** 2 - 4 * P)) / 2.0
        beta = (S - np.sqrt(S ** 2 - 4 * P)) / 2.0
        A = (alpha - k21) / (Vc * (alpha - beta))
        B = (k21 - beta) / (Vc * (alpha - beta))

        R = dose_amt / t_inf

        peak_samples = R * (
                (A / alpha) * ((1 - np.exp(-alpha * t_inf)) / (1 - np.exp(-alpha * interval))) +
                (B / beta) * ((1 - np.exp(-beta * t_inf)) / (1 - np.exp(-beta * interval)))
        )

        # --- NEW: Perform all statistical percentage calculations here ---
        metrics = {
            'p_sub': np.mean(auc_samples < self.target_auc_min) * 100,
            'p_target': np.mean((auc_samples >= self.target_auc_min) & (auc_samples <= self.target_auc_max)) * 100,
            'p_supra': np.mean(auc_samples > self.target_auc_max) * 100,
            'p_peak': np.mean(peak_samples > self.peak) * 100
        }

        return auc_samples, peak_samples, metrics

    def simulate_profile(self, params, clinical_data, sim_step=(1.0 / 60.0), extra_hours=48.0):
        """
        Simulates a single concentration-time profile line.
        Extremely fast point-estimate simulation.
        """
        if clinical_data.empty:
            return np.array([0]), np.array([0])

        max_time = clinical_data['Time_hr'].max()
        times = np.arange(0, max_time + extra_hours, sim_step)
        total_conc = self._compute_superposition(params, clinical_data, times)
        return times, total_conc

    def calculate_ci_boundaries(self, params, clinical_data, sim_step=(1.0 / 60.0), extra_hours=48.0, n_samples=500):
        """
        Explicitly runs the Monte Carlo simulation to return upper and lower bounds.
        Run this ONLY once per fit operation to save computing power.
        """
        if clinical_data.empty:
            return None, None

        max_time = clinical_data['Time_hr'].max()
        times = np.arange(0, max_time + extra_hours, sim_step)

        is_prior = np.array_equal(params, self.population_mean)
        if is_prior:
            mu_log = np.log(self.population_mean)
            cov_matrix = self.full_covariance
        else:
            mu_log = np.log(self.map_params)
            cov_matrix = self.posterior_covariance if self.posterior_covariance is not None else self.full_covariance

        # Generate Monte Carlo samples
        log_samples = np.random.multivariate_normal(mu_log, cov_matrix, n_samples)
        param_samples = np.exp(log_samples)

        matrix_profiles = np.zeros((n_samples, len(times)))
        for idx in range(n_samples):
            matrix_profiles[idx, :] = self._compute_superposition(param_samples[idx], clinical_data, times)

        ci_lower = np.percentile(matrix_profiles, 2.5, axis=0)
        ci_upper = np.percentile(matrix_profiles, 97.5, axis=0)
        return ci_lower, ci_upper

    def _compute_superposition(self, params, clinical_data, times):
        """Helper matrix-builder to compute individual superposition lines."""
        Vc, Vp, CL, Q = params[0], params[1], params[2], params[3]
        k10, k12, k21 = CL / Vc, Q / Vc, Q / Vp
        S, P = k10 + k12 + k21, k10 * k21

        alpha = (S + np.sqrt(S ** 2 - 4 * P)) / 2.0
        beta = (S - np.sqrt(S ** 2 - 4 * P)) / 2.0
        A = (alpha - k21) / (Vc * (alpha - beta))
        B = (k21 - beta) / (Vc * (alpha - beta))

        conc_line = np.zeros_like(times)
        doses = clinical_data[clinical_data['Event'] == 'Dose']

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
                    (B / beta) * (1 - np.exp(-beta * t_rel[during_inf]))
            )
            t_post = t_rel[post_inf] - t_inf
            conc_line[post_inf] += R * (
                    (A / alpha) * (1 - np.exp(-alpha * t_inf)) * np.exp(-alpha * t_post) +
                    (B / beta) * (1 - np.exp(-beta * t_inf)) * np.exp(-beta * t_post)
            )
        return conc_line

    def suggest_regimens(self, dose_step=25.0, n_samples=50000):
        """
        Simulates a matrix of standard intervals and dose increments.
        Returns a dictionary mapping interval strings (e.g., 'q6hr') to the
        optimal dose and its corresponding safety and efficacy metrics.
        """
        if not self.calibrated:
            mu_log = np.log(self.population_mean)
            cov_matrix = self.full_covariance
        else:
            mu_log = np.log(self.map_params)
            cov_matrix = self.posterior_covariance if self.posterior_covariance is not None else self.full_covariance

        # Generate sample population ONCE for extreme efficiency
        log_samples = np.random.multivariate_normal(mu_log, cov_matrix, n_samples)
        param_samples = np.exp(log_samples)

        Vc = param_samples[:, 0]
        CL = param_samples[:, 2]

        # Calculate micro-constants once for the whole population
        k10, k12, k21 = CL / Vc, param_samples[:, 3] / Vc, param_samples[:, 3] / param_samples[:, 1]
        S, P = k10 + k12 + k21, k10 * k21
        alpha = (S + np.sqrt(S ** 2 - 4 * P)) / 2.0
        beta = (S - np.sqrt(S ** 2 - 4 * P)) / 2.0

        A = (alpha - k21) / (Vc * (alpha - beta))
        B = (k21 - beta) / (Vc * (alpha - beta))

        intervals = [6, 8, 12, 24]
        min_dose = max(dose_step, np.floor((5.0 * self.weight) / dose_step) * dose_step)
        max_dose = np.ceil((35.0 * self.weight) / dose_step) * dose_step
        doses = np.arange(min_dose, max_dose + dose_step, dose_step)

        grid_results = {}

        for interval in intervals:
            regimen_scores = []
            for dose in doses:
                t_inf = 1.0 if dose < 1000 else (1.5 if dose < 1500 else 2.0)

                # 1. Deterministic AUC24
                daily_dose = dose * (24.0 / interval)
                auc_samples = daily_dose / CL

                # 2. Deterministic Peak
                R = dose / t_inf
                peak_samples = R * (
                        (A / alpha) * ((1 - np.exp(-alpha * t_inf)) / (1 - np.exp(-alpha * interval))) +
                        (B / beta) * ((1 - np.exp(-beta * t_inf)) / (1 - np.exp(-beta * interval)))
                )

                # Probabilities
                p_target = np.mean((auc_samples >= self.target_auc_min) & (auc_samples <= self.target_auc_max)) * 100
                p_supra = np.mean(auc_samples > self.target_auc_max) * 100
                p_peak = np.mean(peak_samples > self.peak) * 100

                # Scoring: Maximize PTA, penalize toxicity
                score = p_target
                regimen_scores.append({
                    'dose': int(dose),
                    'pta': p_target,
                    'score': score,
                    'supra_risk': p_supra,
                    'peak_risk': p_peak
                })

            # Find the best dose
            sorted_doses = sorted(regimen_scores, key=lambda x: x['score'], reverse=True)
            best = sorted_doses[0]

            grid_results[f"q{interval}hr"] = {
                'Optimal Dose': f"{best['dose']} mg",
                '% PTA': f"{best['pta']:.1f}%",
                'AUC > 600 Risk': f"{best['supra_risk']:.1f}%",
                'Peak > 50 Risk': f"{best['peak_risk']:.1f}%"
            }

        return grid_results

    def get_steady_state_auc(self, clinical_data):
        """
        Calculates the deterministic steady-state AUC based on the last
        known interval in the dosing history.
        """
        if not self.calibrated or clinical_data.empty:
            return None

        # Filter for doses
        doses = clinical_data[clinical_data['Event'] == 'Dose']
        if doses.empty:
            return None

        # Get the most recent dose and interval
        last_dose = doses.iloc[-1]
        # Calculate interval as time between last two doses, default to 8 if only one dose
        if len(doses) > 1:
            interval = round(doses.iloc[-1]['Time_hr'] - doses.iloc[-2]['Time_hr'])
        else:
            interval = 8.0  # Default fallback

        # AUC_ss = Daily Dose / CL
        daily_dose = last_dose['Dose'] * (24.0 / interval)
        CL = self.map_params[2]  # 3rd parameter is CL

        return daily_dose / CL

    def estimate_1compartment_pk(self, clinical_data):
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
            return {
                "error": "Calculated elimination rate constant is zero or negative. Ensure peak is higher than trough."}

        t_half = np.log(2) / ke

        Cmax = np.exp(intercept - ke * t_inf)
        Cmin = np.exp(intercept - ke * interval)

        Rate = last_dose_amt / t_inf
        V = (Rate / ke) * (1 - np.exp(-ke * t_inf)) / (Cmax * (1 - np.exp(-ke * interval)))
        V_kg = V / self.weight

        CL = ke * V

        daily_dose = last_dose_amt * (24.0 / interval)
        AUC = (0.5 * (Cmin + Cmax) * t_inf + (Cmax - Cmin) / ke) * 24 / interval

        return {
            "Half-life (hr)": t_half,
            "Clearance (L/hr)": CL,
            "Volume of Distribution (L/kg)": V_kg,
            "Estimated Cmax (mg/L)": Cmax,
            "Estimated Cmin (mg/L)": Cmin,
            "Estimated AUC24": AUC
        }