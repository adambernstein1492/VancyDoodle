import numpy as np
import pandas as pd
from scipy.optimize import minimize
import models


class VancomycinBayesEngine:
    def __init__(self, weight_kg, height_cm, age_years, creatinine, model="Smit2021",
                 target_auc_min=400.0, target_auc_max=600.0, trough_min=8.0, trough_max=12.0, peak = 50.0):
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

    def _solve_steady_state_analytical(self, log_samples, interval, dose):
        """
        Analytic Steady State solution for a 2-compartment model.
        Eliminates numerical instability and overflow.
        """
        if log_samples.ndim == 1:
            log_samples = log_samples[np.newaxis, :]

        n_sims = log_samples.shape[0]
        params = np.exp(log_samples)
        vc, vp, cl, q = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

        # Calculate micro-constants
        k10 = cl / vc
        k12 = q / vc
        k21 = q / vp

        # Calculate hybrid rate constants (alpha and beta)
        sum_k = k10 + k12 + k21
        prod_k = k10 * k21

        alpha = 0.5 * (sum_k + np.sqrt(sum_k ** 2 - 4 * prod_k))
        beta = 0.5 * (sum_k - np.sqrt(sum_k ** 2 - 4 * prod_k))

        # Coefficients for the two-compartment equation
        A = (alpha - k21) / (vc * (alpha - beta))
        B = (k21 - beta) / (vc * (alpha - beta))

        t_inf = 1.0 if dose < 1000 else 1.5
        t_steps = np.arange(0, interval + 0.1, 0.1)
        results = np.zeros((n_sims, len(t_steps)))

        # Steady State Infusion Formula
        # Css(t) = (Rate/alpha)*A*(1-exp(-alpha*t_inf))/(1-exp(-alpha*tau))*exp(-alpha*(t-t_inf)) ...
        for i, t in enumerate(t_steps):
            # During Infusion
            if t <= t_inf:
                term_a = (A / alpha) * (1 - np.exp(-alpha * t)) / (1 - np.exp(-alpha * interval))
                term_b = (B / beta) * (1 - np.exp(-beta * t)) / (1 - np.exp(-beta * interval))
                # Adjustment for accumulation from previous doses
                # Note: A more precise during-infusion SS formula
                inf_a = (A / alpha) * ((1 - np.exp(-alpha * t)) + (np.exp(-alpha * t) * (
                            np.exp(-alpha * (interval - t_inf)) * (1 - np.exp(-alpha * t_inf)) / (
                                1 - np.exp(-alpha * interval)))))
                inf_b = (B / beta) * ((1 - np.exp(-beta * t)) + (np.exp(-beta * t) * (
                            np.exp(-beta * (interval - t_inf)) * (1 - np.exp(-beta * t_inf)) / (
                                1 - np.exp(-beta * interval)))))
                results[:, i] = (dose / t_inf) * (inf_a + inf_b)

            # After Infusion (Decay)
            else:
                post_a = (A / alpha) * (1 - np.exp(-alpha * t_inf)) / (1 - np.exp(-alpha * interval)) * np.exp(
                    -alpha * (t - t_inf))
                post_b = (B / beta) * (1 - np.exp(-beta * t_inf)) / (1 - np.exp(-beta * interval)) * np.exp(
                    -beta * (t - t_inf))
                results[:, i] = (dose / t_inf) * (post_a + post_b)

        return results

    def _get_derivatives_vectorized(self, y, r, vc, k10, k12, k21):
        cc, cp = y[0], y[1]
        d_cc = (r / vc) - (k10 + k12) * cc + k21 * cp
        d_cp = k12 * cc - k21 * cp
        return np.array([d_cc, d_cp])

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

    def evaluate_regimen(self, dose_amt, interval, t_inf, n_samples=50000):
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



    def simulate_profile(self, params, clinical_data, sim_step=(1.0/60.0), extra_hours=48.0):
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
        Returns a dictionary mapping interval strings (e.g., 'q6hr') to a list
        of 5 doses centered around the optimal choice for that interval.
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
                    'supra_risk': np.mean(auc_samples > self.target_auc_max) * 100  # Keep for table display
                })

            # Find the best dose and its 2 neighbors on each side
            sorted_doses = sorted(regimen_scores, key=lambda x: x['score'], reverse=True)
            best_dose = sorted_doses[0]['dose']
            best_idx = np.where(doses == best_dose)[0][0]

            # Extract 5 surrounding doses (clamped to array boundaries)
            idx_range = np.arange(best_idx - 2, best_idx + 3)
            idx_range = np.clip(idx_range, 0, len(doses) - 1)

            grid_results[f"q{interval}hr"] = [doses[i] for i in idx_range]

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
        # Calculate interval as time between last two doses, default to 12 if only one dose
        if len(doses) > 1:
            interval = doses.iloc[-1]['Time_hr'] - doses.iloc[-2]['Time_hr']
        else:
            interval = 12.0  # Default fallback

        # AUC_ss = Daily Dose / CL
        daily_dose = last_dose['Dose'] * (24.0 / interval)
        CL = self.map_params[2]  # 3rd parameter is CL

        return daily_dose / CL

