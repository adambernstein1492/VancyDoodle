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

        initial_guess = np.log(self.population_mean[self.has_iiv])

        res = minimize(self._objective_function,
            initial_guess,
            args=(doses_list, labs_list),
            method='Nelder-Mead',
            options={'adaptive': True}
        )

        if res.success:
            final_log_params = np.log(self.population_mean.copy())
            final_log_params[self.has_iiv] = res.x
            self.map_params = np.exp(final_log_params)
            self.calibrated = True
        else:
            print("Optimization failed to converge.")
            self.calibrated = False

        return self.map_params if self.calibrated else self.population_mean

    def suggest_optimal_regimen(self, n_sims=50000, infusion_time=1.0, use_prior=False):
        """
        Blazing fast vectorized optimizer that checks both AUC efficacy
        and Peak toxicity using analytical steady-state. Assumes 1-hr
        infusion.
        """
        # 1. Setup Parameters
        mu = np.log(self.population_mean) if use_prior else np.log(self.map_params)
        cov = self.full_covariance
        samples = np.random.multivariate_normal(mu, cov, n_sims)

        intervals = [6, 8, 12, 24]
        possible_mg = np.arange(25, 2000, 25)
        results = {'best_overall': None, 'by_interval': {}}

        for tau in intervals:
            t_inf = infusion_time  # Standardize for optimization; can adjust based on dose if needed

            # 2. Get Unit Values (Performance for a 1mg dose)
            # We fetch the profile across the interval
            unit_ss = self._solve_steady_state_analytical(samples, tau, dose=1.0)

            # Unit AUC: (Average Conc over interval) * 24 hours
            unit_auc = np.mean(unit_ss, axis=1) * 24

            # Unit Peak: Finding the concentration at the end of infusion
            # Our analytic solver returns steps of 0.1hr. t_inf of 1.0 is index 10.
            peak_idx = int(t_inf / 0.1)
            unit_peak = unit_ss[:, peak_idx]

            # 3. Vectorized Scoring for all Possible Doses
            # Broadcast Dose * Unit Performance -> (n_doses, n_sims)
            auc_matrix = possible_mg[:, np.newaxis] * unit_auc
            peak_matrix = possible_mg[:, np.newaxis] * unit_peak

            # Calculate Probabilities
            pta_matrix = (auc_matrix >= 400) & (auc_matrix <= 600)
            toxic_matrix = (peak_matrix > 50.0)

            pta_scores = np.mean(pta_matrix, axis=1) * 100
            toxic_scores = np.mean(toxic_matrix, axis=1) * 100

            # 4. Selection Logic: Maximize PTA where Toxicity Risk is low (<25%)
            safe_mask = toxic_scores < 25.0
            if np.any(safe_mask):
                best_idx = np.where(safe_mask, pta_scores, -1).argmax()
            else:
                # If no dose is "safe", pick the one with the lowest toxicity risk
                best_idx = np.argmin(toxic_scores)

            results['by_interval'][tau] = {
                'interval_hrs': tau,
                'maint_mg': possible_mg[best_idx],
                'pta': pta_scores[best_idx],
                'risk_toxic_peak': toxic_scores[best_idx],
                'predicted_auc': np.mean(auc_matrix[best_idx])
            }

        # 5. Global Best (highest PTA across all intervals)
        all_regimens = list(results['by_interval'].values())
        results['best_overall'] = max(all_regimens, key=lambda x: x['pta'])

        # 6. Iterative Numerical Search for Loading Dose (Target AUC24 > 450)
        mu = np.log(self.population_mean) if use_prior else np.log(self.map_params)
        # Use a smaller sample size for the iterative simulation to maintain performance
        ld_samples = np.random.multivariate_normal(mu, self.full_covariance, 2000)

        maint_mg = results['best_overall']['maint_mg']
        tau = results['best_overall']['interval_hrs']

        possible_loads = np.arange(maint_mg, 3250, 25)
        chosen_ld = maint_mg
        t_grid_24 = np.linspace(0, 24, 1441)  # 24 hours, 1m steps

        for ld in possible_loads:
            # Construct a 24h regimen: Loading dose at t=0, then scheduled maintenance
            regimen_24h = [{"dose": ld, "start": 0.0, "t_inf": 1.5}]
            for t_start in range(tau, 24, tau):
                regimen_24h.append({"dose": maint_mg, "start": float(t_start), "t_inf": 1.0})

            # Solve the actual trajectory for the central compartment
            cc_24, _ = self._solve_trajectory(ld_samples, t_grid_24, regimen_24h)

            # Calculate AUC24 (Average Conc * 24h)
            auc24_samples = np.mean(cc_24, axis=1) * 24

            # Stop at the first (smallest) dose that hits the efficacy target
            if np.median(auc24_samples) >= 450:
                chosen_ld = ld
                break

        # Fallback if no dose hits the threshold or for very low clearance patients
        results['loading_dose_mg'] = chosen_ld if chosen_ld > 0 else 500

        return results

    def simulate_profile(self, params, clinical_data, sim_step=(1.0/60.0), extra_hours=48.0):
        """
        Simulates the concentration-time profile using a 2-compartment analytical
        solution and the superposition principle.

        Returns:
            times (np.array): Array of time points (in hours).
            total_conc (np.array): Simulated vancomycin concentrations (in mg/L).
        """
        Vc, Vp, CL, Q = params[0], params[1], params[2], params[3]

        # 1. Calculate Micro-constants
        k10 = CL / Vc
        k12 = Q / Vc
        k21 = Q / Vp

        # 2. Calculate Macro-constants (alpha, beta, A, B)
        S = k10 + k12 + k21
        P = k10 * k21

        # Quadratic formula for eigenvalues
        alpha = (S + np.sqrt(S ** 2 - 4 * P)) / 2.0
        beta = (S - np.sqrt(S ** 2 - 4 * P)) / 2.0

        # Coefficients
        A = (alpha - k21) / (Vc * (alpha - beta))
        B = (k21 - beta) / (Vc * (alpha - beta))

        # 3. Setup Simulation Timeline
        # Determine how long to simulate based on the last event in the data
        if clinical_data.empty:
            return np.array([0]), np.array([0])

        max_time = clinical_data['Time_hr'].max()
        # Create a dense time array from 0 out to max_time + extra_hours
        times = np.arange(0, max_time + extra_hours, sim_step)
        total_conc = np.zeros_like(times)

        # 4. Apply Superposition Principle
        # Filter for only the dosing events
        doses = clinical_data[clinical_data['Event'] == 'Dose']

        for _, row in doses.iterrows():
            t_start = float(row['Time_hr'])
            dose_amt = float(row['Dose'])
            t_inf = float(row['InfusionTime'])

            # Skip invalid doses
            if pd.isna(dose_amt) or pd.isna(t_inf) or t_inf <= 0:
                continue

            # Infusion Rate (mg/hr)
            R = dose_amt / t_inf

            # Time relative to the start of THIS specific dose
            t_rel = times - t_start

            # Create boolean masks for vectorization
            during_inf = (t_rel >= 0) & (t_rel <= t_inf)
            post_inf = (t_rel > t_inf)

            # Add contribution DURING the infusion
            total_conc[during_inf] += R * (
                    (A / alpha) * (1 - np.exp(-alpha * t_rel[during_inf])) +
                    (B / beta) * (1 - np.exp(-beta * t_rel[during_inf]))
            )

            # Add contribution AFTER the infusion has stopped
            t_post = t_rel[post_inf] - t_inf
            total_conc[post_inf] += R * (
                    (A / alpha) * (1 - np.exp(-alpha * t_inf)) * np.exp(-alpha * t_post) +
                    (B / beta) * (1 - np.exp(-beta * t_inf)) * np.exp(-beta * t_post)
            )

        return times, total_conc

