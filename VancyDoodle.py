import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
from scipy.stats import linregress


class VancomycinBayesEngine:
    def __init__(self, weight_kg, height_cm, age_years, creatinine, model="Smit2021",
                 target_auc_min=400.0, target_auc_max=600.0, trough_min=8.0, trough_max=12.0):
        self.weight = weight_kg
        self.height = height_cm
        self.creatinine = creatinine
        self.age_baseline = age_years
        self.target_auc_min = target_auc_min
        self.target_auc_max = target_auc_max
        self.trough_min = trough_min
        self.trough_max = trough_max
        self.dose_history = []
        self.level_history = []
        self.map_log_params = None
        self.calibrated = False

        # Initialize PK model and error structure
        self.pop_log_means, self.full_covariance, self.has_iiv, self.error_config = self._initialize_model(model)

        # Pre-invert active covariance for Mahalanobis penalty
        active_cov = self.full_covariance[np.ix_(self.has_iiv, self.has_iiv)]
        self.inv_active_cov = np.linalg.inv(active_cov)

    def _initialize_model(self, model_name):
        """Standardizes model selection and error term definitions."""
        if model_name == "Smit2021":
            # Bedside Schwartz for CrCl (capped at 120 mL/min)
            CrCl = min(0.413 * self.height / self.creatinine, 120.0)

            priors = np.array([
                8.9 * self.weight / 22.1,  # Vc
                12.3 * self.weight / 22.1,  # Vp
                2.12 * np.power(self.weight / 22.1, 0.745) * CrCl / 100,  # CL
                1.55 * np.power(self.weight / 22.1, 0.599)  # Q
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
            error_config = {"type": "proportional", "cv": 0.0789}

        return np.log(priors), cov, np.array(iiv, dtype=bool), error_config

    def _calculate_sigma(self, predictions):
        """Calculates SD of error with a safety floor to prevent division by zero."""
        if self.error_config["type"] == "proportional":
            # Add a tiny constant (1e-3) so sigma is never 0
            return np.maximum(predictions * self.error_config["cv"], 0.001)
        return np.ones_like(predictions)

    def _solve_trajectory(self, log_samples, target_times, doses_list):
        if log_samples.ndim == 1:
            log_samples = log_samples[np.newaxis, :]

        n_sims = log_samples.shape[0]
        n_steps = len(target_times)

        # --- Safeguard 1: Parameter Clipping ---
        # Prevent extreme outliers from Monte Carlo draws
        params = np.exp(np.clip(log_samples, -10, 10))
        vc, vp, cl, q = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        k10, k12, k21 = cl / vc, q / vc, q / vp

        y = np.zeros((2, n_sims))
        results = np.zeros((2, n_sims, n_steps))

        def get_derivatives(state, t):
            # --- Safeguard 2: State Clipping ---
            cc = np.maximum(state[0], 0.0)
            cp = np.maximum(state[1], 0.0)

            r = np.zeros(n_sims)
            for dose in doses_list:
                # Check if current time t is within an infusion window
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

            # --- Safeguard 3: Sub-stepping ---
            # If dt is too large (e.g. > 0.1h), break it down to prevent explosion
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
        full_log_params = np.copy(self.pop_log_means)
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
        diff = active_log_params - self.pop_log_means[self.has_iiv]
        penalty = 0.5 * diff.T @ self.inv_active_cov @ diff

        return log_likelihood + penalty

    def fit_patient(self, doses, labs):
        # CHANGE: Only pass the subset of parameters that have IIV (True in has_iiv)
        initial_guess = self.pop_log_means[self.has_iiv]

        res = minimize(self._objective_function, initial_guess, args=(doses, labs), method='Nelder-Mead', options={'adaptive': True})

        if res.success:
            final_log_params = self.pop_log_means.copy()
            final_log_params[self.has_iiv] = res.x
            self.map_log_params = final_log_params
            self.calibrated = True
        else:
            # It's better to return the result object or raise an error for the UI
            print("Optimization failed to converge.")

        return np.exp(self.map_log_params) if self.calibrated else np.exp(self.pop_log_means)

    def run_monte_carlo(self, maint_dose, interval, n_sims=10000):
        """Streamlined PTA calculation using analytical steady-state solution."""
        center = self.map_log_params if self.calibrated else self.pop_log_means
        log_samples = np.random.multivariate_normal(center, self.full_covariance, n_sims)
        sim_cl = np.exp(log_samples[:, 2])

        daily_dose = maint_dose * (24.0 / interval)
        aucs = daily_dose / sim_cl

        return {
            "pta": np.mean((aucs >= self.target_auc_min) & (aucs <= self.target_auc_max)),
            "pct_above": np.mean(aucs > self.target_auc_max),
            "median_auc": np.median(aucs),
            "all_aucs": aucs
        }

    def suggest_optimal_regimen(self, n_sims=50000, infusion_time=1.0, use_prior=False):
        """
        Blazing fast vectorized optimizer that checks both AUC efficacy
        and Peak toxicity using analytical steady-state. Assumes 1-hr
        infusion.
        """
        # 1. Setup Parameters
        mu = self.pop_log_means if use_prior else self.map_log_params
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
        mu = self.pop_log_means if use_prior else self.map_log_params
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

    def calculate_sawchuk_moiser(self, dose, t_inf, c_peak, c_trough, dt_start, dt_peak, dt_trough, tau):
        import numpy as np
        import matplotlib.pyplot as plt

        # 1. Convert absolute datetimes to hours relative to dose start (t=0)
        # We assume dt_start is the 0.0 reference point
        t_peak = (dt_peak - dt_start).total_seconds() / 3600.0
        t_trough = (dt_trough - dt_start).total_seconds() / 3600.0

        # 2. Calculate k (Elimination Constant)
        # k = (ln(C1) - ln(C2)) / (t2 - t1)
        dt = t_trough - t_peak
        k = (np.log(c_peak) - np.log(c_trough)) / dt

        # 3. Extrapolate to True Cmax (end of infusion) and Cmin (end of interval)
        # Cmax = C_peak / e^(-k * time_since_end_of_infusion)
        time_from_inf_end_to_peak = t_peak - t_inf
        c_max = c_peak / np.exp(-k * time_from_inf_end_to_peak)

        # Cmin = C_trough * e^(-k * time_remaining_in_interval)
        time_from_trough_to_end = tau - t_trough
        c_min = c_trough * np.exp(-k * time_from_trough_to_end)

        # 4. Calculate Vd (Sawchuk-Zaske Volume Equation) and CL
        vd = (dose / (t_inf * k)) * ((1 - np.exp(-k * t_inf)) / (c_max - (c_min * np.exp(-k * t_inf))))
        cl = k * vd

        AUC = ((c_min + c_max) / 2 + (c_max - c_min) / k) * 24.0 / tau

        # 5. Plotting the Log-Linear Fit
        t_plot = np.linspace(0, tau, 5000)
        c_plot = np.where(t_plot <= t_inf,
                          ((c_max - c_min) / t_inf) * t_plot + c_min,
                          c_max * np.exp(-k * (t_plot - t_inf)))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_plot, c_plot, color='#e67e22', lw=2.5, label='1-Compartment Fit')
        ax.fill_between(t_plot, c_plot, color='#e67e22', alpha=0.15, label='AUC Integral')

        # Plot markers at the relative times
        ax.scatter([t_peak, t_trough], [c_peak, c_trough], color='red', s=80, zorder=5, label='Measured Levels')

        ax.set_title("Sawchuk-Moiser: Manual PK Estimation")
        ax.set_ylabel("Concentration (mg/L)")
        ax.set_xlabel("Hours Since Dose Start")
        ax.set_ylim(0, max(c_max, c_peak) * 1.2)
        ax.legend()
        ax.grid(alpha=0.2)

        return {
            "k": k, "vd": vd, "vd_l_kg": vd / self.weight,
            "cl": cl, "auc": AUC, "fig": fig,
            "t_peak_rel": t_peak, "t_trough_rel": t_trough
        }

    def get_projection_stats(self, maint_dose, interval, load_dose=0, n_sims=1000, duration_hrs=72, force_prior=False):
        """
        Calculates population trajectories using vectorized numerical integration.
        Optimized for speed and clinical realism.
        """
        num_min = int(duration_hrs * 60) + 1
        t_grid = np.linspace(0, duration_hrs, num_min)

        # 1. Parameter Selection
        center = self.pop_log_means if (force_prior or not self.calibrated) else self.map_log_params
        log_samples = np.random.multivariate_normal(center, self.full_covariance, n_sims)

        # 2. Regimen Construction
        # Regimen A: Standard Maintenance
        reg_std = [{"dose": maint_dose, "start": float(t), "t_inf": 1.0}
                   for t in range(0, duration_hrs, interval)]

        # Regimen B: Loading Dose + Maintenance
        reg_load = []
        if load_dose > 0:
            reg_load.append({"dose": load_dose, "start": 0.0, "t_inf": 1.5})
            for t in range(interval, duration_hrs, interval):
                reg_load.append({"dose": maint_dose, "start": float(t), "t_inf": 1.0})
        else:
            reg_load = reg_std

        # 3. Parallel Vectorized Solve
        cc_std, _ = self._solve_trajectory(log_samples, t_grid, reg_std)
        cc_load, _ = self._solve_trajectory(log_samples, t_grid, reg_load)

        # 4. Statistical Aggregation
        def get_bands(matrix):
            return {
                "low_95": np.percentile(matrix, 2.5, axis=0),
                "low_50": np.percentile(matrix, 25, axis=0),
                "mean": np.mean(matrix, axis=0),
                "high_50": np.percentile(matrix, 75, axis=0),
                "high_95": np.percentile(matrix, 97.5, axis=0)
            }

        return {
            "t_grid": t_grid,
            "standard": get_bands(cc_std),
            "loading": get_bands(cc_load),
            "target_range": [8, 12]  # Updated per your request
        }

    def get_parameter_comparison(self):
        """
        Formats the PK parameter shift (Prior vs Posterior) for Streamlit display.
        Returns a Pandas DataFrame for direct use in st.dataframe() or st.table().
        """
        pop = np.exp(self.pop_log_means)
        ind = np.exp(self.map_log_params) if self.calibrated else [None] * 4
        labels = ["Vc (L)", "Vp (L)", "CL (L/h)", "Q (L/h)"]

        data = []
        for i in range(4):
            row = {
                "Parameter": labels[i],
                "Population Prior": round(pop[i], 3),
                "Individual Estimate": round(ind[i], 3) if ind[i] else "N/A",
                "Percent Shift (%)": round(((ind[i] - pop[i]) / pop[i]) * 100, 1) if ind[i] else "N/A"
            }
            data.append(row)

        return pd.DataFrame(data)

    def format_gui_inputs(self, raw_doses, raw_labs):
        fmt = "%m/%d/%Y %H:%M"
        for d in raw_doses: d['_dt'] = datetime.strptime(d['dt'], fmt)
        for l in raw_labs: l['_dt'] = datetime.strptime(l['dt'], fmt)
        t0 = min([d['_dt'] for d in raw_doses] + [l['_dt'] for l in raw_labs])
        f_doses = [{"dose": float(d['mg']), "start": (d['_dt'] - t0).total_seconds() / 3600, "t_inf": 1.0} for d in
                   raw_doses]
        f_labs = [{"val": float(l['val']), "draw_time": (l['_dt'] - t0).total_seconds() / 3600} for l in raw_labs]
        return f_doses, f_labs, t0

    def add_dose(self, mg, time_hrs, infusion_hrs=1.0):
        """Stores a dose event for Bayesian fitting."""
        self.dose_history.append({
            'mg': mg,
            'time': time_hrs,
            'infusion': infusion_hrs
        })

    def add_level(self, mg_l, time_hrs):
        """Stores a measured drug level for Bayesian fitting."""
        self.level_history.append({
            'value': mg_l,
            'time': time_hrs
        })

    # Visualization Helpers
    def plot_prior_projections(self, stats, measured_level=None, level_time=None):
        """
        Plots population/personalized trajectories with PI bands.

        Parameters:
        - stats: Dict containing 't_grid' and percentile arrays.
        - measured_level: (Optional) The float value of a lab level.
        - level_time: (Optional) The time in hours the level was drawn.
        """
        import matplotlib.pyplot as plt

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        t_grid = stats['t_grid']

        # 1. Plot the Prediction Interval (PI) Bands
        # 95% Interval (Lightest)
        ax.fill_between(t_grid, stats['standard']['low_95'], stats['standard']['high_95'],
                        color='dodgerblue', alpha=0.2, label='95% Population PI')

        # 50% Interval (Darker)
        ax.fill_between(t_grid, stats['standard']['low_50'], stats['standard']['high_50'],
                        color='dodgerblue', alpha=0.4, label='50% Population PI')

        # 2. Plot the Mean Trajectories
        # Maintenance only (Solid)
        ax.plot(t_grid, stats['standard']['mean'], color='navy', lw=2,
                label='Mean (Maintenance Only)')

        # Including Loading Dose (Dashed) - only if loading stats exist
        if 'loading' in stats:
            ax.plot(t_grid, stats['loading']['mean'], color='crimson', lw=2,
                    linestyle='--', label='Mean (With Loading Dose)')

        # 3. Clinical Target Overlays
        # Target Trough Range (8-12 mg/L)
        ax.axhspan(8, 12, color='green', alpha=0.3, label='Target Trough (8-12 mg/L)')

        # Toxicity Threshold
        ax.axhline(50, color='red', linestyle=':', alpha=0.6, label='Toxicity Threshold (50 mg/L)')

        # 4. OPTIONAL: Measured Data Point (For Bayesian Refinement)
        # This is where the NameError was occurring; arguments must be in the 'def' line above.
        if measured_level is not None and level_time is not None:
            ax.scatter(level_time, measured_level, color='red', marker='X', s=120,
                       edgecolor='black', label=f'Measured Level: {measured_level} mg/L', zorder=5)
            # Vertical drop-line for the lab draw time
            ax.axvline(level_time, color='red', linestyle='--', alpha=0.3)

        # 5. Formatting
        ax.set_title("Concentration Projection", fontsize=14, fontweight='bold')
        ax.set_xlabel("Hours post initiation", fontsize=12)
        ax.set_ylabel("Concentration (mg/L)", fontsize=12)
        ax.set_ylim(0, max(60, stats['standard']['high_95'].max() + 10))  # Dynamic scaling
        ax.legend(loc='upper right', fontsize='small', frameon=True)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        return fig

    def plot_risk_histogram(self, auc_samples):
        """
        Displays the distribution of predicted AUCs with unified scaling
        and a figsize tuned to match the height of the trajectory plot.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Height tuned to 7 inches to match the projection plot's vertical presence
        fig, ax = plt.subplots(figsize=(7, 7))

        bin_width = 25
        min_plot, max_plot = 0, 1200
        bins = np.arange(min_plot, max_plot + bin_width, bin_width)

        sub = auc_samples[auc_samples < 400]
        ther = auc_samples[(auc_samples >= 400) & (auc_samples <= 600)]
        tox = auc_samples[auc_samples > 600]

        ax.hist(sub, bins=bins, color='#FF9933', alpha=0.8,
                label=f'Sub (<400): {len(sub) / len(auc_samples):.1%}',
                edgecolor='white', linewidth=0.5)

        ax.hist(ther, bins=bins, color='#2ca02c', alpha=0.8,
                label=f'Target (400-600): {len(ther) / len(auc_samples):.1%}',
                edgecolor='white', linewidth=0.5)

        ax.hist(tox, bins=bins, color='#d62728', alpha=0.8,
                label=f'Toxic (>600): {len(tox) / len(auc_samples):.1%}',
                edgecolor='white', linewidth=0.5)

        ax.axvline(400, color='black', linestyle='-', lw=2)
        ax.axvline(600, color='black', linestyle='-', lw=2)

        ax.set_title("Predicted AUC24 Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Steady-State AUC24", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)

        ax.set_xlim(200, 1000)
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_clinical_trajectory(self, doses, labs, recommendation, n_sims=1000):
        import matplotlib.pyplot as plt
        import numpy as np

        if not self.calibrated:
            raise ValueError("Engine must be calibrated to plot.")

        # 1. Setup Timelines
        last_hist_dose = max([d['start'] for d in doses])
        new_interval = recommendation['interval_hrs']
        # The washout period ends when the first new dose starts
        first_new_dose_time = last_hist_dose + new_interval

        num_future = 8
        future_doses = []
        for i in range(num_future):
            future_doses.append({
                'dose': recommendation['maint_mg'],
                'start': first_new_dose_time + (i * new_interval),
                't_inf': 1.0 if recommendation['maint_mg'] < 1000 else 1.5
            })

        full_timeline = doses + future_doses
        t_max = first_new_dose_time + (num_future * new_interval)
        t_plot = np.linspace(0, t_max, 3000)

        # 2. Simulations
        # Individual Median Fit (History + Future)
        post_cc_raw, _ = self._solve_trajectory(self.map_log_params, t_plot, full_timeline)
        median_cc = post_cc_raw.flatten()

        # Uncertainty Bands (Monte Carlo)
        samples = np.random.multivariate_normal(self.map_log_params, self.full_covariance, n_sims)
        mc_res, _ = self._solve_trajectory(samples, t_plot, full_timeline)

        conc_array = np.array(mc_res)
        if conc_array.ndim == 3: conc_array = conc_array[0]

        p2_5 = np.percentile(conc_array, 2.5, axis=0)
        p25 = np.percentile(conc_array, 25, axis=0)
        p75 = np.percentile(conc_array, 75, axis=0)
        p97_5 = np.percentile(conc_array, 97.5, axis=0)

        # 3. Plotting
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        hist_mask = t_plot <= first_new_dose_time
        proj_mask = t_plot >= first_new_dose_time

        # A. Median Curves (Solid for History, Dashed for Future)
        ax.plot(t_plot[hist_mask], median_cc[hist_mask], color='#2980b9', lw=3, label='Individual Fit')
        ax.plot(t_plot[proj_mask], median_cc[proj_mask], color='#2980b9', lw=3, ls='--', label='Projected SS')

        # B. Shaded Intervals (Projection Only)
        ax.fill_between(t_plot[proj_mask], p2_5[proj_mask], p97_5[proj_mask],
                        color='#3498db', alpha=0.1, label='95% Prediction Interval')
        ax.fill_between(t_plot[proj_mask], p25[proj_mask], p75[proj_mask],
                        color='#3498db', alpha=0.25, label='IQR (25-75%)')

        # C. Measured Labs
        ax.scatter([l['draw_time'] for l in labs], [l['val'] for l in labs],
                   color='#e74c3c', s=120, edgecolors='white', label='Measured Labs', zorder=5)

        # D. Dose Markers
        # History (Green)
        for d in doses:
            ax.axvline(x=d['start'], color='#27ae60', ls='--', alpha=0.3, lw=1.5)
            ax.text(d['start'] + 0.5, 2, f"{int(d['dose'])}mg", color='#27ae60',
                    fontweight='bold', fontsize=8, alpha=0.8)

        # Proposed Plan (Blue)
        for d in future_doses:
            ax.axvline(x=d['start'], color='#2980b9', ls='--', alpha=0.3, lw=1.5)
            ax.text(d['start'] + 0.5, 2, f"Plan: {int(d['dose'])}mg", color='#2980b9',
                    fontweight='bold', fontsize=8, alpha=0.8)

        # E. Target Range
        ax.axhspan(self.trough_min, self.trough_max, color='#2ecc71', alpha=0.08, label='Target Trough')

        # Aesthetics
        ax.set_title(f"Bayesian Individualization & Transition: {recommendation['maint_mg']}mg q{new_interval}h",
                     fontsize=14)
        ax.set_ylabel("Vancomycin Concentration (mg/L)")
        ax.set_xlabel("Hours Since Start")
        ax.legend(loc='upper right', ncol=1, fontsize='small')  # Changed to single column for cleaner look
        ax.grid(alpha=0.1)

        return fig

