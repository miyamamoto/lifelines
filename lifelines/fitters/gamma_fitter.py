# -*- coding: utf-8 -*-
import autograd.numpy as np
from autograd.numpy import exp, log
from scipy.special import gammainccinv, gammaincinv
from autograd_gamma import gammaincc, gammainc, gammaln, gammainccln, gammaincln
from lifelines.fitters import KnownModelParametricUnivariateFitter
from lifelines.utils import CensoringType
from lifelines.utils.safe_exp import safe_exp

# class GeneralizedGammaFitter(KnownModelParametricUnivariateFitter):  # 変更前
class GammaFitter(KnownModelParametricUnivariateFitter):               # 変更後
    （中略）
    # _fitted_parameter_names = ["mu_", "ln_sigma_", "lambda_"]        # 変更前
    # _bounds = [(None, None), (None, None), (None, None)]             # 変更前
    # _compare_to_values = np.array([0.0, 0.0, 1.0])                   # 変更前
    _fitted_parameter_names = ["mu_", "lambda_"]                       # 変更後
    _bounds = [(None, None), (None, None)]                             # 変更後
    _compare_to_values = np.array([0.0, 1.0])                          # 変更後

    _scipy_fit_options = {"ftol": 1e-7}

    def _create_initial_point(self, Ts, E, *args):
        if CensoringType.is_right_censoring(self):
            log_data = log(Ts[0])
        elif CensoringType.is_left_censoring(self):
            log_data = log(Ts[1])
        elif CensoringType.is_interval_censoring(self):
            # this fails if Ts[1] == Ts[0], so we add a some fudge factors.
            log_data = log(Ts[1] - Ts[0] + 0.01)
        # return np.array([log_data.mean(), log(log_data.std() + 0.01), 0.1])  # 変更前
        return np.array([log_data.mean(), 0.1])  # 変更後

    def _survival_function(self, params, times):
        # mu_, ln_sigma_, lambda_ = params  # 変更前
        mu_, lambda_ = params               # 変更後
        # sigma_ = safe_exp(ln_sigma_)      # 削除
        # Z = (log(times) - mu_) / sigma_   # 変更前
        Z = (log(times) - mu_) / lambda_    # 変更後
        if lambda_ > 0:
            return gammaincc(1 / lambda_ ** 2, safe_exp(lambda_ * Z - 2 * np.log(lambda_)))
        else:
            return gammainc(1 / lambda_ ** 2, safe_exp(lambda_ * Z - 2 * np.log(-lambda_)))

    def _cumulative_hazard(self, params, times):
        # mu_, ln_sigma_, lambda_ = params  # 変更前
        mu_, lambda_ = params               # 変更後
        # sigma_ = safe_exp(ln_sigma_)      # 削除
        # Z = (log(times) - mu_) / sigma_   # 変更前
        Z = (log(times) - mu_) / lambda_    # 変更後
        ilambda_2 = 1 / lambda_ ** 2
        if lambda_ > 0:
            v = -gammainccln(ilambda_2, safe_exp(lambda_ * Z - 2 * np.log(lambda_)))
        else:
            v = -gammaincln(ilambda_2, safe_exp(lambda_ * Z - 2 * np.log(-lambda_)))
        return v

    def _log_1m_sf(self, params, times):
        # mu_, ln_sigma_, lambda_ = params # 変更前
        mu_, lambda_ = params              # 変更後
        # sigma_ = safe_exp(ln_sigma_)     # 削除
        # Z = (log(times) - mu_) / sigma_  # 変更前
        Z = (log(times) - mu_) / lambda_   # 変更後
        if lambda_ > 0:
            v = gammaincln(1 / lambda_ ** 2, safe_exp(lambda_ * Z - 2 * np.log(lambda_)))
        else:
            v = gammainccln(1 / lambda_ ** 2, safe_exp(lambda_ * Z - 2 * np.log(-lambda_)))
        return v

    def _log_hazard(self, params, times):
        # mu_, ln_sigma_, lambda_ = params  # 変更前
        mu_, lambda_ = params               # 変更後
        ilambda_2 = 1 / lambda_ ** 2
        # Z = (log(times) - mu_) / safe_exp(ln_sigma_)  # 変更前
        Z = (log(times) - mu_) / lambda_                # 変更後
        if lambda_ > 0:
            v = (
                # log(lambda_)  # 削除
                - log(times)
                # - ln_sigma_   # 削除
                - gammaln(ilambda_2)
                + (lambda_ * Z - safe_exp(lambda_ * Z) - 2 * log(lambda_)) * ilambda_2
                - gammainccln(ilambda_2, safe_exp(lambda_ * Z - 2 * np.log(lambda_)))
            )
        else:
            v = (
                # log(-lambda_)  # 削除
                - log(times)
                # - ln_sigma_    # 削除
                - gammaln(ilambda_2)
                + (lambda_ * Z - safe_exp(lambda_ * Z) - 2 * log(-lambda_)) * ilambda_2
                - gammaincln(ilambda_2, safe_exp(lambda_ * Z - 2 * np.log(-lambda_)))
            )
        return v

    def percentile(self, p):
        lambda_ = self.lambda_
        # sigma_ = exp(self.ln_sigma_)  # 削除
        if lambda_ > 0:
            # 変更前↓
            # return exp(sigma_ * log(gammainccinv(1 / lambda_ ** 2, p) * lambda_ ** 2) / lambda_) * exp(self.mu_)
            # 変更後↓
            return exp(lambda_ * log(gammainccinv(1 / lambda_ ** 2, p) * lambda_ ** 2) / lambda_) * exp(self.mu_)
        # 変更前↓
        # return exp(sigma_ * log(gammaincinv(1 / lambda_ ** 2, p) * lambda_ ** 2) / lambda_) * exp(self.mu_)
        # 変更後↓
        return exp(lambda_ * log(gammaincinv(1 / lambda_ ** 2, p) * lambda_ ** 2) / lambda_) * exp(self.mu_)