import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma as dgamma


def gauss_pdf(X, u, sigma):
    """Calculate Gaussian probability density function values for input array X.

    Args:
        X: Input array of shape (n,) representing scores.
        u: Mean (mu) of the Gaussian distribution.
        sigma: Standard deviation (sigma) of the Gaussian distribution.

    Returns:
        Array of PDF values with same shape as X.
    """
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * np.square((X - u) / sigma))


def gamma_pdf(X, u, sigma):
    """Calculate Gamma probability density function values using mean and std.

    The shape and scale parameters are derived from mean (u) and std (sigma).

    Args:
        X: Input array of shape (n,) representing scores.
        u: Mean of the distribution.
        sigma: Standard deviation of the distribution.

    Returns:
        Array of Gamma PDF values with same shape as X.
    """
    a = (u / sigma) ** 2
    scale = sigma**2 / u
    return dgamma.pdf(X, a=a, scale=scale)


BIC_Max = 1e100
Div_Zero = 1e-50
Max_Iter = 30


class TDA_fmm:
    """Finite Mixture Model (FMM) for Target-Decoy Analysis (TDA).

    This class estimates score distributions using a mixture of Gaussians.
    It supports modeling both target and decoy distributions, where the decoy
    model can be incorporated as an external component in the target model.

    Attributes:
        n_components: Number of Gaussian components in the mixture.
        external_model: Optional fitted decoy model (for target modeling).
        max_iter: Maximum number of EM iterations.
        main_pdf: PDF function used for the first component.
        helper_pdf: PDF function used for other components.
        weights: Learned mixture weights (pi_k).
        mu: Learned means for each component.
        sigma: Learned standard deviations for each component.
    """
    def __init__(self, n_components, external_model=None):
        """Initializes the TDA_fmm model.

        Args:
            n_components: Number of Gaussian components in the mixture.
            external_model: Pre-fitted decoy model. If None, models decoy;
                           if provided, models target with decoy as a component.
        """
        self.n_components = n_components
        self.external_model = external_model
        self.max_iter = Max_Iter
        if external_model is None:
            self.main_pdf = gauss_pdf
        else:
            self.main_pdf = gauss_pdf
        self.helper_pdf = gauss_pdf

    def __call__(self, X):
        return self.pdf_mix(X)

    def pep(self, X, external_pdf=None):
        """Estimates Posterior Error Probabilities (PEP).

        PEP = pi0 * f_decoy(x) / f_mixture(x)

        Args:
            X: Input scores.
            external_pdf: Optional precomputed PDF values from external model.
                          If None and external_model exists, it will be computed.

        Returns:
            Array of PEP values for each score in X. Returns zeros if model not fitted.
        """
        if self.weights is None or self.external_model is None:
            return np.zeros(len(X))
        if external_pdf is None:
            external_pdf = self.external_model.pdf(X)
        return self.weights[-1] * external_pdf / self.pdf_mix(X, external_pdf)

    def get_pi0(self):
        """Returns the estimated proportion of decoy (null) components in the mixture.

        Returns:
            pi0 value (between 0 and 1). Returns 0 if model not fitted or no external model.
        """
        if self.weights is None or self.external_model is None:
            return 0
        else:
            return self.weights[-1]

    def pdf(self, X):
        """Computes the PDF of the main mixture components (excluding external model).

        Args:
            X: Input scores of shape (n,).

        Returns:
            PDF values of shape (n,). Returns zeros if model not fitted.
        """
        if self.weights is None:
            return np.zeros(len(X))
        X = np.array(X)
        pdf = np.zeros((self.n_components, X.shape[0]))
        for i in range(0, self.n_components):
            if i == 0:
                pdf[i, :] = self.main_pdf(X, u=self.mu[i], sigma=self.sigma[i])
            else:
                pdf[i, :] = self.helper_pdf(X, u=self.mu[i], sigma=self.sigma[i])
        return np.dot(self.weights[: self.n_components], pdf)

    def pdf_mix(self, X, external_pdf=None):
        """Computes the full mixture PDF, including external model if present.

        f_mixture(x) = pi0 * f_decoy(x) + (1-pi0) * f_target(x)

        Args:
            X: Input scores.
            external_pdf: Optional precomputed PDF values from external model.

        Returns:
            Mixture PDF values. Returns zeros if model not fitted.
        """
        if self.weights is None:
            return np.zeros(len(X))
        X = np.array(X)
        if self.external_model is not None:
            if external_pdf is None:
                external_pdf = self.external_model.pdf(X)
            pdf0 = external_pdf * self.get_pi0()
            pdf1 = self.pdf(X) * (1 - self.get_pi0())
            return pdf0 + pdf1
        else:
            return self.pdf(X)

    def loglik_BIC(self, X):
        """Computes log-likelihood and Bayesian Information Criterion (BIC).

        BIC = -2 * loglik + num_params * log(n)

        Args:
            X: Input scores.

        Returns:
            A tuple of (log-likelihood, BIC). Returns (0, 0) if model not fitted.
        """
        if self.weights is None:
            return 0, 0
        _has_external = 1 if self.external_model is not None else 0
        loglik = np.sum(np.log(self.pdf(X)))
        BIC = -2 * loglik + (self.n_components + _has_external) * np.log(len(X))
        return loglik, BIC

    def plot(self, title, plot_scores, false_scores=None):
        """Plots the fitted mixture model against histogram of scores.

        If an external model exists and false_scores are provided, plots:
          - Decoy model (external)
          - Target histogram + mixture fit
          - Separated true and false components

        Otherwise, plots only the decoy model fit.

        Args:
            title: Title prefix for plots.
            plot_scores: Scores to plot (e.g., target scores).
            false_scores: Optional decoy scores for comparison.
        """
        binsize = 40
        lw = 1

        if self.external_model is not None and false_scores is not None:
            pass
            self.external_model.plot(title, false_scores)
            scores = np.array(plot_scores)
            binsize *= 2

            step = 0.001
            plt.figure()
            pp = plt.subplot()

            n, bins, patches = pp.hist(
                scores, binsize, density=1, facecolor="green", alpha=0.75
            )
            bins = np.arange(scores.min(), scores.max(), step)
            dis = self.pdf_mix(bins)
            ratio = np.mean(n) / np.mean(dis)
            dis = dis * ratio
            pp.plot(bins, dis, "r--", linewidth=lw)
            pp.set_title(title + " target")

            # plot false and true distribution separately
            plt.figure()
            pp = plt.subplot()
            dis_false = self.external_model.pdf(bins)
            dis_true = self.pdf(bins)
            pp.hist(scores, binsize, density=1, facecolor="green", alpha=0.75)
            dis_false = dis_false * ratio * self.get_pi0()
            dis_true = dis_true * ratio * (1 - self.get_pi0())
            pp.plot(bins, dis_true, "b--", linewidth=lw)
            pp.plot(bins, dis_false, "r--", linewidth=lw)
            pp.set_title(
                title
                + rf" target mixture ($\pi_0$={self.get_pi0():.2f}, $\pi_1$={1 - self.get_pi0():.2f})"
            )

        else:
            scores = np.array(plot_scores)

            step = 0.1
            plt.figure()
            pp = plt.subplot()

            n, bins, patches = pp.hist(
                scores, binsize, density=1, facecolor="blue", alpha=0.75
            )
            bins = np.arange(scores.min(), scores.max(), step)
            dis = self.pdf(bins)
            dis = dis * np.mean(n) / np.mean(dis)
            pp.plot(bins, dis, "r--", linewidth=lw)
            pp.set_title(title + " decoy")

    def fit(self, X):
        """Fits the FMM model using Expectation-Maximization (EM) algorithm.

        Args:
            X: Input scores to fit the model on.
        """
        if len(X) < 10:
            self.weights = None
            return
        X = np.array(X)
        self.mu = np.min(X) + (np.max(X) - np.min(X)) / (
            self.n_components + 1
        ) * np.array(range(1, self.n_components + 1))
        self.sigma = np.ones(self.n_components) * np.var(X)

        if self.external_model is not None:
            d0 = self.external_model.pdf(X)
            _has_external = 1
        else:
            _has_external = 0

        ## weights is pi, i.e. component probabilities, including the probability of external_model.
        self.weights = np.ones(self.n_components + _has_external) / (
            self.n_components + _has_external
        )

        # Response of each component (including the external model),
        # the response: $p_{ik} = w_k*f_k(X_i) / sum(w.*f)$, where k is the kth component.
        post_prob = np.zeros((self.n_components + _has_external, X.shape[0]))

        for __iter in range(self.max_iter):
            ## E-step
            dens = np.zeros((self.n_components + _has_external, X.shape[0]))

            # The external pdf is in the last
            if self.external_model is not None:
                dens[-1, :] = d0

            # Estimate pdf value of X for each component.
            for i in range(0, self.n_components):
                if i == 0:
                    dens[i, :] = self.main_pdf(X, u=self.mu[i], sigma=self.sigma[i])
                else:
                    dens[i, :] = self.helper_pdf(X, u=self.mu[i], sigma=self.sigma[i])

            # Store the sum response, i.e. sum(w.*f), for each X_i
            total_prob = np.dot(self.weights, dens)
            # Calculate response value for each X_i
            for i in range(self.n_components + _has_external):
                post_prob[i, :] = self.weights[i] * dens[i, :] / total_prob

            ## M-step
            sum_prob = np.sum(post_prob, axis=1)
            sum_prob[sum_prob == 0] = 1e-12
            # sum all response (R_i) for X_i to estimate the weight of each component
            new_weights = sum_prob / np.sum(sum_prob)

            # New mean is the weighted mean of the responses.
            new_mu = (
                np.dot(post_prob[: self.mu.shape[0], :], X)
                / sum_prob[: self.mu.shape[0]]
            )
            new_sigma = self.sigma.copy()
            for i in range(self.n_components):
                # New std is the weighted sqrt(variance) of the response
                new_sigma[i] = np.sqrt(
                    np.dot(post_prob[i, :], np.square(X - self.mu[i])) / sum_prob[i]
                )

            # @Todo Check if the new parameters are legal (for example NaN value).
            if (
                np.any(np.isnan(new_sigma))
                or np.any(np.isnan(new_mu))
                or (np.any(np.isinf(new_sigma)) or np.any(np.isinf(new_mu)))
                or (
                    np.any(np.array(new_sigma) <= Div_Zero)
                    or np.any(np.array(new_mu) <= Div_Zero)
                )
            ):
                break

            self.weights = new_weights
            self.mu = new_mu
            self.sigma = new_sigma


class DecoyModel(TDA_fmm):
    """A simplified model for fitting decoy score distributions.

    Uses a single Gaussian, optionally filtering outliers using sigma threshold.
    """
    def __init__(self, gaussian_outlier_sigma, *args, **kwargs):
        """Initializes the decoy model.

        Args:
            gaussian_outlier_sigma: If provided, scores below (mu - sigma * gaussian_outlier_sigma)
                                    are filtered before fitting.
            *args, **kwargs: Ignored, for compatibility.
        """
        self.external_model = None
        self.weights = np.array([1.0])
        self.gaussian_outlier_sigma = gaussian_outlier_sigma

    def fit(self, X):
        """Fits a single Gaussian to the decoy scores.

        Optionally filters left-tail outliers before fitting.

        Args:
            X: Decoy scores.
        """
        self.mu = np.mean(X)
        self.sigma = np.std(X)

        if self.gaussian_outlier_sigma:
            X = X[self.mu - self.gaussian_outlier_sigma * self.sigma < X]
            self.mu = np.mean(X)
            self.sigma = np.std(X)

    def pdf(self, X):
        """Computes Gaussian PDF for given scores.

        Args:
            X: Input scores.

        Returns:
            PDF values.
        """
        self.data = X
        return gauss_pdf(X, self.mu, self.sigma)


def select_best_fmm(target_scores, decoy_fmm, _max_component_=3, verbose=True):
    """Selects the best TDA_fmm model by BIC criterion.

    Fits models with 1 to `_max_component_` components and selects the one with lowest BIC.

    Args:
        target_scores: Scores to fit the target model on.
        decoy_fmm: Pre-fitted decoy model.
        _max_component_: Maximum number of components to try.
        verbose: Whether to print progress.

    Returns:
        Best-fitted TDA_fmm model (target model).
    """
    best_target_BIC = BIC_Max
    for n_component in range(1, _max_component_ + 1):
        target_fmm = TDA_fmm(n_component, external_model=decoy_fmm)
        target_fmm.fit(target_scores)
        log_lik, BIC = target_fmm.loglik_BIC(target_scores)
        if best_target_BIC > BIC:
            best_target_BIC = BIC
            best_fmm = target_fmm
        if verbose:
            print(
                f"[Target FMM] G={n_component:d}, BIC={BIC:f}, best BIC={best_target_BIC:f}"
            )
    return best_fmm
