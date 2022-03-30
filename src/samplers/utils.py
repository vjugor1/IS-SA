from scipy import stats
import numpy as np
from src.samplers.importance_sampler import *


def get_samples_SAIMIN(N: int, eta: float, J: int, A: np.ndarray):
    """Yield samples outside of "useless samples boundary"
        One has inequalities
        A chi + Gamma x <= Beta
        to be satisfied with probability 1 - eta
        "useless samples boundary" is defined as
        A chi <= - Phi^-1 (eta)

    Args:
        N (int): Number of samples
        eta (float): reliability level
        J (int): number of planes in feasibility set
        A (np.ndarray): A from docs

    Returns:
        np.ndarray: SAIMIN samples - outside of "useless samples boundary"
    """
    Phi_inv = stats.norm.ppf(eta)
    Beta_P = np.ones(J) * (-Phi_inv)
    sampler = ConditionedPolytopeGaussianSampler(A, Beta_P)
    generator = sampler.sample()
    samples_SAIMIN = np.array([next(generator) for s in range(N)])
    return samples_SAIMIN


def check_feasibility_out_of_sample(
    x: np.ndarray, Gamma: np.ndarray, Beta: np.ndarray, A: np.ndarray, N: int = 1000
) -> np.float:
    """AI is creating summary for check_feasibility_out_of_sample

    Args:
        x (np.ndarray): Current control variable value
        Gamma (np.ndarray): Plane normals for inequalities
        Beta (np.ndarray): Plane constants for inequalities
        A (np.ndarray): Plane normals for inequalities - standard gaussian component
        N (int, optional): Number of samples. Defaults to 1000.

    Returns:
        np.float: Probability of `x` being feasible estimate
    """
    # Sample from nominal
    samples = np.random.multivariate_normal(
        np.zeros(A.shape[1]), np.eye(A.shape[1]), size=N
    )
    # Assess deterministic feasibility of x
    feas_det = Gamma.dot(x) - Beta  # (J,)
    # Multiply each sample by A matrix
    A_dot_samples = A.dot(samples.T)  # (J, N)
    # Tile `feas_det` to add to A_dot_sample in one operation
    feas_det_tiled = np.tile((feas_det), N).reshape(-1, N)  # (J, N)
    # Add up, compare to zero and check if all satisfied to samples
    sample_res = ((feas_det_tiled + A_dot_samples) <= 0.0).all(axis=0)
    prob_estimate = sample_res.sum() / N

    return prob_estimate
