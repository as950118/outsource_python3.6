import numpy as np
import scipy.stats as stats
#import arviz as az
def posterior_grid_approx(grid_points=100, success=6, tosses=9):

    # define grid
    p_grid = np.linspace(0, 1, grid_points)

    # define prior
    prior = np.repeat(1, grid_points)

    # compute likelihood at each point in the grid
    likelihood = stats.binom.pmf(success, tosses, p_grid)

    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior / unstd_posterior.sum()

    return p_grid, posterior
p_grid, posterior = posterior_grid_approx(grid_points=100, success=2, tosses=10)
np.random.seed(100)
samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)
print(posterior)