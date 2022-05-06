import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import hilbert_gp as hgp


plt.rcParams['figure.figsize'] = (12, 5)
sns.set_context('notebook', font_scale=1.4)
sns.set_style('ticks')
def se_psd(chi, sigma, gamma):
    """
    Power spectral density 
    of squared exponential density
    using Angular freq
    """
    s = np.exp(-chi**2 /(4 * gamma))
    
    return s * sigma**2 * np.sqrt(np.pi / gamma)

def se_kernel(x1, x2, sigma, gamma):
    """
    Squared exponential kernel
    """
    tau = np.subtract.outer(x1, x2)**2
    return sigma**2 * np.exp(-gamma * tau )


def sm_kernel(x1, x2, w, gamma, mu):
    """
    Matern kernel
    """
    assert (len(w) == len(gamma) == len(mu))

    tau = np.subtract.outer(x1, x2)
    gram = np.zeros_like(tau)
    Q = len(w)
    for q in range(Q):
        gram += w[q] * np.exp(-gamma[q] * tau**2 / 2) * np.cos(mu[q] * tau)
    return gram

def sm_psd(xi, w, gamma, mu):
    Q = len(w)
    psd = np.zeros_like(xi)
    gamma = np.maximum(gamma, np.ones(Q)*1e-20)

    for q in range(Q):
        psd_1 = w[q] * np.exp(-(xi - mu[q])**2 / (2 * gamma[q]))
        psd_2 = w[q] * np.exp(-(-xi - mu[q])**2 / (2 * gamma[q]))
        psd += (psd_1 + psd_2) * 0.5 * np.sqrt(2 * np.pi) / np.sqrt(gamma[q])
    return psd

np.random.seed(123)
sigma_n = 2.0

t = np.linspace(-30, 30, 500)
s = np.linspace(0,10,50)
f = lambda t: 10.0 * (np.sin(np.pi*t/7.0)/(np.pi * t/7.0))

i_obs = np.random.choice(np.arange(len(s)), replace=False, size=50)

t_obs = s[i_obs]
y_obs = f(s)[i_obs] + np.random.normal(scale=sigma_n, size=len(t_obs))

plt.plot(t, f(t), '--k', label='Target')
plt.plot(t_obs, y_obs, '.', ms=7, alpha=0.8, mec='w', c='xkcd:strawberry', label='Obs')
plt.legend(ncol=2, fontsize=12)
plt.title('Observations')

# HilbertGP approx parameters
m = 250
L = 15

Q = 0.2
# mixture weights
# w = np.array([.5, .5])
w = np.random.random(2)

# mixture scales
# gamma = np.array([.01, .1]) 
gamma = np.random.random(2)

# mixture means
mu = np.array([5, 5])


sigma_noise = 2

kern_params = {'w':w, 'gamma':gamma, 'mu':mu}
# create model
model_sm = hgp.HilbertGP(
    t[i_obs],
    y_obs,
    kernel='SM',
    L=L,
    m=m,
    Q=Q,
    kern_params=kern_params,
    sigma_noise=sigma_noise)
model_sm.set_kernel_parameters(kern_params)

theta = np.r_[w, gamma, mu, sigma_n]

t_star = np.linspace(-35, 35, 1000)

mu_post, cov_post = model_sm.posterior(t_star)

plt.plot(t_star, f(t_star), '--k', label='Target', lw=2)
plt.plot(t_obs, y_obs, '.r', label='Obs', ms=7, alpha=0.6, mec='w', mew=0.5)
plt.plot(t_star, mu_post, c=sns.color_palette()[0], label='Prediction')
plt.fill_between(t_star,
                 mu_post + 2 * np.sqrt(cov_post),
                 mu_post - 2 * np.sqrt(cov_post),
                 color=sns.color_palette()[0],
                 alpha=0.4,
                 label='c.i.')
plt.legend(loc='upper center', ncol=4, fontsize=12)
plt.title('Untrained Hilbert-GP')

model_sm.fit(niter=1000, tol=1e-30)

mu_post, cov_post = model_sm.posterior(t_star)

plt.plot(t_star, f(t_star), '--k', label='Target', lw=2)
plt.plot(t_obs, y_obs, '.r', label='Obs', ms=7, alpha=0.6, mec='w', mew=0.5)
plt.plot(t_star, mu_post, c=sns.color_palette()[0], label='Pred')
plt.fill_between(t_star,
                 mu_post + 2 * np.sqrt(cov_post),
                 mu_post - 2 * np.sqrt(cov_post),
                 color=sns.color_palette()[0],
                 alpha=0.4,
                 label='c.i.')
plt.legend(loc='upper center', ncol=4, fontsize=12)
plt.title('Trained Hilbert-GP')