#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:50:39 2022

@author: zamankhani
"""

import pymc3 as pm
import arviz as az

from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pylab as plt

import seaborn as sns
import collections
import numpy as np
import pandas as pd
import theano.tensor as tt


sns.set_style(
    style='darkgrid', 
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100




df = pd.read_csv(
    '/home/zamankhani/Desktop/Data4Saba/FLX_FR-Pue_FLUXNET2015_SUBSET_HH_2000-2014_2-4.csv')

df[['TIMESTAMP_START']] = (df[['TIMESTAMP_START']].applymap(str).applymap(
    lambda s: "{}/{}/{} {}:{}".format(s[0:4], s[4:6], s[6:8], s[8:10], s[10:12])))

df['TIMESTAMP_START'] = df['TIMESTAMP_START'].astype('datetime64[ns]')

mask = (df['TIMESTAMP_START'] > '2013-01-01') & (df['TIMESTAMP_START'] <= '2013-02-26')
ts1 = df.loc[mask]

ts1.set_index(['TIMESTAMP_START'], inplace=True)

ts1=ts1.resample('1H').mean()
result = seasonal_decompose(ts1['TA_F'], model='additive')
result.plot()

ts1.reset_index(inplace=True)


def plot_components(dates,
                    component_means_dict,
                    component_stddevs_dict,
                    x_locator=None,
                    x_formatter=None):
    colors = sns.color_palette()
    c1, c2 = colors[0], colors[1]

    axes_dict = collections.OrderedDict()
    num_components = len(component_means_dict)
    fig = plt.figure(figsize=(12, 2.5 * num_components))
    for i, component_name in enumerate(component_means_dict.keys()):
        component_mean = component_means_dict[component_name]
        component_stddev = component_stddevs_dict[component_name]

    ax = fig.add_subplot(num_components,1,1+i)
    ax.plot(dates, component_mean, lw=2)
    ax.fill_between(dates,
                     component_mean-2*component_stddev,
                     component_mean+2*component_stddev,
                     color=c2, alpha=0.5)
    ax.set_title(component_name)
    if x_locator is not None:
        ax.xaxis.set_major_locator(x_locator)
        ax.xaxis.set_major_formatter(x_formatter)
    axes_dict[component_name] = ax
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, axes_dict




def norm(data):
    norm_data = (data-data[0])/np.std(data)
    return norm_data


#ts1['TA_F'] = (ts1['TA_F'])
#ts1['RECO_NT_VUT_50'] = norm(ts1['RECO_NT_VUT_50'])
#ts['RECO_NT_VUT_REF'] = norm(ts['RECO_NT_VUT_REF'])
#ts1['SW_IN_F'] = norm(ts1['SW_IN_F'])
#ts = pd.DataFrame(ts1.loc[:,['TIMESTAMP_START', 'TA_F','RECO_NT_VUT_50']])
ts = pd.DataFrame(ts1.loc[:,['TIMESTAMP_START', 'TA_F','RECO_NT_VUT_50']])

colors = sns.color_palette()
c1, c2, c3 = colors[0], colors[1], colors[2]

fig = plt.figure(figsize=(18, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(ts['TIMESTAMP_START'],
        ts['TA_F'], lw=2, c=c1)
ax.set_ylabel("reco")


first_reco = ts['TA_F'][0]
std_reco = np.std(ts['TA_F'])
#y = ts['RECO_NT_VUT_50'].values
#x = ts['TIMESTAMP_START'].values
num_forecast = 24 * 7 *2  # two weeks
data_training = ts[:-num_forecast]
data_test = ts[-num_forecast:]
#x_test = x[-num_forecast:]
#y_test = y[-num_forecast:]


#ax = fig.add_subplot(2, 1, 2)
#ax.plot(data_training['TIMESTAMP_START'],
#        data_training['RECO_NT_VUT_50'], lw=2, c=c2)
#ax.set_ylabel('RECO_NT_VUT_50')
##ax.set_title("RECO_NT_VUT_50")
#fig.suptitle("RECO_NT_VUT_50", fontsize=11)

#ax = fig.add_subplot(3, 1, 3)
#ax.plot(data_training['TIMESTAMP_START'],
#        data_training['SW_IN_F'], lw=2, label="training SW_IN_F", c=c3)
#ax.set_ylabel('SW_IN_F')
#ax.set_title('SW_IN_F')
#fig.suptitle('SW_IN_F', fontsize=11)

plt.show()

#
# plot the priors
#
x = np.linspace(0, 10, 1000)
priors = [
    ("ℓ_pdecay",  pm.Gamma.dist(alpha=4,  beta=0.5)),
    ("ℓ_psmooth", pm.Gamma.dist(alpha=4,  beta=3)),
    ("period",    pm.Normal.dist(mu=1,  sigma=5)),
    ("ℓ_med",     pm.Gamma.dist(alpha=1,  beta=0.75)),
    ("α",         pm.Gamma.dist(alpha=5,  beta=2)),
    ("ℓ_trend",   pm.Gamma.dist(alpha=4,  beta=0.1)),
    ("ℓ_noise",   pm.Gamma.dist(alpha=4,  beta=4))]

for i, prior in enumerate(priors):
    plt.plot(x, np.exp(prior[1].logp(x).eval()), label=prior[0])
plt.legend(loc="upper right")
plt.xlabel("time")
plt.show();

x = np.linspace(0, 10, 1000)
priors = [
    ("η_per",   pm.HalfCauchy.dist(beta=2)),
    ("η_med",   pm.HalfCauchy.dist(beta=1.0)),
    ("η_trend", pm.HalfCauchy.dist(beta=3)), 
    ("σ",       pm.HalfNormal.dist(sigma=0.25)),
    ("η_noise", pm.HalfNormal.dist(sigma=0.5))]

for i, prior in enumerate(priors):
    plt.plot(x, np.exp(prior[1].logp(x).eval()), label=prior[0])
plt.legend(loc="upper right")
plt.xlabel("time")
plt.show();


def dates_to_idx(data):
    
    t = (np.arange(len(data))/len(data)) 
    return t
   


t = dates_to_idx(data_training['TIMESTAMP_START'])[:,None]
y = data_training['TA_F'].values
#y = np.expand_dims(y, axis=0)
#
# define and fit the model
#

with pm.Model() as model:
   
    # yearly periodic component x long term trend
    η_per = pm.HalfCauchy("η_per", beta=2, testval=1.0)
    ℓ_pdecay = pm.Gamma("ℓ_pdecay", alpha=10, beta=0.075)
    period = pm.Normal("period", mu=1, sigma=0.05)
    ℓ_psmooth = pm.Gamma("ℓ_psmooth ", alpha=4, beta=3)
    cov_seasonal = (
        η_per ** 2 * pm.gp.cov.Periodic(1, period, ℓ_psmooth) * pm.gp.cov.Matern52(1, ℓ_pdecay)
    )
    gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)

    # small/medium term irregularities
    η_med = pm.HalfCauchy("η_med", beta=0.5, testval=0.1)
    ℓ_med = pm.Gamma("ℓ_med", alpha=2, beta=0.75)
    α = pm.Gamma("α", alpha=5, beta=2)
    cov_medium = η_med ** 2 * pm.gp.cov.RatQuad(1, ℓ_med, α)
    gp_medium = pm.gp.Marginal(cov_func=cov_medium)

    # long term trend
    η_trend = pm.HalfCauchy("η_trend", beta=2, testval=2.0)
    ℓ_trend = pm.Gamma("ℓ_trend", alpha=4, beta=0.1)
    cov_trend = η_trend ** 2 * pm.gp.cov.ExpQuad(1, ℓ_trend)
    gp_trend = pm.gp.Marginal(cov_func=cov_trend)

    # noise model
    η_noise = pm.HalfNormal("η_noise", sigma=0.5, testval=0.05)
    ℓ_noise = pm.Gamma("ℓ_noise", alpha=2, beta=4)
    σ = pm.HalfNormal("σ", sigma=0.25, testval=0.05)
    cov_noise = η_noise ** 2 * pm.gp.cov.Matern32(1, ℓ_noise) + pm.gp.cov.WhiteNoise(σ)

    # The Gaussian process is a sum of these three components
    gp = gp_seasonal + gp_medium + gp_trend


    # Since the normal noise model and the GP are conjugates, we use `Marginal` with the `.marginal_likelihood` method
    y_ = gp.marginal_likelihood("y", X=t, y=y, noise=cov_noise)
    

    #optimizer to find the MAP
    #mp = pm.sample(draws=2000, chains=3, tune=500)
    
    mp=pm.sample(draws=200, chains=4, tune=200)                    
                     
    #mp = pm.find_MAP(include_transformed=True)
    summary = pm.summary(mp)
    print(summary)
    
    
    # fitted model parameters
    a=sorted([name + ":" + str(mp[name]) for name in mp.keys() if not name.endswith("_")])
    print(a)
    

    #pm.plot_trace(mp)
    #dates = pd.date_range(start='2013-02-12', end="2013-02-26", freq="1H")[:-1]
    tnew = dates_to_idx(data_test)[:,None]
    
    
    #first_y = 0
    #std_y = 1
    #mu_pred, cov_pred = gp.predict(tnew, point=mp)
    #mean_pred = mu_pred * std_reco + first_reco
    #var_pred = cov_pred * std_reco ** 2
    
    print("Sampling gp predictions ...")
    mu_pred, cov_pred = gp.predict(tnew, point=mp)
    samples = pm.MvNormal.dist(mu=mu_pred, cov=cov_pred, shape=(len(data_test))).random()
    samples = samples * std_reco + first_reco
        
        # make dataframe to store fit results
    fit = pd.DataFrame(
        {"t": tnew.flatten(), "mu_total": mu_pred, "sd_total": np.sqrt(np.diag(cov_pred))},
        index=data_test['TIMESTAMP_START'],
    )
    
    print("Predicting with gp_trend ...")
    mu, var = gp_seasonal.predict(
        tnew, point=mp, given={"gp": gp, "X": t, "y": y, "noise": cov_noise}, diag=True
    )
    fit = fit.assign(mu_trend=mu * std_reco + first_reco, sd_trend=np.sqrt(var * std_reco ** 2))
    
    print("Predicting with gp_medium ...")
    mu, var = gp_medium.predict(
        tnew, point=mp, given={"gp": gp, "X": t, "y": y, "noise": cov_noise}, diag=True
    )
    fit = fit.assign(mu_medium=mu * std_reco + first_reco, sd_medium=np.sqrt(var * std_reco ** 2))
    
    print("Predicting with gp_seasonal ...")
    mu, var = gp_trend.predict(
        tnew, point=mp, given={"gp": gp, "X": t, "y": y, "noise": cov_noise}, diag=True
    )
    fit = fit.assign(mu_seasonal=mu * std_reco + first_reco, sd_seasonal=np.sqrt(var * std_reco ** 2))
    print("Done")
    
    
    
    # draw samples, and rescale
    n_samples = 2000
    
    
    mu_pred_sc = mu_pred * std_reco + first_reco
    sd_pred_sc =  np.sqrt(np.diag(cov_pred) * std_reco ** 2) 
    print('Sampled Sucessfully')
    upper = mu_pred_sc + 2 * sd_pred_sc
    lower = mu_pred_sc - 2 * sd_pred_sc
    
    c = sns.color_palette()
    
        # total fit
    plt.plot(fit.index, fit.mu_total, linewidth=1, color=c[5], label="Total fit")
    plt.fill_between(fit.index, lower, upper, color=c[0], alpha=0.4)
    
    # trend
    plt.plot(fit.index, fit.mu_trend, linewidth=1, color=c[1], label="Long term trend")
    
    # medium
    plt.plot(fit.index, fit.mu_medium, linewidth=1, color=c[2], label="Medium range variation")
    
    # seasonal
    plt.plot(fit.index, fit.mu_seasonal, linewidth=1, color=c[3], label="Seasonal process")
    
    # true value
    plt.plot(data_test['TIMESTAMP_START'], data_test['TA_F'], linewidth=2, color=c[4], label="Observed data")
    
    plt.ylabel("Temperature")
    plt.title("Signal decomposition")
    plt.legend(loc="upper right")
    plt.show();
    
    #
    # plot separate components of the decomposition 
    #
    demand_component_means = {
        'Ground truth': data_test['TA_F'],
        'Total fit': fit.mu_total,
        'Trend': fit.mu_trend,
        'Medium': fit.mu_medium,
        'Seasonal': fit.mu_seasonal,
    }
    demand_component_stddevs = {
        'Ground truth': np.zeros(len(data_test['TA_F'])),
        'Total fit': fit.sd_total,
        'Trend': fit.sd_trend,
        'Medium': fit.sd_medium,
        'Seasonal': fit.sd_seasonal,
    }
    
    fig, axes = plot_components(
      data_test['TIMESTAMP_START'],
      demand_component_means,
      demand_component_stddevs,
      
      )
        
        
        
    
    
    c = sns.color_palette()
    plt.plot(data_test['TIMESTAMP_START'], mu_pred_sc, linewidth=0.5, color=c[0], label="Total fit")
    plt.fill_between(data_test['TIMESTAMP_START'], lower, upper, color=c[0], alpha=0.4)
    
    # some predictions
    idx = np.random.randint(0, samples.shape[0], 10)
    for i in idx:
        plt.plot(data_test['TIMESTAMP_START'], samples, color=c[0], alpha=0.5, linewidth=2)
    
    # true value
    plt.plot(data_test['TIMESTAMP_START'], data_test['TA_F'], linewidth=2, color=c[1], label="Observed data")
    
    plt.ylabel("taf")
    plt.title("taf forecast")
    plt.legend(loc="upper right")
    plt.show();



    #print("Sampling gp predictions...")['TIMESTAMP_START', 'TA_F','RECO_NT_VUT_50']
    
    # draw samples, and rescale
    #n_samples = 300
    #samples = pm.MvNormal.dist(mu=mu_pred, cov=cov_pred)
    #sampless = np.random.sample(samples)
    #samples = samples * std_y + first_y
    
    ### plot mean and 2σ region of total prediction
    #fig = plt.figure(figsize=(16, 6))
    
    # scale mean and var
    #mu_pred_sc = mu_pred * std_y + first_y
    #sd_pred_sc = np.sqrt(np.diag(cov_pred) * std_y**2 )
    #upper = mu_pred_sc + 2*sd_pred_sc
    #lower = mu_pred_sc - 2*sd_pred_sc
    
    #c = sns.color_palette()
    #plt.plot(dates, mu_pred_sc, linewidth=2, color=c[0], label="Total fit")
    #plt.fill_between(data_test.date, lower, upper, color=c[0], alpha=0.4)
    #pm.sample_posterior_predictive
    # some predictions
    #idx = np.random.randint(0, samples.shape[0], 10)
    #for i in idx:
    #    plt.plot(data_test.date, samples[i,:], color=c[0], alpha=0.5, linewidth=0.5)
    
    # true value
   # plt.plot(data_test.date, data_test.demand, linewidth=2, color=c[1], label="Observed data")
    
   # plt.ylabel("Demand")
   # plt.title("Demand forecast")
   # plt.legend(loc="upper right")
   # plt.show();
   
       # predict at a 1 hour granularity
       #dates = pd.date_range(start='2013-02-12', end="2013-02-26", freq="1H")[:-1]
      
     
     #
   
     
     
    
    



















