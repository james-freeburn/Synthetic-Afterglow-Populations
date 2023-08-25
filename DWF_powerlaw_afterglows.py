import numpy as np
import os
import afterglowpy as grb
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import dask
import dask.dataframe as dd
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import scipy.stats as stats

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx

def GRBFR(z):
    R = (0.00157 + 0.118*z)/(1+(z/3.23)**4.66)
    e = (1+z)**1.7
    return R*e

def GRB_spectrum(E,E_p):
    alpha = -1.0
    beta =  -2.3
    
    E_0 = E_p/(alpha+2.)
    if E < (alpha - beta)*E_0:
        return ((E/100.)**alpha)*np.exp(-E/E_0)
    else:
        return ((alpha-beta)*E_0/100.)**(alpha-beta)*np.exp(
            beta-alpha)*(E/100.)**beta

def GRB_params(theta_js,Gamma_0s,E_iso,E_ps,zs,d_Ls,band,rng,nevents):
    
    # Truncated log-normal distribution for GRB duration.
    mu3, sigma3 = 27.5,0.35
    T90 = [0.]*nevents
    for i in range(nevents):
        roll = stats.lognorm.rvs(sigma3, scale=mu3)
        while roll < 2.0:
            roll = stats.lognorm.rvs(sigma3, scale=mu3, size=nevents)
        T90[i] = roll
    
    # Calculating peak luminosity of GRB and bolometric flux of GRB
    Lpeak = 2.*E_iso*(1.+np.array(zs))/T90
    Pbol = Lpeak/(4*np.pi*d_Ls**2)
    Fbol = E_iso*(1.+np.array(zs))/(4*np.pi*d_Ls**2)

    # GRB Spectrum
    Es = np.geomspace(1e0,1e4,int(1e3)) #keV
    #spectrum = np.array([GRB_spectrum(E,100.) for E in Es])
    normalisation_factor = np.array([0.]*len(E_ps))
    integrations = np.array([0.]*len(E_ps))
    photon_integrations = np.array([0.]*len(E_ps))
    for i,E_p in enumerate(E_ps):
        spectrum = np.array([GRB_spectrum(E,E_p) for E in Es])
        SwiftRange = [find_nearest(Es,band[0]*(1+zs[i])),
                      find_nearest(Es,band[1]*(1+zs[i]))]
        # Integrate across Swift's wavelength range to get a flux and fluence.
        normalisation_factor[i] = np.trapz(spectrum*u.keV.to(u.erg)*Es,
                                           u.keV.to(u.erg)*Es)
        integrations[i] = np.trapz(spectrum[SwiftRange[0]:SwiftRange[1]]
                                   *u.keV.to(u.erg)*
                                   Es[SwiftRange[0]:SwiftRange[1]],
                                   u.keV.to(u.erg)*
                                   Es[SwiftRange[0]:SwiftRange[1]])
        # Convert flux to a photon flux.
        photon_integrations[i] = np.trapz(spectrum[SwiftRange[0]:
                                                   SwiftRange[1]],
                                          u.keV.to(u.erg)*
                                          Es[SwiftRange[0]:SwiftRange[1]])
        
    P = (Pbol/normalisation_factor)*integrations
    F = (Fbol/normalisation_factor)*integrations
    photon_flux = (Pbol/normalisation_factor)*photon_integrations
    # If photon flux is above 2.6, we get a Swift detection.
    Swift_GRB = photon_flux > 2.6
    
    return T90,F,P,photon_flux,Swift_GRB

def population_synthesis(rng,nevents,cosmo):
    
    # Intrinsic energy of GRBs
    E_gamma_dash = 1.5e48
    E_p_dash = 1.5
    
    # Lorentz factor distribution
    mu1,sigma1 = 4.525,1.475
    mu2,sigma2 = 1.742,0.916

    # The desired covariance matrix.
    cov = np.array([
            [sigma2, -.87],
            [-.87, sigma1]
        ])

    L = np.linalg.cholesky(cov)

    uncorrelated = [np.log(stats.lognorm.rvs(sigma2, 
                                             scale=np.exp(mu2), 
                                             size=nevents)),
                    np.log(stats.lognorm.rvs(sigma1, 
                                             scale=np.exp(mu1), 
                                             size=nevents))]
    correlated = np.dot(L, uncorrelated)

    uncorr_mean = [np.mean(uncorrelated[0]), np.mean(uncorrelated[1])]
    corr_mean = [np.mean(correlated[0]), np.mean(correlated[1])]
    correlated = correlated - np.array(
        corr_mean).reshape(2, 1) + np.array(
            uncorr_mean).reshape(2, 1)

    X, Y = correlated
    
    Gamma_0s = np.exp(Y)
    theta_js = np.exp(X)*np.pi/180.
    
    mask = (theta_js < np.pi/2.) & (Gamma_0s < 8e3) & (Gamma_0s > 1.)
    
    Gamma_0s = Gamma_0s[mask]
    theta_js = theta_js[mask]
    
    df = pd.DataFrame([])
    df = df.assign(Gamma_0=Gamma_0s,
                   theta_j=theta_js)
    tlength = rng.integers(20,200,len(df))
    tGRB = [rng.integers(-tlength_val + 3, 1440) 
                          for tlength_val in tlength]    
    df = df.assign(Beta_0 = (1.-(1./df.Gamma_0**2.))**0.5,
                   E_gamma = E_gamma_dash*df.Gamma_0,
                   tlength = tlength,
                   tGRB = tGRB,
                   n0 = rng.uniform(0.1,30.,len(df)),
                   b = rng.uniform(0.0,3.0,len(df)),
                   epsilon_B = [0.008]*len(df),
                   epsilon_e = [0.02]*len(df),
                   p = [2.3]*len(df),
                   theta_w = [rng.uniform(theta_j,np.pi/2) 
                              for theta_j in df.theta_j])
    
    # Generating probability density function for viewing angle.
    theta = np.linspace(0.0, np.pi/2, 100000)
    probability_density = np.sin(theta)
    probability_density = probability_density/np.sum(probability_density)
    
    df = df.assign(theta_v=[rng.choice(theta, p=probability_density) 
                            for theta_w in df.theta_w],
                   E_p = E_p_dash*5.*df.Gamma_0/(5.-2.*df.Beta_0),
                   Eiso = [E_gamma/(1-np.cos(theta_j)) 
                           if 1./Gamma_0 < np.sin(theta_j) 
                           else E_gamma*(1+Beta_0)*Gamma_0**2
                           for E_gamma,Gamma_0,Beta_0,theta_j 
                           in zip(df.E_gamma,df.Gamma_0,df.Beta_0,df.theta_j)])
    
    # Placing GRBs at redshifts based on comoving volumes and SFH.
    zs,d_Ls = GRB_redshifts(rng, len(df), cosmo)
    df = df.assign(z=zs,d_L=d_Ls)

    # Uniform distribution of theta_wing values (this is what we're testing)
    
    PO_mask = (df.theta_v < df.theta_j) | (np.sin(df.theta_v) < 1./df.Gamma_0)
    
    PO_events = df[PO_mask]
    NPO_events = df[PO_mask == False]
    
    T90,F,P,photon_flux,Swift_GRB = GRB_params(PO_events.theta_j,
                                               PO_events.Gamma_0,
                                               PO_events.Eiso,
                                               PO_events.E_p,
                                               PO_events.z,
                                               PO_events.d_L,
                                               [15.,150.],
                                               rng,len(PO_events))
    
    PO_events = PO_events.assign(T90=T90,
                                 GRB_fluence=F,
                                 GRB_flux=P,
                                 GRB_photon_flux=photon_flux,
                                 Swift_GRB=Swift_GRB)
    
    NPO_events = NPO_events.assign(T90=np.nan,
                                   GRB_fluence=np.nan,
                                   GRB_flux=np.nan,
                                   GRB_photon_flux=np.nan,
                                   Swift_GRB=False)
    
    df = pd.concat([PO_events,NPO_events]).reset_index(drop=True)
    
    return df[['theta_v',
              'theta_w',
              'theta_j',
              'b',
              'z',
              'd_L',
              'n0',
              'p',
              'epsilon_B',
              'epsilon_e',
              'Eiso',
              'tlength',
              'tGRB',
              'Gamma_0',
              'E_p',
              'T90',
              'GRB_fluence',
              'GRB_flux',
              'GRB_photon_flux',
              'Swift_GRB']]

def GRB_redshifts(rng,nevents,cosmo):
    redshifts = np.linspace(0.01,10.0,1000)
    V = cosmo.comoving_volume(redshifts)
    volume_element = np.gradient(V,redshifts)
    
    probability = GRBFR(redshifts)*volume_element.value/(1+redshifts)
    probability_density = probability/np.sum(probability)
    
    zs = rng.choice(redshifts, nevents, p=probability_density)
    d_Ls = cosmo.luminosity_distance(np.array(zs)).to(u.cm).value
    return zs,d_Ls

def check_detectability(row):
    d = {}

    # For convenience, place arguments into a dict.
    Z = {'jetType':     grb.jet.PowerLawCore,     # Power Law with Core
         'specType':    0,                  # Basic Synchrotron Spectrum
         'thetaObs':    row.theta_v,   # Viewing angle in radians
         'E0':          row.Eiso, # Isotropic-equivalent energy in erg
         'thetaCore':   row.theta_j,    # Half-opening angle in radians
         'thetaWing':   row.theta_w,    # Setting thetaW
         'b':           row.b,
         'n0':          row.n0,    # circumburst density in cm^{-3}
         'p':           row.p,    # electron energy distribution index
         'epsilon_e':   row.epsilon_e,    # epsilon_e
         'epsilon_B':   row.epsilon_B,   # epsilon_B
         'xi_N':        1.0,    # Fraction of electrons accelerated
         'd_L':         row.d_L, # Luminosity distance in cm
         'z':           row.z}   # redshift

    wavel = 473*10**(-9)
    c = 299792458
    # g-band central frequency
    nu =  c/wavel
    
    t = np.linspace(row.tGRB, row.tGRB + row.tlength-1, int(row.tlength))*60
    
    if -2.5*np.log10(np.max(grb.fluxDensity(
            np.geomspace(1,500,10)*60.,nu, **Z))*10**(-3)) + 8.9 > 23.:
        d['detectable'] = False
        d['peak_mag'] = np.nan
    else:
        Fnu_GRB = grb.fluxDensity(t[t>0], nu, **Z)
        Fnu_before = np.array([np.nan for time in t[t<=0]])
        Fnu = np.append(Fnu_before,Fnu_GRB)
        
        gmag = -2.5*np.log10(np.array(Fnu)*10**(-3)) + 8.9
        d['detectable'] = np.min(gmag[np.isnan(gmag) == False]) < 23.
        d['peak_mag'] = np.min(gmag[np.isnan(gmag) == False])
        
    return pd.Series(d, dtype=object)

if __name__ == "__main__":

    rng = np.random.default_rng(seed=12345)
    nevents = 10000000
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    out = 'GRB_afterglows_powerlaw_new.csv'

    if os.path.exists(out):
        GRB_population = pd.read_csv(out)
    else:
        # Generating GRB parameters.
        print('Generating ',nevents,'GRB events.')
        GRB_population = population_synthesis(rng,nevents,cosmo)
        GRB_population.Swift_GRB = GRB_population.Swift_GRB.astype(bool)
        GRB_population.to_csv(out,index=False)
        print('done!')

    dask.config.set({'distributed.scheduler.allowed-failures': 500})
    cluster = SLURMCluster(cores=1,
                       processes=1,
                       memory="800MB",
                       walltime="5:00:00",
                       interface="ib0",
                       worker_extra_args=["--lifetime", "295m",
                                          "--lifetime-stagger", "4m"]
                       )
    cluster.adapt(minimum=1,maximum=500)
    client = Client(cluster)
    GRB_population_dd = dd.from_pandas(GRB_population,npartitions=1000)
    detectable_arr = GRB_population_dd.apply(check_detectability,axis=1,
                                    meta={'detectable':bool,
                                          'peak_mag':float}).compute()
    detectable_arr.to_csv(out.replace('.csv','_detectability.csv'))
    GRB_population = GRB_population.assign(
        DWF_afterglow = detectable_arr.detectable,
        afterglow_peakmag = detectable_arr.peak_mag)
    GRB_population.to_csv(out,index=False)
        
