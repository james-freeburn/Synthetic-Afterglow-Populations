import numpy as np
import afterglowpy as grb
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
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

def GRB_spectrum(E):
    alpha = -1.0
    beta =  -2.3
    
    E_0 = 250./(alpha-beta)
    if E < (alpha - beta)*E_0:
        return ((E/100.)**alpha)*np.exp(-E/E_0)
    else:
        return ((alpha-beta)*E_0/100.)**(alpha-beta)*np.exp(beta-alpha)*(E/100.)**beta

def GRB_parameters(rng,nevents):
    # Generating GRB times and light curve coverage.
    tlength = rng.integers(20,200,nevents)
    tGRB = [rng.integers(-tlength_val + 3, 1440) for tlength_val in tlength]
    
    n0s = rng.uniform(0.1, 30.,nevents)
    b_vals = rng.uniform(0.1,6.0,nevents)
    ep_Bs = [0.008]*nevents
    ep_es = [0.02]*nevents
    ps = [2.5]*nevents
    
    # Use 5e50 as intrinsic energy of GRB
    E_gamma = 1e50
    
    # Lorentz factor distribution
    mu1,sigma1 = 4.525,1.475
    Gamma_0s = np.exp(rng.normal(mu1,sigma1,nevents))
    Beta_0s = (1+Gamma_0s**-2)**0.5
    
    # theta_j distribution
    mu2,sigma2 = 1.742,0.916
    theta_js = [0.]*nevents
    for i in range(nevents):
        theta_j = np.exp(rng.normal(mu2,sigma2))
        while theta_j > 90.:
            theta_j = np.exp(rng.normal(mu2,sigma2))
        theta_js[i] = (np.pi/180.)*theta_j
    theta_js = np.array(theta_js)
    
    # Isotropic energy is given by theta_j and Lorentz factor.
    E_iso = np.array([E_gamma/(1-np.cos(theta_j)) if 1./Gamma_0 < np.sin(theta_j) 
             else E_gamma*(1+Beta_0)*Gamma_0**2
            for Gamma_0,Beta_0,theta_j in zip(Gamma_0s,Beta_0s,theta_js)])
    
    # Uniform distribution of theta_wing values (this is what we're testing)
    theta_wings = [rng.uniform(theta_j,np.pi/2) for theta_j in theta_js]

    # Generating probability density function for viewing angle.
    theta = np.linspace(0.0, np.pi/2, 100000)
    omega = 4*np.pi*np.sin(0.5*theta)**2
    probability_density = np.array([((omega[i] - omega[i-1])/
                                     (theta[i] - theta[i-1]))
                                    for i in range(1,len(theta))])/omega[-1]
    # Getting viewing angle values based on probability density.
    thetaObs_vals = []
    for i in range(nevents):
        roll = 1.0
        likelihood = probability_density[0]
        while (roll > likelihood):
            random_index = rng.integers(0,len(theta[theta < theta_wings[i]])-2)
            val = theta[random_index]
            likelihood = probability_density[random_index]
            roll = rng.uniform(0.0,1.0)
        thetaObs_vals.append(val)

    # Placing GRBs at redshifts based on comoving volumes and SFH.
    zs,d_Ls = GRB_redshifts(rng, nevents)
    
    # Truncated log-normal distribution for GRB duration.
    upper,lower = np.log(1e5),np.log(2.0)
    mu3, sigma3 = np.log(27.5), -np.log(0.35)
    T90 = np.exp(stats.truncnorm(
            (lower - mu3) / sigma3, (upper - mu3) / sigma3, loc=mu3, scale=sigma3).rvs(nevents))
    
    # Calculating peak luminosity of GRB and bolometric flux of GRB
    Lpeak = 2.*np.array(E_iso)*(1.+np.array(zs))/T90
    Pbol = Lpeak/(4*np.pi*d_Ls**2)

    # GRB Spectrum
    Es = np.geomspace(1.0,int(1e4),int(1e5)) #keV
    spectrum = np.array([GRB_spectrum(E) for E in Es])
    
    # For re-normalising the spectrum for integration  
    P_peak = np.trapz(spectrum,Es)
    
    # Integrate across Swift's wavelength range to get a flux (P).
    range_min,range_max = find_nearest(Es,15.),find_nearest(Es,150.)
    P = (Pbol/P_peak)*np.trapz(spectrum[range_min:range_max],Es[range_min:range_max])
    
    # Convert flux to a photon flux.
    photon_flux = np.array([np.mean(flux/(u.keV.to(u.erg)*Es[range_min:range_max])) for flux in P])
    # If photon flux is above 2.6, we get a Swift detection.
    Swift_GRB = photon_flux > 2.6
    
    return pd.DataFrame(np.transpose([thetaObs_vals,
                                      theta_wings,
                                      theta_js,
                                      b_vals,
                                      zs,
                                      d_Ls,
                                      n0s,
                                      ps,
                                      ep_Bs,
                                      ep_es,
                                      E_iso,
                                      tlength,
                                      tGRB,
                                      Gamma_0s,
                                      T90,
                                      P,
                                      photon_flux,
                                      Swift_GRB]),
                        columns=['theta_v',
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
                                 'T90',
                                 'GRB_flux',
                                 'GRB_photon_flux',
                                 'Swift_GRB'])

def GRB_redshifts(rng,nevents):
    redshifts = np.linspace(0.01,10.0,100000)
    SFH = GRBFR(redshifts)
    V = cosmo.comoving_volume(redshifts)
    volume_element = np.gradient(V,redshifts)
    probability = SFH*volume_element
    probability_density = probability/np.trapz(probability,redshifts)
    zs = []
    for i in range(nevents):
        roll = 1.0
        likelihood = probability_density[0]
        while (roll > likelihood):
            random_index = rng.integers(0,len(redshifts)-2)
            z = redshifts[random_index]
            likelihood = probability_density[random_index]
            roll = rng.uniform(0.0,1.0)
        zs.append(z)
    d_Ls = cosmo.luminosity_distance(np.array(zs)).to(u.cm).value
    return zs,d_Ls

def check_detectability(row):
    d = {}

    # For convenience, place arguments into a dict.
    Z = {'jetType':     grb.jet.PowerLaw,     # Power Law with Core
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
    nevents = 1000000
    out = 'GRB_afterglows_powerlaw_new.csv'

    # Generating GRB parameters.
    print('Generating Parameters for',nevents,'events.')
    meta_data = GRB_parameters(rng,nevents)
    print('done!')

    dask.config.set({'distributed.scheduler.allowed-failures': 100})
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
    meta_data_dd = dd.from_pandas(meta_data,npartitions=1000)
    detectable_arr = meta_data_dd.apply(check_detectability,axis=1,
                                    meta={'detectable':bool,
                                          'peak_mag':float}).compute()
    meta_data = meta_data.assign(DWF_afterglow = detectable_arr.detectable,
                                 afterglow_peakmag = detectable_arr.peak_mag)
    meta_data.to_csv(out)
        
