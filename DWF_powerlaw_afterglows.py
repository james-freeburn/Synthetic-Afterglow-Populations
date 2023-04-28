import numpy as np
import afterglowpy as grb
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import dask.dataframe as dd
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

def yuksel_SFR(redshifts):
    a,b,c = 3.4,-0.3,-3.5
    eta = -10.
    p0 = 0.02
    B = 5000.
    C = 9.
    return p0*((1.+redshifts)**(a*eta) + 
               ((1.+redshifts)/B)**(b*eta) +
               ((1.+redshifts)/C)**(c*eta))**(1./eta)

def GRB_parameters(rng,nevents):
    # Generating GRB times and light curve coverage.
    tlength = rng.integers(50,100,nevents)
    tGRB = [rng.integers(-tlength_val + 3, 1440) for tlength_val in tlength]
    
    theta_wings = rng.uniform(0.1,np.pi/2,nevents)
    n0s = 10**rng.normal(1.0, 1.0,nevents)
    b_vals = rng.uniform(0.1,6.0,nevents)
    ps = []
    ep_Bs = []
    ep_es = []
    for i in range(nevents):
        val_e = 1.0
        val_B = 1.0
        val_p = 1.0
        while val_e >= 1.0:
            val_e = 10**rng.uniform(np.log10(0.15), 0.0)
        ep_es.append(val_e)
        while val_B >= 1.0:
            val_B = 10**rng.normal(-4.0, 1.0)
        ep_Bs.append(val_B)
        while val_p < 2.0:
            val_p = rng.normal(2.21, 0.36)
        ps.append(val_p)
    ep_Bs = np.array(ep_Bs)
    ep_es = np.array(ep_es)
    ps = np.array(ps)
    mu, sigma = 52.5, 1.05
    Eiso_vals = 10**rng.normal(mu, sigma, nevents)
    
    # Generating viewing angles based on solid angle distributions.
    theta = np.linspace(0.0, np.pi/2, 100000)
    omega = 4*np.pi*np.sin(0.5*theta)**2
    probability_density = np.array([((omega[i] - omega[i-1])/
                                     (theta[i] - theta[i-1]))
                                    for i in range(1,len(theta))])/omega[-1]
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
    
    return pd.DataFrame(np.transpose([thetaObs_vals,
                                      theta_wings,
                                      b_vals,
                                      zs,
                                      d_Ls,
                                      n0s,
                                      ps,
                                      ep_Bs,
                                      ep_es,
                                      Eiso_vals,
                                      tlength,
                                      tGRB]),
                        columns=['theta_v',
                                 'theta_w',
                                 'b',
                                 'z',
                                 'd_L',
                                 'n0',
                                 'p',
                                 'epsilon_B',
                                 'epsilon_e',
                                 'Eiso',
                                 'tlength',
                                 'tGRB'])

def GRB_redshifts(rng,nevents):
    redshifts = np.linspace(0.01,10.0,100000)
    SFH = yuksel_SFR(redshifts)
    V = cosmo.comoving_volume(redshifts)
    volume_element = np.array([((V[i] - V[i-1])/
                                (redshifts[i]-redshifts[i-1])).value
                               for i in range(1,len(redshifts))])/V[-1].value
    probability = SFH[1:]*volume_element
    probability_density = probability/np.trapz(probability,redshifts[1:])
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
         'thetaCore':   0.1,    # Half-opening angle in radians
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
    #g-band central frequency
    nu =  c/wavel
    
    t = np.linspace(row.tGRB, row.tGRB + row.tlength-1, int(row.tlength))*60
    
    if -2.5*np.log10(np.max(
            grb.fluxDensity(t[t>0][:5], nu, **Z))*10**(-3)) + 8.9 > 22.25:
        d['detectable'] = False
    else:
        Fnu_GRB = grb.fluxDensity(t[t>0], nu, **Z)
        Fnu_before = np.array([np.nan for time in t[t<=0]])
        Fnu = np.append(Fnu_before,Fnu_GRB)
        gmag = -2.5*np.log10(np.array(Fnu)*10**(-3)) + 8.9

        d['detectable'] = np.min(gmag[np.isnan(gmag) == False]) < 22.25
    return pd.Series(d, dtype=object)
if __name__ == "__main__":    
    
    rng = np.random.default_rng(seed=12345)
    nevents = 10000000
    
    # Generating GRB parameters.
    print('Generating Parameters for',nevents,'events.')
    meta_data = GRB_parameters(rng,nevents)
    print('done!')
    
    cluster = SLURMCluster(cores=1,
                       processes=1,
                       memory="200MB",
                       walltime="01:00:00",
                       interface="ib0",
                       worker_extra_args=["--lifetime", "55m", 
                                          "--lifetime-stagger", "4m"]
                       )
    cluster.adapt(minimum=1,maximum=100)
    client = Client(cluster)
    
    meta_data_dd = dd.from_pandas(meta_data,npartitions=100)
    detectable_arr = meta_data_dd.apply(check_detectability,axis=1,
                                        meta={'detectable':bool}).compute()
    meta_data = meta_data.assign(detectable = detectable_arr)
    meta_data.to_csv('GRB_afterglows_test.csv')
        
