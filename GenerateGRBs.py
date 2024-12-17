import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import xspec
import argparse

def GRB_spectrum(Es,E_p,alpha=-1.,beta=-2.3):
    E_0 = E_p/(alpha+2.)
    return np.array([((E/100.)**alpha)*np.exp(-E/E_0) 
                     if E < (alpha - beta)*E_0 
                     else 
                     (((alpha-beta)*E_0/100.)**(alpha-beta))*
                     (np.exp(beta-alpha)*(E/100.)**beta)
                     for E in Es])

def integrate_spectrum(GRB,fs1,A=1.,
                       alpha=-1.,beta=-2.3,
                       e1=15.,e2=150.,
                       A_eff=1400):
    
    m1 = xspec.Model("GRBm",
            setPars={1:alpha,
                     2:beta,
                     3:GRB.E_p/(1+GRB.z),
                     4:A})
    xspec.AllData.fakeit(1,fs1)
    xspec.AllModels.calcFlux(str(e1) + ' ' + str(e2))
    
    energy = xspec.AllData(1).flux[0]
    photons = xspec.AllData(1).flux[3]#/A_eff

    xspec.AllModels.clear()
    xspec.AllData.clear()
    
    return energy,photons

def GRB_observables(GRB,fs1):
    d = {}
    
    # Isotropic Energy
    E1,E2 = 1e0,1e4
    Es = np.geomspace(E1,E2,int(1e3))
    spectrum = GRB_spectrum(Es,GRB.E_p)
    I = np.trapz(Es*spectrum,Es)
    
    Fbol = GRB.Eiso/((4*np.pi*GRB.d_L**2)/(1+GRB.z)**2)
    Lpeak = 2.*GRB.Eiso*(1.+GRB.z)/GRB.T90
    Pbol = Lpeak/(4.*np.pi*GRB.d_L**2)
    
    energy,photons = integrate_spectrum(GRB,fs1)
    
    A_flux = Pbol*u.erg.to(u.keV)/I
    A_fluence = Fbol*u.erg.to(u.keV)/I
    
    print(A_flux,A_fluence)
    
    d['GRB_id'] = GRB.GRB_id
    d['GRB_fluence'] = A_fluence*energy
    d['GRB_flux'] = A_flux*energy
    d['GRB_photon_flux'] = A_flux*photons
    
    return pd.Series(d, dtype=object)

def generate_GRBs(rng,nevents):
    mu_gamma = 1.95
    sigma_gamma = 0.65
    sigma_theta_j = 0.3
    
    m = 2.5
    q = 1.45
    
    log_Gamma_0s = rng.normal(mu_gamma,sigma_gamma,nevents)
    log_theta_js = np.array([rng.normal((-1./m)*log_Gamma_0 + q,sigma_theta_j) 
                             for log_Gamma_0 in log_Gamma_0s])
    Gamma_0s = 10**log_Gamma_0s
    theta_js = (np.pi/180.)*10**log_theta_js
    mask = (Gamma_0s > 1.) & (Gamma_0s < 8e3) & (theta_js < np.pi/2.) 
    
    return Gamma_0s[mask],theta_js[mask]

def GRBFR(z):
    R = (0.00157 + 0.118*z)/(1+(z/3.23)**4.66)
    e = (1+z)**1.7
    return R*e

def GRB_redshifts(rng,nevents,cosmo):
    redshifts = np.linspace(0.01,10.0,1000)
    V = cosmo.comoving_volume(redshifts)
    volume_element = np.gradient(V,redshifts)
    
    probability = GRBFR(redshifts)*volume_element.value/(1+redshifts)
    probability_density = probability/np.sum(probability)
    
    zs = rng.choice(redshifts, nevents, p=probability_density)
    d_Ls = cosmo.luminosity_distance(np.array(zs)).to(u.cm).value
    return zs,d_Ls

def population_synthesis(rng,nevents,cosmo):
    
    # Intrinsic energy of GRBs
    E_gamma_dash = 1.5e48
    E_p_dash = 1.5
    
    print('\tGenerating Data ... ')
    Gamma_0s,theta_js = generate_GRBs(rng,nevents)
    nevents = len(Gamma_0s)
    
    print('\tGenerating theta_v values ... ')
    # Generating probability density function for viewing angle.
    theta = np.linspace(0.0, np.pi/2, 1000)
    probability_density = np.sin(theta)
    probability_density = probability_density/np.sum(probability_density)
    thetaObs_vals = rng.choice(theta, nevents, p=probability_density)
    
    print('\tCalculating params ... ')
    df = pd.DataFrame([])
    df = df.assign(Gamma_0=Gamma_0s,
                   theta_j=theta_js,
                   theta_v=thetaObs_vals,
                   Beta_0 = (1.-(1./Gamma_0s**2.))**0.5,
                   E_gamma = E_gamma_dash*Gamma_0s)
    
    df = df.assign(GRB_id = df.index,
                   E_p = E_p_dash*5.*df.Gamma_0/(5.-2.*df.Beta_0),
                   Eiso = [E_gamma/(1-np.cos(theta_j)) 
                           if 1./Gamma_0 < np.sin(theta_j)
                           else E_gamma*(1+Beta_0)*Gamma_0**2
                           for E_gamma,Gamma_0,Beta_0,theta_j 
                           in zip(df.E_gamma,df.Gamma_0,df.Beta_0,df.theta_j)])
    
    print('\tAssigning redshifts ...')
    # Placing GRBs at redshifts based on comoving volumes and SFH.
    zs,d_Ls = GRB_redshifts(rng, len(df), cosmo)
    df = df.assign(z=zs,d_L=d_Ls)
    
    PO_mask = (
        (df.theta_v <= df.theta_j) | (np.sin(df.theta_v) <= 1./df.Gamma_0)) & (
            (df.E_p/(1+df.z) > 15.) & (df.E_p/(1+df.z) < 2000.)
            )
    PO_events = df[PO_mask].reset_index(drop=True)
    NPO_events = df[PO_mask == False].reset_index(drop=True)
    
    print('\tGenerating GRB params ...')
    # Truncated log-normal distribution for GRB duration.
    mu3, sigma3 = 27.5,0.35
    T90 = [0.]*len(PO_events)
    for i in range(len(PO_events)):
        roll = 10**(rng.normal(np.log10(mu3),sigma3))
        while roll < 2.0:
            roll = 10**(rng.normal(np.log10(mu3),sigma3))
        T90[i] = roll
    
    PO_events = PO_events.assign(T90 = T90)
    fs1 = xspec.FakeitSettings("bat.rsp",exposure=1e5)
    PO_events = pd.merge(PO_events,
                         PO_events.apply(GRB_observables,axis=1,fs1=fs1),
                         left_on='GRB_id',
                         right_on='GRB_id')
    PO_events = PO_events.assign(
        Swift_GRB = (PO_events.GRB_photon_flux > 2.6))
    
    NPO_events = NPO_events.assign(T90=np.nan,
                                   GRB_fluence=np.nan,
                                   GRB_flux=np.nan,
                                   GRB_photon_flux=np.nan,
                                   Swift_GRB=False)
    df = pd.concat([PO_events,NPO_events]).reset_index(drop=True)

    return df[['theta_v',
              'theta_j',
              'z',
              'd_L',
              'Eiso',
              'Gamma_0',
              'E_p',
              'T90',
              'GRB_fluence',
              'GRB_flux',
              'GRB_photon_flux',
              'Swift_GRB']]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Synthetic GRBs')
    parser.add_argument('-n', '--nevents',
                        type=int,
                        default=[10000],
                        nargs=1,
                        help='Number of GRBs to simulate.')
    args = parser.parse_args()

    rng = np.random.default_rng(seed=12345)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    # Generating GRB parameters.
    print('Generating ',args.nevents[0],'GRB events.')
    GRB_population = population_synthesis(rng,args.nevents[0],cosmo)
    GRB_population.Swift_GRB = GRB_population.Swift_GRB.astype(bool)
    GRB_population.to_csv("GRB_population.csv",index=False)
    print('done!')

