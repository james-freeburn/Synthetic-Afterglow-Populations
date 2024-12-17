import numpy as np
import afterglowpy as grb
import pandas as pd
from astropy.constants import c
import dask
import dask.dataframe as dd
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import argparse

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx

def generate_parameters(df,lc_lengths,cadence,rng,tGRBmin,tGRBmax):
    if len(lc_lengths) == 1:
        tlength = [lc_lengths.iloc[0].length]*len(df)
    else:
        tlength = rng.choice(lc_lengths.length, len(df), p=lc_lengths.prob)
    if tGRBmin is not None:
        tGRB = [rng.uniform(tGRBmin, tGRBmax) for tlength_val in tlength]
    else:
        tGRB = [rng.uniform(-tlength_val*cadence + 2.*cadence, tGRBmax) 
                for tlength_val in tlength]
    # Generating GRB times and light curve coverage.
    df = df.assign(tlength = tlength,
                   tGRB = tGRB,
                   n0 = rng.uniform(0.1,30.,len(df)),
                   b = [0.8]*len(df),#rng.uniform(0.0,3.0,len(df)),
                   epsilon_B = [0.008]*len(df),
                   epsilon_e = [0.02]*len(df),
                   p = [2.3]*len(df),
                   theta_w = [1.]*len(df))#[rng.uniform(theta_j,np.pi/2) 
                              #for theta_j in df.theta_j])
    return df

def check_detectability(row,cadence=50.,limmag=23.,wavel=473.,mindetections=1):
    d = {}
    
    # For convenience, place arguments into a dict.
    Z = {'jetType':     row.jet,     # Power Law with Core
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

    # g-band central frequency
    nu =  c/(wavel*10**(-9))
    t = np.linspace(row.tGRB, row.tGRB + (row.tlength-1)*cadence, 
                    int(row.tlength))
    
    intrinsic_peakmag = -2.5*np.log10(np.max(grb.fluxDensity(
        np.geomspace(1,500,10)*60.,nu, **Z))*10**(-3)) + 8.9
    d['intrinsic_peakmag'] = intrinsic_peakmag
    
    if intrinsic_peakmag > limmag:
        d['detectable'] = False
        d['ndetections'] = 0
        d['peak_mag'] = np.nan
    else:
        Fnu_GRB = grb.fluxDensity(t[t>0], nu, **Z)
        Fnu_before = np.array([np.nan for time in t[t<=0]])
        Fnu = np.append(Fnu_before,Fnu_GRB)
        
        gmag = -2.5*np.log10(np.array(Fnu)*10**(-3)) + 8.9
        d['detectable'] = len(gmag[gmag < limmag]) > mindetections
        d['ndetections'] = len(gmag[gmag < limmag])
        d['peak_mag'] = np.min(gmag[np.isnan(gmag) == False])
        
    return pd.Series(d, dtype=object)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Synthetic GRBs')
    parser.add_argument('-g', '--grbpopulation',
                        type=str,
                        default='GRB_population.csv',
                        nargs=1,
                        help='GRB population to draw from.')
    parser.add_argument('-s', '--seed',
                        type=int,
                        default=[12345],
                        nargs=1,
                        help='Seed for random number generator.')
    parser.add_argument('-l', '--limmag',
                        type=float,
                        default=[23.],
                        nargs=1,
                        help='Limiting AB magnitude for afterglow \
                            classification.')
    parser.add_argument('-j', '--jet',
                        type=int,
                        default=[1],
                        nargs=1,
                        help='Jet model: -1 TopHat, 0 Gaussian, 1 PowerlawCore, \
                            2 GaussianCore, 3 Spherical or 4 PowerLaw.')
    parser.add_argument('-c', '--cadence',
                        type=float,
                        default=[50.],
                        nargs=1,
                        help='Cadence of observations in seconds.')
    parser.add_argument('-f', '--filter',
                        type=float,
                        default=[473.],
                        nargs=1,
                        help='Central wavelength of filter in nm.')
    parser.add_argument('-t', '--tGRBmax',
                        type=float,
                        default=[24.*60*60],
                        nargs=1,
                        help='Maximum time before observations to generate\
                            a GRB in seconds.')
    parser.add_argument('-i', '--tGRBmin',
                        type=float,
                        default=[None],
                        nargs=1,
                        help='Maximum time after observations start to generate\
                            a GRB in seconds.')
    parser.add_argument('-w', '--tlength',
                        type=str,
                        default='lc_lengths.csv',
                        nargs=1,
                        help='csv file containing lengths of observing windows\
                            in seconds with weights.')
    parser.add_argument('-n', '--name',
                        type=str,
                        default='test',
                        nargs=1,
                        help='Name of observing programme.')
    parser.add_argument('-u', '--mindetections',
                        type=int,
                        default=[1],
                        nargs=1,
                        help='Required minimum number of detections for an id.')
    parser.add_argument('-d', '--distribute',
                        dest='distribute',
                        default=False,
                        action='store_true',
                        help='Use dask to distribute jobs.')
    parser.add_argument('-m', '--memory',
                        type=int,
                        default=[800],
                        nargs=1,
                        help='Amount of memory per work in MB.')
    parser.add_argument('-p', '--partitions',
                        type=int,
                        default=[500],
                        nargs=1,
                        help='Number of partitions.')
    args = parser.parse_args()
    rng = np.random.default_rng(seed=args.seed[0])
    
    cadence = args.cadence[0]
    jet = args.jet[0]
    limmag = args.limmag[0]
    tGRBmax = args.tGRBmax[0]
    tGRBmin = args.tGRBmin[0]
    print(tGRBmax) 
    jet_names = ['TopHat','Gaussian','PowerlawCore','GaussianCore','Spherical',
                 'PowerLaw']
    pop = args.grbpopulation[0]
    out = 'GRB_afterglows_' + jet_names[jet+1] + '_' + args.name[0] + '.csv'

    # Generating GRB parameters.
    GRB_population = pd.read_csv(pop) 
    GRB_population = GRB_population.assign(jet=jet)
    print('Generating afterglow parameters for',
          len(GRB_population),'GRB events.')

    lc_lengths = pd.read_csv(args.tlength[0])

    GRB_population = generate_parameters(GRB_population,lc_lengths,
                                         cadence,rng,tGRBmin,tGRBmax)
    print('done!')
    if args.distribute:
        dask.config.set({'distributed.scheduler.allowed-failures': 500})
        cluster = SLURMCluster(cores=1,
                           processes=1,
                           memory= str(args.memory[0]) + "MB",
                           walltime="5:00:00",
                           interface="ib0",
                           worker_extra_args=["--lifetime", "295m",
                                              "--lifetime-stagger", "4m"]
                           )
        cluster.adapt(minimum=1,maximum=500)
        client = Client(cluster)
        GRB_population_dd = dd.from_pandas(GRB_population,
                                           npartitions=args.partitions[0])
        detectable_arr = GRB_population_dd.apply(
            check_detectability,
            cadence=cadence,
            limmag=limmag,
            wavel=args.filter[0],
            mindetections=args.mindetections[0],
            axis=1,
            meta={'intrinsic_peakmag':float,
                  'detectable':bool,
                  'ndetections':int,
                  'peak_mag':float}).compute()
    else:
        detectable_arr = GRB_population.apply(check_detectability,
                                              cadence=cadence,
                                              limmag=limmag,
                                              wavel=args.filter[0],
                                              axis=1)
    GRB_population = GRB_population.assign(
        intrinsic_peakmag = detectable_arr.intrinsic_peakmag,
        detectable = detectable_arr.detectable,
        ndetections = detectable_arr.ndetections,
        detected_peakmag = detectable_arr.peak_mag)
    GRB_population.to_csv(out,index=False)
