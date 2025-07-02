"""Class to generate simulation ICS, separated out for clarity."""
from __future__ import print_function
from typing import Tuple, List, Type, Any
import os.path
import math
import subprocess
import json
import shutil
#To do crazy munging of types for the storage format
import importlib
import numpy as np
import configobj
import classylss
import classylss.binding as CLASS
#from . import utils
#from . import clusters
#from . import cambpower
import datetime

# DM-only
class SimulationICs(object):
    """
    Class for creating the initial conditions for a simulation.
    There are a few things this class needs to do:

    - Generate CAMB input files
    - Generate MP-GenIC input files (to use CAMB output)
    - Run CAMB and MP-GenIC to generate ICs

    The class will store the parameters of the simulation.

    We also save a copy of the input and enough information to reproduce the
    results exactly in SimulationICs.json.

    Many things are left hard-coded.

    We assume flatness.

    Init parameters:
    ----
    box        - Box size in comoving Mpc/h
    npart      - Cube root of number of particles
    redshifts   - redshifts at which to generate ICs
    omegab     - baryon density. Note that if we do not have gas particles,
        still set omegab, but set separate_gas = False
    omega0     - Total matter density at z=0 (includes massive neutrinos and 
        baryons)
    hubble     - Hubble parameter, h, which is H0 / (100 km/s/Mpc)
    scalar_amp - A_s at k = 0.05, comparable to the Planck value.
    ns         - Scalar spectral index
    m_nu       - neutrino mass
    unitary    - if true, do not scatter modes, but use a unitary gaussian
        amplitude.

    Remove:
    ----
    separate_gas - if true the ICs will contain baryonic particles;
        If false, just DM.
    """
    def __init__(self, *,
            box: int,  npart: int,
            seed :         int   = 9281110,      redshifts: float = 99,
            redend:        float = 0,            omega0:   float = 0.288, 
            omegab:        float = 0.0472,       hubble:   float = 0.7,
            scalar_amp:    float = 2.427e-9,     ns:       float = 0.97,
            rscatter:      bool  = False,        m_nu:     float = 0,
            nu_hierarchy:  str   = 'normal', uvb:      str   = "pu",
            nu_acc:        float = 1e-5,         unitary:  bool  = True,
            w0_fld:        float = -1.,           wa_fld:   float = 0., 
            N_ur: float = 3.044, alpha_s: float = 0, MWDM_therm: float = 0,
            python:        str = "python") -> None:
        #Check that input is reasonable and set parameters
        #In Mpc/h
        print("__init__: initializing parameters...", datetime.datetime.now())
        assert box  < 20000
        self.box      = box

        #Cube root
        assert npart > 1 and npart < 16000
        self.npart    = int(npart)

        #Physically reasonable
        assert omega0 <= 1 and omega0 > 0
        self.omega0   = omega0

        assert omegab > 0 and omegab < 1
        self.omegab   = omegab

        #assert redshifts > 1 and redshifts < 1100
        self.redshifts = redshifts

        assert redend >= 0 and redend < 1100
        self.redend = redend

        assert hubble < 1 and hubble > 0
        self.hubble = hubble

        assert scalar_amp < 1e-7 and scalar_amp > 0
        self.scalar_amp = scalar_amp

        assert ns > 0 and ns < 2
        self.ns      = ns
        self.unitary = unitary

        # assert w0_fld < 0
        self.w0_fld = w0_fld

        # assert wa_fld < 1 and wa_fld > -1
        self.wa_fld = wa_fld

        assert N_ur >= 0
        self.N_ur = N_ur

        assert alpha_s < 1 and alpha_s > -1
        self.alpha_s = alpha_s

        T_CMB = 2.7255  # default cmb temperature
        omegag = 4.480075654158969e-07 * T_CMB**4 / self.hubble**2
        self.omegag = omegag
        self.omega_ur = omegag * 0.22710731766023898 * (self.N_ur - 1.013198221453432*3)  # MP-Gadget
        # assert self.omega_ur >= 0

        assert MWDM_therm >= 0
        self.MWDM_therm = MWDM_therm

        
        #Neutrino accuracy for CLASS
        self.nu_acc  = nu_acc

        #UVB? Only matters if gas
        self.uvb = uvb
        assert self.uvb == "hm" or self.uvb == "fg" or self.uvb == "sh" or self.uvb == "pu"

        self.rscatter = rscatter


        #Structure seed.
        self.seed = seed

        #Baryons?
        self.separate_gas = False # separate_gas

        #If neutrinos are combined into the DM,
        #we want to use a different CAMB transfer when checking output power.
        self.separate_nu  = False
        self.m_nu         = m_nu
        self.nu_hierarchy = nu_hierarchy

        # initialize the cluster object: to store the submit info for specific
        # cluster in used
        #self._cluster = cluster_class(
        #    gadget = self.gadgetexe, param      = self.gadgetparam, 
        #    genic  = self.genicexe,  genicparam = self.genicout, 
        #    nproc  = nproc,          cores      = cores,
        #    gadget_dir = gadget_dir, mpi_ranks = mpi_ranks, threads = threads)                     # add nproc and cores
        #                                                 # make them optional
        #assert self._cluster.gadget_dir == os.path.expanduser(gadget_dir)

        #For repeatability, we store git hashes of Gadget, GenIC, CAMB and ourselves
        #at time of running.
        #self.simulation_git = utils.get_git_hash(os.path.dirname(__file__))

        # sometime the path to python is slightly different, especially you have
        # multiple pythons and multiple virtual env
        self.python = python
        print("__init__: done.", datetime.datetime.now(),"\n")

    @property
    def json(self) -> dict:
        """
        these are the variables will be saved into json
        """
        return self.__dict__



    def cambfile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the IC power spectrum using classylss.
        
        Basically is using pre_params feed into class and compute the powerspec
        files based on the range of redshift and redend. All files are stored in
        the directory camb_out.

        Return:
        ----
        camb_output (str) : power specs directory. default: "camb_linear/"
        """
        #Load high precision defaults
        print("cambfile: loading defaults...", datetime.datetime.now())
        pre_params = {
            'tol_background_integration': 1e-9, 'tol_perturb_integration' : 1.e-7,
            'tol_thermo_integration':1.e-5, 'k_per_decade_for_pk': 50,'k_bao_width': 8,
            'k_per_decade_for_bao':  200, 'neglect_CMB_sources_below_visibility' : 1.e-30,
            'transfer_neglect_late_source': 3000., 'l_max_g' : 50,
            'l_max_ur':150, 'extra metric transfer functions': 'y'}

        #Set the neutrino density and subtract it from omega0
        omeganu = self.m_nu/93.14/self.hubble**2
        omcdm   = (self.omega0 - self.omegab) - omeganu
        gparams = {'h': self.hubble, 'Omega_cdm': omcdm,'Omega_b': self.omegab,
            'Omega_k': 0, 'n_s': self.ns, 'A_s': self.scalar_amp, 'alpha_s': self.alpha_s}

        #Lambda is computed self-consistently
        if self.w0_fld != -1.0 or self.wa_fld != 0.:
            gparams['Omega_Lambda'] = 0
            gparams['w0_fld'] = self.w0_fld 
            gparams['wa_fld'] = self.wa_fld
            if (gparams['w0_fld'] + gparams['wa_fld'] + 1) * (1 + gparams['w0_fld']) > 0: # if not phantom-crossing
                gparams['use_ppf'] = 'no'
        else:
            gparams['Omega_fld'] = 0

        numass = get_neutrino_masses(self.m_nu, self.nu_hierarchy)

        #Set up massive neutrinos
        if self.m_nu > 0:
            print("cambfile: setting up massive neutrinos...")
            # gparams['m_ncdm'] = '%.8f,%.8f,%.8f' % (numass[2], numass[1], numass[0])
            # gparams['N_ncdm'] = 3

            m_ncdm = ''
            N_ncdm = 0
            for i, numa in enumerate(numass):
                if numa == 0:
                    continue
                if N_ncdm == 0:
                    m_ncdm += '%.8f' % numa
                else:
                    m_ncdm += ',%.8f' % numa 
                N_ncdm += 1           
            gparams['m_ncdm'] = m_ncdm
            gparams['N_ncdm'] = N_ncdm
            
            # gparams['N_ur'] = 0.00641
            # gparams['N_ur'] = self.N_ur - 3
            gparams['N_ur'] = self.N_ur - gparams['N_ncdm'] * 1.013198221453432 # for N_ncdm = 3, N_ur = N_ur^desired - 3.0395946643602962
            #Neutrino accuracy: Default pk_ref.pre has tol_ncdm_* = 1e-10,
            #which takes 45 minutes (!) on my laptop.
            #tol_ncdm_* = 1e-8 takes 20 minutes and is machine-accurate.
            #Default parameters are fast but off by 2%.
            #I chose 1e-5, which takes 6 minutes and is accurate to 1e-5
            gparams['tol_ncdm_newtonian'] = self.nu_acc #min(self.nu_acc,1e-5)
            gparams['tol_ncdm_synchronous'] = self.nu_acc
            gparams['tol_ncdm_bg'] = 1e-10
            gparams['l_max_ncdm'] = 50
            #This disables the fluid approximations, which make P_nu not match 
            # camb on small scales.
            #We need accurate P_nu to initialise our neutrino code.
            gparams['ncdm_fluid_approximation'] = 3 # 3: disable approximation; # enum ncdmfa_method {ncdmfa_mb,ncdmfa_hu,ncdmfa_CLASS,ncdmfa_none};
            #Does nothing unless ncdm_fluid_approximation = 2
            #Spend less time on neutrino power for smaller neutrino mass
            gparams['ncdm_fluid_trigger_tau_over_tau_k'] = 30000.* (self.m_nu / 0.4)
        else:
            # gparams['N_ur'] = 3.046
            gparams['N_ur'] = self.N_ur # for mnu = 0, N_ur cannot be set to 0 in CLASS

        #Initial cosmology
        pre_params.update(gparams)

        maxk        = 2 * math.pi / self.box * self.npart * 8
        powerparams = {'output': 'dTk vTk mPk', 'P_k_max_h/Mpc' : maxk, 
            "z_max_pk" : 99 + 1}
        pre_params.update(powerparams)

        #At which redshifts should we produce CAMB output: we want the start and
        # end redshifts of the simulation, but we also want some other values
        # for checking purposes
        camb_zz = self.redshifts

        classconf = configobj.ConfigObj()
        classconf.filename = None
        classconf.update(pre_params)
        classconf['z_pk'] = camb_zz
        classconf.write()

        # feed in the parameters and generate the powerspec object
        print("cambfile: generating the powerspec object...")
        engine  = CLASS.ClassEngine(pre_params)
        powspec = CLASS.Spectra(engine) # powerspec is an object

        # bg = CLASS.Background(engine)
        # pre_params['Omega_fld'] = 1 - self.omega0 + bg.Omega0_lambda  # so that Omega0_lambda == 0 (forced)

        # engine  = CLASS.ClassEngine(pre_params)
        # powspec = CLASS.Spectra(engine) # powerspec is an object



        #Get transfer fucntion and linear power spectrum
        print(f"cambfile: getting the transfer functions... for z = {camb_zz} ")
        all_ks = []
        all_pk_lins = []
        for zz in camb_zz:
            trans = powspec.get_transfer(z=zz)

            #fp-roundoff
            trans['k (h/Mpc)'][-1] *= 0.9999

            pk_lin = powspec.get_pklin(k=trans['k (h/Mpc)'], z=zz)
            all_ks.append(trans['k (h/Mpc)'])
            all_pk_lins.append(pk_lin)
        print("cambfile: done.", datetime.datetime.now(),"\n")
        return np.array(all_ks), np.array(all_pk_lins)


    def _fromarray(self) -> None:
        """Convert the data stored as lists back to what it was."""
        for arr in self._really_arrays:
            self.__dict__[arr] = np.array(self.__dict__[arr])
        self._really_arrays = []
        for arr in self._really_types:
            #Some crazy nonsense to convert the module, name
            #string tuple we stored back into a python type.
            mod = importlib.import_module(self.__dict__[arr][0])
            self.__dict__[arr] = getattr(mod, self.__dict__[arr][1])
        self._really_types = []


    def _other_params(self, config: configobj.ConfigObj) -> configobj.ConfigObj:
        """Function to override to set other config parameters"""
        return config

   
def get_neutrino_masses(total_mass: float, hierarchy: str) -> np.ndarray:
    """Get the three neutrino masses, including the mass splittings.
        Hierarchy is 'inverted' (two heavy), 'normal' (two light) or degenerate."""
    #Neutrino mass splittings
    nu_M21 = 7.53e-5 #Particle data group 2016: +- 0.18e-5 eV2
    nu_M32n = 2.44e-3 #Particle data group: +- 0.06e-3 eV2
    nu_M32i = 2.51e-3 #Particle data group: +- 0.06e-3 eV2

    if hierarchy == 'normal':
        nu_M32 = nu_M32n
        #If the total mass is below that allowed by the hierarchy,
        #assign one active neutrino.
        # if total_mass < np.sqrt(nu_M32n) + np.sqrt(nu_M21):
        if total_mass < .06:
            return np.array([total_mass, 0, 0])
    elif hierarchy == 'inverted':
        nu_M32 = -nu_M32i
        if total_mass < 2*np.sqrt(nu_M32i) - np.sqrt(nu_M21):
            return np.array([total_mass/2., total_mass/2., 0])
    #Hierarchy == 0 is 3 degenerate neutrinos
    else:
        return np.ones(3)*total_mass/3.

    #DD is the summed masses of the two closest neutrinos
    DD1 = 4 * total_mass/3. - 2/3.*np.sqrt(total_mass**2 + 3*nu_M32 + 1.5*nu_M21)
    #Last term was neglected initially. This should be very well converged.
    DD = 4 * total_mass/3. - 2/3.*np.sqrt(total_mass**2 + 3*nu_M32 + 1.5*nu_M21+0.75*nu_M21**2/DD1**2)
    nu_masses = np.array([ total_mass - DD, 0.5*(DD + nu_M21/DD), 0.5*(DD - nu_M21/DD)])
    assert np.isfinite(DD)
    assert np.abs(DD1/DD -1) < 2e-2
    assert np.all(nu_masses >= 0)
    return nu_masses