import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as units
from astropy.table import Table
import scifysim as sf
import nifits
sf.logit.setLevel(sf.logging.ERROR)
from scifysim.dummy import makesim
import sys
import os
from pathlib import Path

import argparse

from time import time

start_full = time()

parser = argparse.ArgumentParser(
                    prog='SCIFYsim Script',
                    description='Computes a simulation of NOTT observation with realistic errobars',
                    epilog='Thank you for using SCIFYsim')
parser.add_argument("-c", "--config", dest="config", type=str)
parser.add_argument("-t", "--target", dest="target", type=str, default="None", help="if None: [target]-> target")
parser.add_argument("-o", "--out", dest="out", type=str, help="Directory for output")
parser.add_argument("-m", "--mode", dest="mode", default="None", type=str, help="if `None` : [configuration]-> mode")
parser.add_argument("-r", "--replace", dest="replace", action="store_true")
parser.add_argument("-n", "--nloop_covar", dest="nloop_covar", default=400, type=int, help="if 0  : [appendix]-> nloop_covar")
parser.add_argument("-d", "--dit", dest="dit", default=0., type=float, help="if 0. : [configuration]-> dit")
parser.add_argument("-s", "--schedule", dest="schedule", action="store_true", help="If true, quits after plotting elevation")
parser.add_argument("-a", "--n_coadds", dest="n_coadds", default=100, type=int, help="if 0 : [configuration]-> n_coadds")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
args = parser.parse_args()

print(args.__dict__)
print("______________________________")
config = args.config
target = args.target
mode = args.mode
outdir = args.out
nloop_covar = args.nloop_covar
exptime = args.dit
n_coadds = args.n_coadds
verbose = args.verbose

first_parse = sf.parsefile.parse_file(config)

if target == "None":
    target = first_parse.get("target", "target")
else:
    first_parse.set("target", "target", target)
if mode == "None":
    mode = first_parse.get("configuration", "mode")
else:
    first_parse.set("configuration", "mode", mode)
if nloop_covar == 0:
    nloop_covar = first_parse.getint("appendix", "nloop_covar")
else:
    first_parse.set("appendix", "nloop_covar", str(nloop_covar))
if exptime <= 1e-8:
    exptime = first_parse.getfloat("configuration", "dit")
else:
    first_parse.set("configuration", "dit", str(exptime))
if n_coadds == 0:
    n_coadds = first_parse.getint("configuration", "n_coadds")
else:
    first_parse.set("configuration", "n_coadds", str(n_coadds))

if Path(outdir).is_dir():
    if args.replace is False:
        print("Directory already exists")
        sys.exit(100)
else:
    os.mkdir(outdir)

if not Path(config).is_file():
    print("Config not found")
    sys.exit(100)
print("===================================================")
print("Config:")
with open(config, "r") as afile:
    print(afile.read())

asim = makesim(first_parse,
               target=target)
# asim = makesim("/home/romain/Documents/hi5/didactic/MAP1_2023_NOTT_symmetric_mode/config/perfect_R400.ini",
#                target="Gl 86 A ")

with open(f"{outdir}/config.ini", "w") as afile:
    asim.config.write(afile)

from kernuller import pairwise_kernel


from kernuller.diagrams import plot_chromatic_matrix, plot_outputs_smart
import sympy as sp


expname = mode

inpol = np.array(["s","s","s","s"])[None,:]
outpol = np.array(["s","s","s","s","s","s","s","s"])[None,:]

if mode == "planet":
    pass
elif mode == "disk":
    asim.combiner = sf.combiner.combiner.from_config(asim.config, ph_shifters=(0,-sp.pi/2)) # 
    asim.combiner.chromatic_matrix(asim.lambda_science_range)
elif mode == "backup-disk":
    asim.combiner = sf.combiner.combiner.from_config(asim.config, ph_shifters=(0,0))
    vec_backup = sp.exp(sp.I*sp.pi * sp.diag(0,1,0,1))
    # vec_backup2 = sp.exp(sp.I*sp.pi/2 * sp.diag(0,0,1,1))
    asim.combiner.M = sp.Matrix(asim.combiner.M@(vec_backup))
    asim.combiner.dark = np.array([False,False,True,False,False,True,False,False,])
    asim.combiner.bright = np.array([False,False,False,True,True,False,False,False,])
    

elif mode == "backup-disk+":
    asim.combiner = sf.combiner.combiner.from_config(asim.config, ph_shifters=(0,0))
    vec_backup = sp.exp(sp.I*sp.pi * sp.diag(0,1,0,1))
    vec_backup2 = sp.exp(sp.I*sp.pi/2 * sp.diag(0,0,1,1))
    asim.combiner.M = sp.Matrix(asim.combiner.M@(vec_backup*vec_backup2))
    asim.combiner.dark = np.array([False,False,True,False,True,True,False,False,])
    asim.combiner.bright = np.array([False,False,False,True,False,False,False,False,])
    

asim.combiner.chromatic_matrix(asim.lambda_science_range)
asim.point(asim.sequence[4], asim.target, refresh_array=True)

ak = pairwise_kernel(2)
myk = np.hstack((np.zeros((1,3)), ak, np.zeros((1,3))))
asim.combiner.K = myk
del ak
del myk


from kernuller.diagrams import plot_chromatic_matrix, plot_outputs_smart
# asim.combiner = sf.combiner.combiner.from_config(asim.config, ph_shifters=(0,-np.pi/2))
# asim.combiner.chromatic_matrix(asim.lambda_science_range)
# asim.point(asim.sequence[10], asim.target, refresh_array=True)

asim.point(asim.sequence[0], asim.target)
g_atmo = asim.offband_model.get_phase_science_values(asim.pistons)
g_internal = asim.corrector.get_phasor(asim.lambda_science_range)
# plt.plot(asim.lambda_science_range, g_atmo)
# plt.plot(asim.lambda_science_range, g_internal)
Mcnc = asim.combiner.Mcn*g_internal[:,None,:]

if verbose:
    
    fig1, axs = plot_outputs_smart(matrix = asim.combiner.Mcn[30])
    fig1.show()
    fig2, axs, matrix = plot_chromatic_matrix(asim.combiner.M,
                                             sf.combiners.lamb, asim.lambda_science_range,
                                             verbose=False, returnmatrix=True,minfrac=0.9,
                                             plotout=True, show=False, title="With Tepper couplers")
    fig2.show()
fig3, axs, matrix = plot_chromatic_matrix(asim.combiner.M,
                                         sf.combiners.lamb, asim.lambda_science_range,
                                         verbose=False, returnmatrix=True,minfrac=0.9,
                                         plotout=g_internal, show=False, title="With Tepper couplers")
fig3.savefig(f"{outdir}/matrix_plot_final.pdf", dpi=100)
plt.close()

print("Checking elevation")
sf.observatory.plots.plot_altitude(asim.target, asim.obs.observatory_location, asim.sequence[:], )
plt.gcf().savefig(f"{outdir}/elevation.pdf")
plt.close()
for i in range(len(asim.sequence[:])):
    fig = sf.plot_tools.plot_projected_pupil(asim, i, perspective=False, usize=200)
    fig.savefig(f"{outdir}/projected_{i:03d}.pdf")
    plt.close()

if args.schedule:
    print("Exiting after saving elevation.")
    sys.exit(0)

print("Checking maps")
asim.build_all_maps(mapres=130, mapcrop=0.6 )
figs = sf.plot_tools.plot_response_map(asim, sequence_index=(0,), show=False)
figs[0].savefig(f"{outdir}/map_wide.pdf", dpi=200)

asim.build_all_maps(mapres=130, mapcrop=0.1 )
figs = sf.plot_tools.plot_response_map(asim, sequence_index=(0,), show=False)
figs[0].savefig(f"{outdir}/map_tight.pdf", dpi=200)



print("Spectro view")
t_exp = 1.0
# asim.combiner.chromatic_matrix(asim.lambda_science_range)
halfway = len(asim.sequence)//2
asim.point(asim.sequence[halfway], asim.target)

integ = asim.make_metrologic_exposure(asim.src.planet, asim.src.star, asim.diffuse,
                                      texp=t_exp)
integ.prepare_t_exp_base()
integ.consolidate_metrologic()
print("Metrologic: ")
print(integ.static_list)
print(integ.sums)

perfect_injection = np.ones((asim.lambda_science_range.shape[0], asim.ntelescopes))
dummy_collected = np.ones(asim.lambda_science_range.shape[0])
fig = sf.plot_tools.plot_output_sources(asim, integ, asim.lambda_science_range, t_exp=1)
fig.savefig(f"{outdir}/spectro_view.pdf", dpi=200)
plt.close()


from tqdm import tqdm
from time import time

traces =  []
for i, atime in enumerate(asim.sequence):
    print("==========================================================================================")
    asim.point(atime, asim.target)
    integ.reset()
    integ = asim.make_metrologic_exposure(asim.src.planet, asim.src.star, asim.diffuse,
                                          texp=0.1)
    traces.append(integ.static[2][:,3])
traces = np.array(traces)
plt.figure()
for i, atrace in enumerate(traces):
    plt.plot(asim.lambda_science_range, atrace, label=str(i))
plt.title(str(i))
plt.close()

asource = asim.diffuse[1]
aspectrum = asource.get_downstream_transmission(asim.lambda_science_range, inclusive=False) \
            * asource.get_own_brightness(asim.lambda_science_range)
vigneted_spectrum = asim.injector.vigneting.vigneted_spectrum(aspectrum,
                                                asim.lambda_science_range,
                                                0.005)
perfect_injection = np.ones((asim.lambda_science_range.shape[0], asim.ntelescopes))
dummy_collected = np.ones(asim.lambda_science_range.shape[0])
astat = asim.combine_light(asource, perfect_injection, asim.obs.get_projected_array(), dummy_collected, dosum=False)
for i in range(10):
    # plt.figure()
    # plt.imshow(astat.reshape((50,50,67,8))[:,:,-i,4])
    # plt.colorbar()
    # plt.show()
    plt.figure()
    plt.scatter(asim.injector.vigneting.xx, asim.injector.vigneting.yy, c=astat[:,-1,4], s=5)
    plt.colorbar()
    plt.gca().set_aspect("equal")
    plt.close()


print("Covariance evaluation")


start = time()
asim.point(asim.sequence[0], asim.target, )
screen_age = 0.
integ.reset()
loop2 = 10
loop1 = nloop_covar//10
aseq = []
for i in tqdm(range(loop1)):
    for i in range(loop2):
        screen_age += exptime
        if screen_age>=20. :
            print("generating screen")
            asim.injector.update_screens()
            screen_age = 0.
        myint = asim.make_exposure(asim.src.planet, asim.src.star, asim.diffuse, texp=exptime)
        aseq.append(myint.get_total(n_pixsplit=8.0))
        myint.reset()
    myint.reset()
aseq = np.array(aseq)
akseq = np.einsum("k o, t w o -> t w k", asim.combiner.K, aseq)
obs_mean = np.mean(aseq, axis=0)
obs_std= np.std(aseq, axis=0)
kobs_mean = np.mean(akseq, axis=0)
kobs_std= np.std(akseq, axis=0)
kobs_cov = np.cov(akseq.reshape(akseq.shape[0], -1).T)
t_covar = time() - start
print(f"Covariance evaluation done in {t_covar}")


print("Actual exposures")
start = time()

screen_age = 0
Iout_means = []
KIout_means = []
for i, atime in enumerate(tqdm(asim.sequence)):
    asim.point(atime, asim.target)
    print("generating screen")
    asim.injector.update_screens()
    bseq = []
    for k in range(n_coadds):
        screen_age += t_exp
        if screen_age>=20. :
            screen_age = 0.
        myint = asim.make_exposure(asim.src.planet, asim.src.star, asim.diffuse, texp=exptime)
        bseq.append(myint.get_total(n_pixsplit=8.0))
        myint.reset()
    bseq = np.array(bseq)
    bkseq = np.einsum("k o, t w o -> t w k", asim.combiner.K, bseq)
    
    fig = plt.figure()
    plt.imshow(bkseq, cmap="coolwarm")
    plt.colorbar()
    fig.savefig(f"{outdir}/frame_{i:40d}.png", dpi=50)
    plt.close()
    obs_mean = np.mean(bseq, axis=0)
    kobs_mean = np.mean(bkseq, axis=0)
    Iout_means.append(obs_mean)
    KIout_means.append(kobs_mean)
Iout_means = np.array(Iout_means)
KIout_means = np.array(KIout_means)

t_simul = time() - start
print(f"Simulation done in {t_simul}")





##########################################################################################

import nifits.io.niio as niio
from astropy.table import Table, Column
import astropy.units as units
def save_to_fits(self, Iout=None, KIout=None, Sigma=None, int_times_array=None):
    # def sim2nifits(self,)
    wl_data = np.hstack((self.lambda_science_range[:,None], np.gradient(self.lambda_science_range)[:,None]))
    wl_table = Table(data=wl_data, names=("EFF_WAVE", "EFF_BAND"), dtype=(float, float))
    del wl_data
    oi_wavelength = niio.OI_WAVELENGTH(data_table=wl_table,)
    # oi_wavelength = niio.OI_WAVELENGTH()
    ni_catm = niio.NI_CATM(data_array=self.combiner.Mcn)
    
    mykmat = niio.NI_KMAT(data_array=self.combiner.K)
    oi_target = niio.OI_TARGET.from_scratch()
    oi_target.add_target(target=self.target.name, 
                          raep0=self.target.ra.deg, 
                          decep0=self.target.dec.deg)
    
    from copy import copy
    my_FOV_header = copy(niio.NI_FOV_DEFAULT_HEADER)
    my_FOV_header["NIFITS FOV_TELDIAM"] = self.injector.pdiam
    my_FOV_header["NIFITS FOV_TELDIAM_UNIT"] = "m"
    ni_fov = niio.NI_FOV.simple_from_header(header=my_FOV_header, lamb=self.lambda_science_range,
                                      n=len(self.sequence))
    overhead = 0.3
    n_telescopes = self.ntelescopes
    # dateobs = Time("2035-06-23T00:00:00.000") + times_relative*u.s
    dateobs = self.sequence
    mjds = dateobs.to_value("mjd")
    seconds = (dateobs - dateobs[0]).to_value("s")
    exptimes = np.gradient(seconds) * overhead
    target_id = np.zeros_like(seconds)
    app_index = np.arange(self.ntelescopes)[None,:]*np.ones(len(self.sequence))[:,None]
    target_ids = 0 * np.ones(len(self.sequence))
    if int_times_array is None:
        int_times_array = np.gradient(seconds)
    
    appxy = []
    mod_phas = []
    for atime in self.sequence:
        self.point(atime, self.target)
        g_atmo = self.offband_model.get_phase_science_values(self.pistons)
        g_internal = self.corrector.get_phasor(self.lambda_science_range)
        throughput = self.src.sky.get_downstream_transmission(self.lambda_science_range, )
        total_phasor = np.sqrt(throughput[:,None]) * np.exp(1j*g_atmo) * g_internal
        mod_phas.append(total_phasor)
        appxy.append(self.obs.get_projected_array())
        
    mod_phas = np.array(mod_phas)
    appxy = np.array(appxy)
    
    arrcol = np.ones((len(self.sequence), self.ntelescopes)) * self.injector.collecting
    fov_index = np.ones(len(self.sequence))
    
    app_index         = Column(data=app_index, name="APP_INDEX",
                       unit=None, dtype=int)
    target_id         = Column(data=target_ids, name="TARGET_ID",
                       unit=None, dtype=int)
    times_relative    = Column(data=seconds, name="TIME",
                       unit="", dtype=float)
    mjds              = Column(data=mjds, name="MJD",
                       unit="day", dtype=float)
    int_times         = Column(data=int_times_array, name="INT_TIME",
                       unit="s", dtype=float)
    mod_phas          = Column(data=mod_phas, name="MOD_PHAS",
                       unit="rad", dtype=complex)
    appxy             = Column(data=appxy, name="AP_XY",
                       unit="m", dtype=float)
    arrcol            = Column(data=arrcol, name="COL_AR",
                       unit="m^2", dtype=float)
    fov_index         = Column(data=fov_index, name="FOV_INDEX",
                       unit=None, dtype=int)
    mymod_table = Table()
    mymod_table.add_columns((app_index, target_id, times_relative, mjds,
                            int_times, mod_phas, appxy, arrcol, fov_index))
    mymod_table
    mynimod = niio.NI_MOD(mymod_table)

    outbright = data=asim.combiner.bright[None,:]
    outphot = asim.combiner.photometric[None,:]
    outdark = asim.combiner.dark[None,:]
    ni_iotags = niio.NI_IOTAGS.from_arrays(outbright=outbright, outdark=outdark, outphot=outphot,
                             inpola = inpol, outpola=outpol)
    
    myheader = niio.fits.Header()

    if Iout is not None:
        Iout_table = Table(data=(Iout,), names=("VALUE",), dtype=(float,), )
        myiout = niio.NI_IOUT(data_table=Iout_table, unit=(units.ph/units.s))
        # myiout.name="NI_IOUT"
    if KIout is not None:
        KIout_table = Table(data=(KIout,), names=("VALUE",), dtype=(float,), )
        mykiout = niio.NI_KIOUT(data_table=KIout_table, unit=(units.ph/units.s))
        # mykiout.name="NI_KIOUT"
    if Sigma is not None:
        kcov_header = niio.fits.Header()
        kcov_header["NIFITS SHAPE"] = ("frame (wavelength output)", "The shape of the covariance array.")
        mykcov = niio.NI_KCOV(data_array=Sigma, header=kcov_header, unit=(units.ph/units.s)**2)
        mykcov.name = "NI_KCOV"
    mynifit = niio.nifits(header=myheader,
                        ni_catm=ni_catm,
                        ni_fov=ni_fov,
                        oi_target=oi_target,
                        oi_wavelength=oi_wavelength,
                        ni_mod=mynimod,
                        ni_iout=myiout,
                        ni_kiout=mykiout,
                        ni_kcov=mykcov,
                        ni_kmat=mykmat,
                        ni_iotags=ni_iotags)
    return mynifit

############################################################################################################
mykcov = 1/n_coadds * np.ones(len(asim.sequence))[:,None,None] * kobs_cov[None,:,:]
anifits = save_to_fits(asim, Iout=Iout_means/exptime, KIout=KIout_means, Sigma=mykcov,
                       int_times_array=np.ones(len(asim.sequence)) * exptime * n_coadds)
myhdul = anifits.to_nifits("dummy_name", overwrite=False, writefile=False)
myhdul["PRIMARY"].header.append(("SCIFYSIM DISTANCE", asim.src.distance, "[pc] Distance to target"))
myhdul["PRIMARY"].header.append(("SCIFYSIM R_STAR", asim.src.star.radius, "[R_sun] Radius of star"))
myhdul["PRIMARY"].header.append(("SCIFYSIM T_STAR", asim.src.star.T, "[K] Temperature of star"))
myhdul["PRIMARY"].header.append(("SCIFYSIM R_PLANET", asim.src.planet.radius, "[R_sun] Radius of star"))
myhdul["PRIMARY"].header.append(("SCIFYSIM T_PLANET", asim.src.planet.T, "[K] Temperature of planet"))
myhdul["PRIMARY"].header.append(("SCIFYSIM SEP", asim.src.planet_separation, "[mas] Separation of planet"))
myhdul["PRIMARY"].header.append(("SCIFYSIM PA", asim.src.planet_position_angle, "[deg] Position angle (east of north) possible error"))
myhdul["PRIMARY"].header.append(("SCIFYSIM TARNAME", asim.target.name , ""))
myhdul.writeto(f"{outdir}/result_{mode}.nifits", overwrite=True)

elapsed = time()-start_full
myelapsed = (elapsed/60) * units.min
print(f"Computation done in : {myelapsed:.2f}")
