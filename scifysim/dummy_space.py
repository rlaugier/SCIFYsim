"""
A simple routine to get a generic simulator started with vanilla config file
"""
from pathlib import Path
import scifysim as sf

parent = Path(__file__).parent.absolute()

fname = str(parent/"config/test_space_default.ini")
fname_interp = str(parent/"config/test_space_interp_default.ini")
atarget = "GJ 86 A"
def makesim(fname, target=atarget, compensate_chromatic=True):
    asim = sf.utilities.prepare_all(afile=fname,
                        thetarget=target, update_params=True,
                        seed=1, compensate_chromatic=compensate_chromatic,
                        verbose=False, update_start_end=False)
    asim.point(asim.sequence[0], asim.target,
                        refresh_array=False, disp_override=None)

    asim.context = sf.analysis.spectral_context(asim.config,
                                                verbose=False,
                                                compensate_chromatic=compensate_chromatic)
    diffuse = [asim.src.sky, asim.src.UT, asim.src.warm_optics, asim.src.combiner, asim.src.cold_optics]
    asim.diffuse = diffuse
    return asim

if "__name__" == "__main__":
    asim = makesim(fname)
    interpsim = makesim(fname_interp)


