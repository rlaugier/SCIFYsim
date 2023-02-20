"""
A simple routine to get a generic simulator started with vanilla config file
"""
import scifysim as sf

fname = "config/test_default.ini"
target = "GJ 86 A"
asim = sf.utilities.prepare_all(afile=fname,
                    thetarget=target, update_params=True,
                    seed=1, compensate_chromatic=True,
                    verbose=False)
asim.point(asim.sequence[0], asim.target,
                    refresh_array=False, disp_override=None)

asim.context = sf.analysis.spectral_context(asim.config,
                                            verbose=False)

