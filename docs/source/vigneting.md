

# Injection vigneting

## Basis

instance of `injection.injection_vigneting`

* `self.vigneted_spectrum(wl, (rr, spectrum))` (spectrum collected and transmited)

	- calls `self.vig_func` which is `injector.injection_rate` which is a 2D interpolation

		+ `injector.injection_rate` is computed in the method `compute_injection_function` which is called at the `init` of the `injection_vigneting object`

	- `get_efunc` defined by `give_interpolated` 

		+ Calls `get_injection` 

## To obtain a pupil/image injection function

We must build an alternate version of compute_injection_function

