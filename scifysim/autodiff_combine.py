import numpy as np
import jax.numpy as jp

def combine_light_jax(self, asource, injected, input_array, collected,
                      dosum=True, dtype=np.float64):
        """
        Computes the combination for a discretized light source object 
        for all the wavelengths of interest.
        Returns the intensity in all outputs at all wavelengths.
        
        **Parameters:**
        
        * asource     : Source object or transmission_emission object
          including ss and xx_r attributes
        * injected    : complex phasor for the instrumental errors
        * input_array  : The projected geometric configuration of the array
          use observatory.get_projected_array()
        * collected   : The intensity across wavelength and the difference source origins
        * dtype       : The data type to use (use np.float32 for maps)
        
        """
        # Ideally, this collected operation should be factorized over all subexps
        intensity = asource.ss * collected[:,None]
        amplitude = jp.sqrt(intensity)
        b = []
        if dtype is np.float64:
            for alpha, beta, anamp in zip(asource.xx_r, asource.yy_r, amplitude.T):
                geom_phasor = self.geometric_phasor(alpha, beta, input_array)
                c = self.combine(injected, geom_phasor, amplitude=anamp)
                b.append(c)
        else:
            for alpha, beta, anamp in zip(asource.xx_r, asource.yy_r, amplitude.T):
                geom_phasor = self.geometric_phasor(alpha, beta, input_array)
                c = self.combine_32(injected, geom_phasor, amplitude=anamp)
                b.append(c)
        b = np.array(b)
        if dosum:
            outputs = jp.sum(jp.abs(b)**2, axis=0)
        else:
            #set_trace()
            outputs = jp.abs(b)**2
        return outputs

def geometric_phasor_jax(self, alpha, beta, anarray):
        """
        Returns the complex phasor corresponding to the locations
        of the family of sources. Backend is jax
        
        **Parameters:**
        
        * alpha         : The coordinate matched to X in the array geometry
        * beta          : The coordinate matched to Y in the array geometry
        * fanarray       : The array geometry (n_input, 2)
        
        **Returns** : A vector of complex phasors
        """
        a = jp.array((alpha, beta), dtype=jp.float64)
        phi = self.k[:,None] * anarray.dot(a)[None,:]
        b = jp.exp(1j*phi)
        return b

def combine_jax(self, inst_phasor, geom_phasor, amplitude=None):
        """
        Computes the output INTENSITY based on the input AMPLITUDE
        or maps)
        
        **Parameters:**
        
        * inst_phasor   : The instrumental phasor to apply to the inputs
          dimension (n_wl, n_input)
        * geom_phasor   : The geometric phasor due to source location
          dimension (n_wl, n_input)
        * amplitude     : Complex amplitude of the spectrum of the source
          dimension (n_wl, n_src)
        
        **Returns** Output complex amplitudes
        
        """
        if amplitude is None:
            amplitude = jp.ones_like(self.k)
        myinput = inst_phasor*geom_phasor*amplitude[:,None]
        output_amps = jp.einsum("ijk,ik->ij",self.combiner.Mcj, myinput)
        return output_amps

def prepare_jax(self):
    """
    Prepares the simulator to run in autodiff mode
    """
    self.combiner.Mcj = jp.array(self.combiner.Mcn)
