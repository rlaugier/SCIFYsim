import pathlib
r = pathlib.Path(__file__).parent.absolute()
print(r/"nm_mathar.dat")

    def getimage(self, phscreen=None):
        ''' Produces an image, given a certain number of phase screens,
        and updates the shared memory data structure that the camera
        instance is linked to with that image

        If you need something that returns the image, you have to use the
        class member method get_image(), after having called this method.
        -------------------------------------------------------------------

        Parameters:
        ----------
        - atmo    : (optional) atmospheric phase screen
        - qstatic : (optional) a quasi-static aberration
        ------------------------------------------------------------------- '''

        # nothing to do? skip the computation!

        mu2phase = 4.0 * np.pi / self.wl / 1e6  # convert microns to phase

        phs = np.zeros((self.csz, self.csz), dtype=np.float64)  # phase map
        
        if phscreen is not None:  # a phase screen was provided
            phs += phscreen
            
        wf = np.exp(1j*phs*mu2phase)

        wf *= np.sqrt(self.signal / self.pupil.sum())  # signal scaling
        wf *= self.pupil                               # apply the pupil mask
        self._phs = phs * self.pupil                   # store total phase
        self.fc_pa = self.sft(wf)                      # focal plane cplx ampl
        return self.fc_pa