import numpy as np
from lmfit import Parameters, minimize

import logging

logit = logging.getLogger(__name__)
def set_logging_level(level=logging.WARNING):
    logit.setLevel(level=level)
    logit.info("Setting the logging level to %s"%(level))

class control_loop(object):
    
    def __init__(self, config, section, suffix, f_range=(1e-2, 5e+4),
                 n_dif=None, t_sens=None, act_name=None, t_late=None, t_dac=None,
                 ctrl_name=None):
        """
        PURPOSE:
        ;   General routine for emulating a classical, continuous closed feedback control loop,
        ;   taking into account the noise from the sensor and the actuator,
        ;   and effects from latency and digital-to-analog conversion.
        ;   The controller is modelled as a cascade of PID stages, defined by the keyword PID
        ;
        ; INPUTS:
        ;   freq: array with frequencies on which the input powerspectrum,
        ;   psd :  is defined
        ;
        ; OUTPUT:
        ;   The corrected psd, filter by the closed loop transfer function
        ;
        ; KEYWORDS:
        ;   config:    the parsefile config object used mostly to recover the PID parameters
        ;   section:   the section of the config object containing the controler parameters
        ;   suffix:    the suffix used in the parameter names in this section 
                       (Must be the same for the whole section)
        ;   act_name:  the name of the function used to compute the transfer function of the actuator
        ;   AFTER:     if set, enter the actuator noise at the output. Default is the input.
        ;   BUTTER:    parameters that define the Butterworth filter to suppress known resonances
        ;              for each resonance two parameters are specified: a center frequency f and a quality factor q,
        ;              such that on input BUTTER is a 2N array [f1, q1, f2, q2, ..., fN, qN]
        ;   CALIBRATE: automatic gain calibration so as to obtain a 30� margin (ensuring optimum performance of the loop)
        ;              when CALIBRATE is set, the optimum PID parameters are passed on output through the parameter PID
        ;   CTRL_NAME: the name of the function that describes the open loop tranfer function of the controller
        ;   f_range:   set this keyword to [f_min, f_max] to set the frequency range inside which the PID settings should be calibrated
        ;   INFO:      set this keyword to a value > 2 to print detailed info to the screen
        ;   P_ACT:     noise spectrum of the actuator at frequencies 'freq'
        ;   F_CUT:     cut-off frequency of the actuator
        ;   P_SENS:    noise spectrum of the sensor at frequencies 'freq'
        ;   PID:       On input: scalar or a 3 x N array that defines the transfer function (TF) of the controller as a cascade of PID stages
        ;              If scalar, it contains only a proportional gain
        ;              If 3-element array, it defines the TF as TF(s)=PID[0]+PID[1]/s+PID[2]*s
        ;              If a 3 x N array, TF(s)=(PID[0,0]+PID[1,0]/s+PID[2,0]*s) * (PID[0,1]+PID[1,1]/s+PID[2,1]*s) * ... * (PID[0,N]+PID[1,N]/s+PID[2,N]*s)
        ;              On output, if CALIBRATE is set: optimum PID parameters for the loop
        ;   n_dif      If set, the differential term of the PID part is modified to PID[2]*s / (1D0 + (PID[2]*s)/(N_DIF*PID[0]))
        ;   PLOT:      set this keyword to plot the transfer functions of the sensor, controller, actuator and of the open loop
        ;   t_dac:     hold time for the DA converter
        ;   t_late:    time constant for the latency, the time delay between the controller and the actuator
        ;   t_sens:    time constant for the sensor (previously called the repetition time)
        ;   TF_ACT:    on output: closed-loop transfer function for the actuator noise
        ;   TF_SENS:   on output: closed-loop transfer function for the sensor noise
        ;   TF_CL:     on output: closed-loop transfer function of the whole servo loop
        ;
        ; MODIFICATION HISTORY:
        ;   Version 1.0,  26-NOV-2003, Roland den Hartog, ESA/ESTEC Genie team, rdhartog@rssd.esa.int
        ;   Version 1.1,  04-DEC-2003, OA:  avoid division by 0 for frequency = 0
        ;   Version 1.2,  22-DEC-2003, OA:  implemented gain optimization
        ;   Version 1.3,  15-JAN-2003, OA:  updated parameters for intensity matching loop
        ;   Version 1.4,  08-MAR-2004, OA:  Compensation of VLTI DL resonance peaks implemented (keyword COMP_RES)
        ;   Version 1.5,  22-APR-2004, OA:  New parameter F_CUT for actuator cut-off frequency
        ;   Version 1.6,  17-MAY-2004, RdH: General controller definition allowed
        ;   Version 1.7,  02-JUN-2004, RdH: Corrected the transfer function for the actuator noise. Removed bug.
        ;   Version 1.8,  07-JUN-2004, RdH: Implemented option between applying actuator noise at input of output. Default is input.
        ;   Version 1.9,  09-JUN-2004, RdH: Frequency range for PID calibration can now be entered via a keyword
        ;   Version 1.10, 14-JUN-2004, OA:  Butterworth filtering removed from calibration part
        ;   Version 1.11, 01-JUL-2004, RdH: Give MARGINS several chances to come up with a stability criterion
        ;   Version 1.12, 19-AUG-2004, RdH: Small error in line 259, pointed out by Oswald Wallner, corrected
        ;   Version 1.13, 10-NOV-2004, OA:  DAC term corrected.
        ;   Version 1.14, 21-DEC-2004, RdH: Modified differential term in PID cf. PF's criticism
        ;   Version 1.15, 05-MAR-2005, RdH: Implemented test output for PF's control loop assessment
        ;   Version 1.16, 06-OCT-2009, OA:  Avoid foating underflows
        ;
        ; TESTING:
        ;   22-DEC-2003, OA:  Tested through VLTI_OPD, GENIE_OPD and GENIE_DISP. Frequency range and sampling still to be optimized.
        """
        
        self.min_amargin = 45. # Minimum phase margin: 45 deg
        self.min_pmargin = 12. # Minimum gain margin: 12 dB
        
        self.count = 0
        self.warning = 0
        self.f_range = f_range
        self.no_margins = False
        
        for count in range(3): #Doing loops to extend the frequency range
            # !RL range loop TBD
            if no_margins: # If the no_margin condition was returned last call, increase range
                self.f_range = (0.2*self.f_range[0], 5*self.f_range[1])
            #Sampling on an exponential distribution
            self.f= np.geomspace(min(self_range), max(sel.f_range), 10000)

            # Laplace variable
            s = (0 + 1j*2*np.pi) * self.f

            self.pid = config.get_array(section, "pid"+suffix)
            try:
                self.pid = self.pid.reshape((self.pid.shape[0]//3, 3))
            except:
                print("ERROR - PID must be a 3 x N array (for optimization)")

            if n_dif is not None: # Testing for the n_dif modifier
                # start atomic
                H_ctrl = []
                for PID in pid: # Looping for all the PID controllers
                    if np.isclose(PID[0], 0., atol=1e-12): # Separating the Kp=0 to not divide by 0
                        H = (PID[0] + PID[1]/s \
                                     + PID[2]*s / (1. + (PID[2]*s)/(PID[0]*n_dif)))

                    else: # case where Kp=0: normal PID
                        H = (PID[0] + PID[1]/s + PID[2]*s)
                    H_ctrl.append(H)
                H_ctrl = np.array(H_ctrl)
                # Associate the controllers in a chain
                H_ctrl = np.product(H_ctrl)

            else : # Case with n_dif not set: normal PID
                H_ctrl = []
                for PID in pid:
                    H = (PID[0] + PID[1]/s + PID[2]*s)
                    H_ctrl.append(H)
                H_ctrl = np.array(H_ctrl)
                # Associate the controllers in a chain
                self.H_ctrl = np.product(H_ctrl)
                # !RL Here, it looked like the sourde code did the same thing twice?
            # !RL For the normalization: you end up normalizing by integral gain of the 1st loop???
            # !RL I don't know how IDL manages PID[1] when PID has dimension 2
            # Storing the Transfer Function
            self.H_ctrl = H_ctrl / pid[0,1] 
            del H_ctrl #Just to make sure no one uses it.


            if t_sens is not None:
                self.H_sens = (1. - np.exp(-t_sens*s))/(t_sens * s)
            else: H_sens = 1.
            if act_name is not None:
                self.H_act = act_name(f) # !RL Needs a check for the setup of the relevant function
            else: H_act = 1.
            if t_late is not None:
                self.H_late = np.exp(-t*s)
            else: H_late = 1.
            if t_dac is not None:
                self.H_dac = (1. - np.exp(-t_dac*s))/(t_sens*s)
            else: H_dac = 1.


            # Compute the open-loop tranfer function (with unitary integrator gain)
            self.TF_openloop = self.H_ctrl * self.H_dac * self.H_act* self.H_sens  * self.H_late 
            # Avoid too low values for open loop transfer function (can happen due to SINC functions)
            toolow = np.log10(np.abs(self.TF_openloop)) < -30
            if np.count_nonzero(toolow) is not 0:
                self.TF_openloop[toolow] = self.TF_openloop[toolow] * 1e-30/np.abs(self.TF_openloop[toolow])
            #Do a few bode plots for test:
            #allH = np.array([H_ctrl, H_sens, H_act, self.TF_openloop])
            #bode(self.fallH, labels=["Control", "Sensor", "Actuator", "Open loop"])

            # Initializing the optimization


            params = Parameters()
            params.add("gain", value=1./t_sens, vmin=1e-2/t_sens, vmax=1e2/t_sens)

            sol = minimize(self.margins, params, method="nelder", atol=5e-3)
            #Tolerence of 5e-3 was deemed a good compromise in the original method
            best_gain  = sol.params["gain"].value
        # End of loops to extend the frequency range
        
        # Computing the closed-loop transfer functio
        # re-define the Laplace Variable
        
        s = (0 + 1j*2*np.pi) * self.f
        #This parameter is used to collect the 1/s terms for the controller (to avoid division by 0 for s=0)
        one = np.ones_like(s)
        if ctrl_name is not None:
            H_ctrl = ctrl_name(s)
        else:
            try: pid.shape[0]
            except:
                print("Several PID terms")
            if len(pid.shape) is 1:# This is normally satisfied, even for single PID loops
                for PID in pid:
                    H = []
                    if np.isclose(PID[0], 0., atol=1e-12): # Separating the Kp=0 to not divide by 0
                        H.append(PID[0]*s + PID[1] + PID[2]*s^2/(1.+(PID[2]*s)/(PID[0]*n_dif)))
                    else: # case where Kp=0: normal PID
                        H.append((PID[0]*s + PID[1] + PID[2]*s^2))
                H = np.prod(H)
                one = one*s
                
        raise NotImplementedError("That is where the scifysim code ends")
            
        
        
    def getplots(self, plottype="bode"):
        """
        Makes a few nice plots of for the controler
        """
        tfs = [self.H_act, self.H_dac, self.H_sens, self.H_late]
        tflabels = ["self.H_act", "self.H_dac", "self.H_sens", "self.H_late"]
        ploted = []
        labels = []
        for tf, i in enumerate(tfs):
            try:
                ploted.append(tf)
                labels.append(tflabels[i])
            except nameError:
                print("tf not found")
        
        ploted = np.array(ploted)
        
        if "bode" in plottype:
            bode(self.f, ploted, labels=labels)
        if "nyquist" in plottype:
            nyquist(self.f, ploted, labels=labels)
        
        
    def cleanup(self):
        """
        Can use this method to remove unnecessary memory usage.
        Will remove self.H_sens,  self.H_act, self.H_late, self.H_dac
        and leave only self.H_ctrl and self.TF_openloop
        """
        try: del self.H_act
        except nameError:
            print("act not found")
        try: del self.H_dac
        except nameError:
            print("dac not found")
        try: del self.H_sens
        except nameError:
            print("sens not found")
        try: del self.H_late
        except nameError:
            print("late not found")
        
        
    def margins(self, gain):
        
        
        abs_tf_dB = amp2dB(np.abs(gain * TF_openloop))
        arg_tf_rad = np.angle(TF_openloop)
        # np.angle gives a rad angle between -pi and pi
        arg_tf_deg = np.rad2deg(np.mod(arg_tf_rad + 2*np.pi, 2*np.pi) - 2*np.pi)
        no_amar = False
        no_pmar = False
        
        
        # !RL Trying to fix the phase problem by a floor divide? (If there is even a problem)
        # !RL Check if there was a problem in the first place.
        jump = np.abs(arg_tf_deg - np.roll(arg_tf_deg, 1, axis=0)) # With this method there is 
        #always the jump at the begining or end of the series
        if np.count_nonzero(jump) > 1:
            print("There were", np.count_nonzero(jump)-1, "to account for")
        
        # Looking for the critical point in gain
        cross = np.sign(abs_tf_dB) != np.sign(np.shift(abs_tf_dB, 1, ))
        cross[0] = 0 #With this method there is an artefact on the first term
        #Compute the (most defavorable) phase margin at 0 dB with respect to -180�
        if np.count_nonzero(cross) > 0:
            pmargin = np.min(arg_tf_deg[cross]) + 180
        else:
            no_pmar = True
        
        # Looking for the critical point in phase
        # !RL Double check that we are getting the most relevant crossing
        cross = np.sign(arg_tf_deg + 180. ) != np.sign(np.shift(arg_tf_deg, 1, ) + 180)
        cross[0] = 0 #With this method there is an artefact on the first term
        # Compute the (most defavorable) amplitude margin at -180� with respect to 0 dB
        if np.count_nonzero(cross) > 0:
            amargin = np.max(abs_tf_dB[cross])
        else:
            no_amar = True
        
        self.no_amar = no_amar
        self.no_pmar = no_pmar
        if self.no_amar or self.no_pmar:
            self.no_margins = True
        else:
            self.no_margins = False
            
        if no_amar and no_pmar and self.warning:
            print("WARNING - The frequency domain is too narrow to assess control loop stability:\n"+\
                 "          the magnitude of the open loop TF does not cross the 0 dB line\n"+\
                 "          and its phase does not cross the -180� line within the frequency range,\n"+\
                 "          Proceeding with the default control loop parameters...")
        
        #perror = float(not no_pmar)*np.abs(pmargin - 45.)/45.*(-np.sign(pmargin-45.) + 1.01)*100.
        #aerror = float(not no_amar)*np.abs(amargin + 12.)/12.*(np.sign(amargin+12.) + 1.01)*100.
        #error = perror + aerror
        error = get_error_from_margins(amargin, pmargin, no_pmar=no_pmar, no_amar=no_amar)
        
        return error

def amp2dB(value):
    return 20.*np.log10(value)
def dB2amp(gain):
    return 10**(gain/20.)

def bode(f, H, labels=None, getfig=False):
    """
    Just a shortcut to plot a bunch of Bode plots easily
    f     : Hz n_samples np.array The frequency samples
    H     :    array-like containing either:
            One array of amplitudes
            n_series x n_samples (2D array)
    Plots the Bode plot
    
    Note: future improvements: option to save the plots or return the figure.
    """
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(6,6), dpi=100)
    plt.subplots(121)
    if len(H.shape) is 2:
        logit.info("Plotting an array of %d amplitudes"%(H.shape[0]))
        for myH, i in enumerate(H):
            plt.loglog(f, np.abs(myH), label=labels[i])
    elif len(H.shape) is 1:
        logit.info("Plotting only one TF (amplitude)")
        plt.loglog(f, np.abs(H), label=labels)
    plt.title("Bode plot")
    plt.ylabel("Amplitude")
    plt.legend()
    
    plt.subplots(122)
    if len(H.shape) is 2:
        logit.info("Plotting an array of %d phases"%(H.shape[0]))
        for myH, i in enumerate(H):
            plt.loglog(f, np.angle(myH), label=labels[i])
    elif len(H.shape) is 1:
        logit.info("Plotting only one TF (phase)")
        plt.loglog(f, np.angle(H), label=labels)
    plt.ylabel("Phase [rad]")
    plt.xlabel("Frequency (Hz)")
    plt.show()
    if getfig:
        return fig

def Nyquist(f, H, labels=None, getfig=False):
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(6,6), dpi=100)
    plt.subplots(111, projection="polar")
    if len(H.shape) is 2:
        logit.info("Plotting an array of %d amplitudes"%(H.shape[0]))
        for myH, i in enumerate(H):
            plt.polar(np.angle(myH), np.abs(myH), label=labels[i])
    elif len(H.shape) is 1:
        logit.info("Plotting only one TF (amplitude)")
        plt.polar(np.angle(myH), np.abs(H), label=labels)
    plt.title("Nyquist plot")
    plt.legend()
    plt.show()
    if getfig:
        return fig
    
    
    
def get_error_from_margins(amargin, pmargin, no_pmar=False, no_amar=False):
    #amargin = -12.
    #pmargin = 46.
    logit.debug("amargin %f.2f, pmargin%.1f, no_pmar %r, no_amar"%(amargin, pmargin, no_pmar, no_amar))
    no_pmar = False
    no_amar = False
    perror = float(not no_pmar)*np.abs(pmargin - 45.)/45.*(-np.sign(pmargin-45.) + 1.01)*100.
    aerror = float(not no_amar)*np.abs(amargin + 12.)/12.*(np.sign(amargin+12.) + 1.01)*100.
    error = perror+ aerror
    return error
        
def test_error_margins():
    """
    Test harness for the margins cost function
    Plots a map of the error returned by get_error_from_margins()
    """
    aa, pp = np.meshgrid(np.linspace(-20., 0., 200), np.linspace(0., 60., 200))
    errormap = get_error_from_margins(aa, pp)
    extent=[np.min(aa), np.max(aa), np.max(pp), np.min(pp)]
    plt.figure()
    plt.imshow(np.log10(errormap), extent=extent)
    #plt.contour(aa, extent=extent)
    plt.title(r"$log_{10}(error)$")
    plt.colorbar()
    plt.xlabel("The gain margin [dB]")
    plt.ylabel("The phase margin [deg]")
    plt.show()
                