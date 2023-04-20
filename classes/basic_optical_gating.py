# General imports
import matplotlib.pyplot as plt
import numpy as np
import j_py_sad_correlation as jps
import tifffile as tf
import time
import gc
import glob
import plistlib

# Useful imports
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

# Logger
from .logger import Logger, Colour

# TODO: Write synthetic data class - gen sequence and reference and period
# TODO: Make synthetic data class work with the optical gating class
# TODO: Add minima plots

# SECTION: Helper functions
def v_fitting(y_1, y_2, y_3, x_1, x_2, x_3):
    """
    Fit using a symmetric 'V' function, to find the interpolated minimum for three datapoints y_1, y_2, y_3,
    which are considered to be at coordinates x_1, x_2, x_3

    Args:
        y_1 (float): left y-coordinate
        y_2 (float): middle y-coordinate
        y_3 (float): right y-coordinate
        x_1 (float): left x-coordinate
        x_2 (float): middle x-coordinate
        x_3 (float): right x-coordinate

    Returns:
        tuple: m_1_l, m_3_l, c_1_l, c_3_l, m_3_r, m_1_r, c_1_r, c_3_r, x_l, y_l, x_r, y_r
    """    
    # Fit using a symmetric 'V' function, to find the interpolated minimum for three datapoints y_1, y_2, y_3,
    # which are considered to be at coordinates x_1, x_2, x_3
    
    # Calculate the gradients of our two lines
    m_1 = (y_2 - y_1) / (x_2 - x_1)
    m_3 = (y_2 - y_3) / (x_2 - x_3)

    # Line 1
    m_3_l = - m_1
    m_1_l = m_1
    c_1_l = y_1 - m_1_l * x_1
    c_3_l = y_3 - m_3_l * x_3

    x_l = (c_3_l - c_1_l) / (m_1_l - m_3_l)
    y_l = m_1_l * x_l + c_1_l

    # Line 2
    m_1_r = - m_3
    m_3_r = m_3
    c_1_r = y_1 - m_1_r * x_1
    c_3_r = y_3 - m_3_r * x_3


    x_r = (c_3_r - c_1_r) / (m_1_r - m_3_r)
    y_r = m_1_r * x_r + c_1_r

    return m_1_l, m_3_l, c_1_l, c_3_l, m_3_r, m_1_r, c_1_r, c_3_r

def v_fitting_standard(y_1, y_2, y_3):
    # Fit using a symmetric 'V' function, to find the interpolated minimum for three datapoints y_1, y_2, y_3,
    # which are considered to be at coordinates x=-1, x=0 and x=+1
    if y_1 > y_3:
        x = 0.5 * (y_1 - y_3) / (y_1 - y_2)
        y = y_2 - x * (y_1 - y_2)
    else:
        x = 0.5 * (y_1 - y_3) / (y_3 - y_2)
        y = y_2 + x * (y_3 - y_2)

    return x, y

def adapted_v(x, I_1, I_2, x_n, c):
    return np.abs(I_1 * (x - x_n) + 0.5 * I_2 * (x**2 - x_n**2)) + c

# !SECTION

# SECTION: Basic optical gating
# Simplified version of the optical gating class for testing and implementing new features
class BasicOpticalGating(): 
    # TODO: Add setting to use drift correction
    # TODO: Add option to scale frame values to a range so different sequences can be compared
    # TODO: Add method to generate reference sequence and determine reference period
    # TODO: Rename phase to frame

    # Use basic_optical_gating_plotter to plot results
    # NOTE: Uses variables named _phases but actually returns frame estimates

    def __init__(self) -> None:
        """ Class for performing basic optical gating on a sequence of images.
            Run in order (*optional):
                set_sequence
                set_reference_sequence
                get_bias_correction*
                get_sads
                get_phases*
                get_delta_phases*
                get_unwrapped_phases*
            Alternatively call run() to run all steps in order.
        """     
        # Initialise our logger
        self.logger = Logger("BOG")
        self.logger.set_quiet()

        # Initialise vars
        self.drift = (0, 0)
        self.reference_period = None
        self.reference_framerates = None
        self.framerate_correction = False
        self.roi = None
        # FIXME: bias correction is defaulted to true to stop errors but should sometimes be false
        self.has_run = {
            "set_sequence" : False,
            "set_reference_sequence" : False,
            "get_bias_correction": True,
            "get_sads" : False,
            "get_phases" : False,
            "get_delta_phases" : False,
            "get_unwrapped_phases" : False,
            "set_region_of_interest" : False
        }

    def __repr__(self): 
        return_string = ""
        return_string += "Basic optical gating class\n"
        for key, value in self.has_run.items():
            return_string += f"\t{key}:"
            if value:
                return_string += f"{Colour.GREEN} True{Colour.END}\n"
            else:
                return_string += f"{Colour.RED} False{Colour.END}\n"

        return return_string

    def _check_if_run(self, check_name = None):
        # Checks if the relevant functions have been run

        if self.has_run[check_name] == False:
            self.logger.print_message("ERROR", f"Run {check_name} before running this function")
            raise Exception(f"Run {check_name} before running this function")
        else:
            return True
                
    def _load_data(self, data_src, frames = None):
        if isinstance(data_src, str):
            if frames == None:
                data = tf.imread(data_src)
            else:
                data = tf.imread(data_src, key = frames)
        elif isinstance(data_src, np.ndarray):
            data = data_src
        else:
            self.logger.print_message("ERROR", "Data source must be a string or numpy array")
            raise Exception("Data source must be a string or numpy array")

        return data
    
    def set_region(self, roi):
        """ Set the region of interest

            Args:
                roi (tuple): Tuple of the form (x, y, width, height)
        """        
        self.roi = roi
        self.has_run["set_region_of_interest"] = True

    def set_sequence(self, sequence_src, frames = None):
        """ Set the sequence to run optical gating on

            Args:
                sequence_src (str): Path to the sequence
                frames (str, optional): Range of frames to load. Defaults to all.
        """        

        self.logger.print_message("INFO", "Loading sequence...")
        
        self.sequence = self._load_data(sequence_src, frames)

        if self.sequence.shape[0] <= 1:
            self.logger.print_message("ERROR", "Sequence must have at least one frame")
            raise Exception("Sequence must have at least one frame")
        
        self.has_run["set_sequence"] = True
        self.logger.print_message("SUCCESS", f"Sequence loaded with {self.sequence.shape[0]} frames")

    def set_reference_sequence(self, reference_sequence_src):
        """ Set the reference sequence to use for optical gating

                Args:
                    reference_sequence_src (str): Path to the reference sequence
        """        
        # Load the reference sequence
        self.logger.print_message("INFO", "Loading reference sequence...")

        #self.reference_sequence = tf.imread(reference_sequence_src)
        self.reference_sequence = self._load_data(reference_sequence_src)

        # Generate a default bias correction
        self.cs = CubicSpline(np.arange(self.reference_sequence.shape[0]), np.arange(self.reference_sequence.shape[0]))

        if self.reference_sequence.shape[0] <= 1:
            self.logger.print_message("ERROR", "Reference sequence must have at least one frame")
            raise Exception("Reference sequence must have at least one frame")
        
        self.has_run["set_reference_sequence"] = True
        self.logger.print_message("SUCCESS", f"Reference sequence loaded with {self.reference_sequence.shape[0]} frames")
    
    def set_reference_period(self, reference_period = None):
        """
        Set the reference period either as a float or a reference to a file containing the reference period

        Args:
            reference_period ((float, string), optional): Reference period as output from open optical gating. Defaults to reference sequence length.

        Raises:
            Exception: _description_
        """            
        if isinstance(reference_period, (int, float)):
            self.logger.print_message("INFO", f"Setting reference period to {self.reference_period}")
            self.reference_period = reference_period
        elif isinstance(reference_period, str):
            self.reference_period = np.loadtxt(reference_period)
            self.logger.print_message("INFO", f"Setting reference period to {self.reference_period}")
        else:
            self.logger.print_message("ERROR", "Reference period must be a float or string")
            raise Exception("Reference period must be a float or string")
    
    def set_reference_plist(self, plist_src_folder):
        # NOTE: Temporary rushed code
        # Loads up a plist file corresponding to the reference sequence
        # Will be used to apply the bias correction based upon the reference
        # sequence framerate
        # This is just a test to see if this is what is causing the 
        # bias correction to fail.

        files = f"{plist_src_folder}/*plist"
        framerates = []
        for file in glob.glob(files):
            with open(file, 'rb') as f:
                plist = plistlib.load(f)
                for i in range(len(plist["frames"])):
                    framerates.append(plist["frames"][i]["est_framerate"])
        self.reference_framerates = np.cumsum(framerates)

    def set_reference_framerates(self, reference_framerates):
        self.reference_framerates = reference_framerates

    def _pre_process_reference_sequence_bias_correction(self, frame_number = None):
        # Pre-process the reference sequence for bias correction
        # NOTE: This is a separate function so that it can be overriden by subclasses
        if self.roi is not None:
            x, y, width, height = self.roi
            return self.reference_sequence[:, y:y+height, x:x+width].astype(np.int32)
        else:
            return self.reference_sequence.astype(np.int32)
    
    def set_framerate_correction(self, framerate_correction):
        self.framerate_correction = True
    
    def get_bias_correction(self):
        """ Get the bias correction. Attempts to reduce biasing caused by varying rate of change in
            the sum of absolute differences between neighbouring frames which causes V-fitting to be biased.
            Currently in testing but shown to be effective in reducing bias in synthetic data.
        """       
        if self._check_if_run("set_reference_sequence") == True:   
            self.logger.print_message("INFO", "Calculating bias correction...")
            # Pre-process the reference sequence - used for subclassing
            reference_sequence = self._pre_process_reference_sequence_bias_correction()

            # Get the difference between neighbouring reference frames
            diffs = [0]
            for i in range(1, reference_sequence.shape[0]):
                diffs.append(np.sum(np.abs(reference_sequence[i] - reference_sequence[i - 1])))
            diffs = np.array(diffs)

            if np.any(diffs < 0):
                self.logger.print_message("WARNING", "Negative diffs found")

            # Get running sum of differences
            diffs = np.cumsum(diffs)

            # Normalise - not strictly neccessary but makes it easier to compare plots
            diffs = diffs / diffs[-1]
            diffs *= (diffs.shape[0] - 1)

            # Create a cubic spline to interpolate the differences
            if self.framerate_correction == True:
                if self.reference_framerates is not None:
                    xs = self.reference_framerates
                    xs -= self.reference_framerates[0]
                    xs = (len(diffs) - 1) * self.reference_framerates / np.max(self.reference_framerates)
                else:
                    self.logger.print_message("ERROR", "Reference framerates not set")
                    raise Exception("Reference framerates not set")
            else:
                xs = np.arange(len(diffs))
                                
            self.cs = CubicSpline(xs, diffs)

            self.has_run["get_bias_correction"] = True
            self.logger.print_message("SUCCESS", "Bias correction complete")

    def _preprocess_frame(self, frame_number):
        # Pre-process the sequence
        # NOTE: This is a separate function so that it can be overriden by subclasses
        if self.roi is not None:
            x, y, width, height = self.roi
            return self.sequence[frame_number][y:y+height, x:x+width]
        else:
            return self.sequence[frame_number]
    
    def _preprocess_reference_sequence(self, frame_number = None):
        # Pre-process the reference sequence
        # NOTE: This is a separate function so that it can be overriden by subclasses
        if self.roi is not None:
            x, y, width, height = self.roi
            return self.reference_sequence[:,y:y+height, x:x+width]
        else:
            return self.reference_sequence
        
    def get_sad(self, frame, reference_sequence, use_jps = True, drift_correct = True):
        """ Get the sum of absolute differences for a single frame and our reference sequence.
            NOTE: In future it might be better to pass this function the frame and reference sequence
            instead of the frame number.

            Args:
                frame_number (int): frame number to get the SAD for
                use_jps (bool, optional): Whether to use Jonny's SAD code which is significantly faster but requires correct dtype. Defaults to True.
                reference_sequence (np.array, optional): The reference sequence. Defaults to None.

            Returns:
                np.array: The sum of absolute differences between the frame and the reference sequence
        """
        # Get the sum of absolute differences between two frames
        # NOTE: Copied from OOG
        if drift_correct:
            dx, dy = self.drift
            rectF = [0, frame.shape[0], 0, frame.shape[1]]  # X1,X2,Y1,Y2
            rect = [
                0,
                reference_sequence[0].shape[0],
                0,
                reference_sequence[0].shape[1],
            ]  # X1,X2,Y1,Y2

            if dx <= 0:
                rectF[0] = -dx
                rect[1] = rect[1] + dx
            else:
                rectF[1] = rectF[1] - dx
                rect[0] = dx
            if dy <= 0:
                rectF[2] = -dy
                rect[3] = rect[3] + dy
            else:
                rectF[3] = rectF[3] - dy
                rect[2] = +dy

            frame_cropped = frame[rectF[0] : rectF[1], rectF[2] : rectF[3]]
            reference_frames_cropped = [
                f[rect[0] : rect[1], rect[2] : rect[3]] for f in reference_sequence
            ]
        else:
            frame_cropped = frame
            reference_frames_cropped  = reference_sequence

        if use_jps:
            sad = jps.sad_with_references(frame_cropped, reference_frames_cropped)
        else:
            sad = []
            for i in range(len(reference_frames_cropped)):
                sad.append(np.sum(np.abs(frame_cropped - reference_frames_cropped[i])))
            sad = np.array(sad)
            
            """reference_frames_cropped = np.array(reference_frames_cropped).astype(np.int)
            frame_cropped = np.array(frame_cropped).astype(np.int)
            sad = reference_frames_cropped - frame_cropped
            sad = np.abs(sad)
            sad = np.sum(sad, axis = (1,2))"""

        if drift_correct:
            dx, dy = self.drift
            self.drift = update_drift_estimate(frame, reference_sequence[np.argmin(sad)], (dx, dy))
            self.drifts.append(self.drift)

        return sad

    def get_sads(self, drift_correct = True): 
        """ Get the SADs for every frame in our sequence"""       
        if self._check_if_run("set_sequence") == True and self._check_if_run("set_reference_sequence") == True:    
            self.logger.print_message("INFO", "Calculating SADs...")      
            self.sads = []
            self.drifts = []

            # Calculate SADs
            reference_sequence = self._preprocess_reference_sequence()
            for i in range(self.sequence.shape[0]):
                frame = self._preprocess_frame(i)
                reference_sequence = self._preprocess_reference_sequence(i)
                self.sads.append(self.get_sad(frame, reference_sequence, use_jps = True, drift_correct = drift_correct))

            self.sads = np.array(self.sads)

            self.has_run["get_sads"] = True
            self.logger.print_message("SUCCESS", "SADs calculated")

            return self.sads
        else:
            self.logger.print_message("ERROR", "Cannot calculate SADs without a sequence and reference sequence")
    
    def get_phases(self):
        """ Get the phase estimates for our sequence"""        
        if self._check_if_run("get_sads") == True:
            self.logger.print_message("INFO", "Calculating phases...")
            self.phases = []
            self.frame_minimas = []

            # Track number of frames outside the reference period
            outside_errors = []
            
            # Get the frame estimates
            for i, sad in enumerate(self.sads):           
                # Get the frame estimate
                frame_minima = np.argmin(sad[2:-2]) + 2
                y_1 = sad[frame_minima - 1]
                y_2 = sad[frame_minima]
                y_3 = sad[frame_minima + 1]
                x_1 = frame_minima - 1
                x_2 = frame_minima
                x_3 = frame_minima + 1

                if y_1 > y_2 and y_3 > y_2:
                    # Do v-fitting and bias correction
                    # NOTE: This is an incredibly inneficient way of doing this - if implementing in production code we'll find an analytic 
                    # expression for the bias correction

                    # Get the v-fitting parameters
                    m_1_l, m_3_l, c_1_l, c_3_l, m_3_r, m_1_r, c_1_r, c_3_r = v_fitting(y_1, y_2, y_3, x_1, x_2, x_3)

                    # Generate a sequence used for finding the minima
                    cdiff_l = self.cs(x_2) - self.cs(x_1)
                    cs_normalised_l = lambda x: (self.cs(x) - self.cs(x_2)) / cdiff_l
                    cs_normalised_r = lambda x: (self.cs(x) - self.cs(x_3)) / cdiff_l
                    xs = np.linspace(x_1, x_3, 10000)

                    # Do fitting using corrected V
                    v_l_c_l = m_1_l * (x_2 + cs_normalised_l(xs)) + c_1_l
                    v_r_c_l = m_3_l * (x_3 + cs_normalised_r(xs)) + c_3_l
                    # Get minima location
                    x_min_l = (xs)[np.argmin(abs(v_l_c_l - v_r_c_l))]
                    y_min_l = m_3_l * (x_3 + cs_normalised_r(x_min_l)) + c_3_l
                    """# New method
                    # NOTE: Slower than current method
                    f1 = lambda x: m_1_l * (x_2 + cs_normalised_l(x)) + c_1_l
                    f2 = lambda x: m_3_l * (x_3 + cs_normalised_r(x)) + c_3_l
                    func_l = lambda x: np.abs((f1(x)) - (f2(x)))
                    min_l = scipy.optimize.minimize_scalar(func_l, bounds = (x_1, x_3), method = "bounded")
                    x_min_l = min_l.x
                    y_min_l = f1(x_min_l)"""

                    # Do fitting using corrected V
                    cdiff_r = self.cs(x_3) - self.cs(x_2)
                    cs_normalised_l = lambda x: (self.cs(x) - self.cs(x_1)) / cdiff_r
                    cs_normalised_r = lambda x: (self.cs(x) - self.cs(x_2)) / cdiff_r
                    
                    v_l_c_r = m_1_r * (x_1 + cs_normalised_l(xs)) + c_1_r
                    v_r_c_r = m_3_r * (x_2 + cs_normalised_r(xs)) + c_3_r
                    # Get minima location
                    x_min_r = (xs)[np.argmin(abs(v_l_c_r - v_r_c_r))]
                    y_min_r = m_3_r * (x_2 + cs_normalised_r(x_min_r)) + c_3_r
                    """# New method
                    # NOTE: Slower than current method
                    f1 = lambda x: m_1_r * (x_1 + cs_normalised_l(x)) + c_1_r
                    f2 = lambda x: m_3_r * (x_2 + cs_normalised_r(x)) + c_3_r
                    func_r = lambda x: np.abs((f1(x)) - (f2(x)))
                    min_r = scipy.optimize.minimize_scalar(func_r, bounds = (x_1, x_3), method = "bounded")
                    x_min_r = min_r.x
                    y_min_r = f2(x_min_r)"""

                    if y_min_l < y_min_r:
                        self.phases.append(x_min_l - 2)
                    else:
                        self.phases.append(x_min_r - 2)

                    self.frame_minimas.append(frame_minima - 2)
                else:
                    outside_errors.append(i)
                    self.phases.append(frame_minima - 2)
                    self.frame_minimas.append(frame_minima - 2)

            if len(outside_errors) > 0:
                self.logger.print_message("WARNING", f"Minima for frame(s) {outside_errors} outside of the valid range.")


            self.phases = np.array(self.phases)
            self.frame_minimas = np.array(self.frame_minimas)
            self.drifts = np.array(self.drifts)

            self.has_run["get_phases"] = True
            self.logger.print_message("SUCCESS", "Phases calculated")

    def get_delta_phases(self):
        """ Get the delta phases for our sequence.
            TODO: Do checks for reference period earlier
        """        
        if self._check_if_run("get_phases") == True:
            self.logger.print_message("INFO", "Calculating delta phases...")
            # If reference period isn't set then use the length of our referene sequence as a guide
            if self.reference_period == None:
                self.logger.print_message("WARNING", "No reference period specified, using reference sequence frames as period.")
                self.reference_period = self.reference_sequence.shape[0] - 4
                self.logger.print_message("INFO", f"Reference period set to {self.reference_period}")
            elif self.reference_period < 0:
                self.logger.print_message("ERROR", "Reference period must be positive.")
                raise ValueError("Reference period must be positive.")
            elif (self.reference_period > self.reference_sequence.shape[0] - 4 + 1) or (self.reference_period < self.reference_sequence.shape[0] - 4 - 1):
                self.logger.print_message("WARNING", f"Reference period significantly different to reference sequence length: {self.reference_sequence.shape[0] - 4} vs {self.reference_period}.")

            # Initialise the delta frames
            self.delta_phases = []

            # Get our delta frames
            for i in range(1, self.phases.shape[0]):
                delta_frame = (self.phases[i] - self.phases[i - 1]) % self.reference_period - self.reference_period
                
                # Correct for wraparound
                while delta_frame < -self.reference_period / 2:
                    delta_frame += self.reference_period

                # Append the delta frame
                self.delta_phases.append(delta_frame)

            self.delta_phases = np.array(self.delta_phases)

            self.has_run["get_delta_phases"] = True
            self.logger.print_message("SUCCESS", "Delta phases calculated")
    
    def get_unwrapped_phases(self):
        """ Get the unwrapped phases"""        
        if self._check_if_run("get_delta_phases") == True:
            self.logger.print_message("INFO", "Unwrapping phases...")
            self.unwrapped_phases = [0]
            for i in range(0, self.delta_phases.shape[0]):
                self.unwrapped_phases.append(self.unwrapped_phases[-1] + self.delta_phases[i])
            self.unwrapped_phases = np.array(self.unwrapped_phases)

            self.has_run["get_unwrapped_phases"] = True
            self.logger.print_message("SUCCESS", "Unwrapped phases calculated")
    
    def clear_memory(self):
        """Deletes the reference to our sequence to free up memory

            Returns:
                Bool: Returns True if successful
        """       
        if self._check_if_run("set_sequence") == True:
            self.logger.print_message("INFO", "Clearing memory...")
            try:
                del self.sequence
                self.has_run["set_sequence"] = False
                gc.collect()
                return True
            except:
                self.logger.print_message("ERROR", "Failed to clear memory.")
                return False
    
    def run(self, bias_correct = False, clear_memory = False, drift_correct = True):
        """ Run the full optical gating on our sequence.
            Outputs the phases, delta phases and unwrapped phases.
            Optionally run bias correct and clear memory after completion

            Args:
                bias_correct (bool, optional): Run with bias correction. Defaults to False.
                clear_memory (bool, optional): Delete our sequence from memory. Defaults to False.
        """        
        if bias_correct:
            self.get_bias_correction()
        self.get_sads(drift_correct = drift_correct)
        self.get_phases()
        self.get_delta_phases()
        self.get_unwrapped_phases()

        if clear_memory == True:
            if self.clear_memory():
                self.logger.print_message("WARNING", "Cleared memory, rerun set_sequence if SADs need to be regenerated.")


        self.logger.print_message("SUCCESS", "Finished processing sequence.")
# !SECTION

# SECTION: Drift correction
# NOTE: This is just copied from Jonny's code
# Need to get understanding of what it does
# TODO: Reimpliment this for understanding
# TODO: Move this into the BOG class
def update_drift_estimate(frame0, bestMatch0, drift0):
    """ Determine an updated estimate of the sample drift.
        We try changing the drift value by ±1 in x and y.
        This just calls through to the more general function get_drift_estimate()

        Parameters:
            frame0         array-like      2D frame pixel data for our most recently-received frame
            bestMatch0     array-like      2D frame pixel data for the best match within our reference sequence
            drift0         (int,int)       Previously-estimated drift parameters
        Returns
            new_drift      (int,int)       New drift parameters
        """
    return get_drift_estimate(frame0, [bestMatch0], dxRange=range(drift0[0]-1, drift0[0]+2), dyRange=range(drift0[1]-1, drift0[1]+2))

def get_drift_estimate(frame, refs, matching_frame=None, dxRange=range(-30,31,3), dyRange=range(-30,31,3)):
    """ Determine an initial estimate of the sample drift.
        We do this by trying a range of variations on the relative shift between frame0 and the best-matching frame in the reference sequence.

        Parameters:
            frame          array-like      2D frame pixel data for the frame we should use
            refs           list of arrays  List of 2D reference frame pixel data that we should search within
            matching_frame int             Entry within reference frames that is the best match to 'frame',
                                        or None if we don't know what the best match is yet
            dxRange        list of int     Candidate x shifts to consider
            dyRange        list of int     Candidate y shifts to consider

        Returns:
            new_drift      (int,int)       New drift parameters
        """
    # frame0 and the images in 'refs' must be numpy arrays of the same size
    assert frame.shape == refs[0].shape

    # Identify region within bestMatch that we will use for comparison.
    # The logic here basically follows that in phase_matching, but allows for extra slop space
    # since we will be evaluating various different candidate drifts
    inset = np.maximum(np.max(np.abs(dxRange)), np.max(np.abs(dyRange)))
    rect = [
            inset,
            frame.shape[0] - inset,
            inset,
            frame.shape[1] - inset,
            ]  # X1,X2,Y1,Y2

    candidateShifts = []
    for _dx in dxRange:
        for _dy in dyRange:
            candidateShifts += [(_dx,_dy)]

    if matching_frame is None:
        ref_index_to_consider = range(0, len(refs))
    else:
        ref_index_to_consider = [matching_frame]

    # Build up a list of frames, each representing a window into frame with slightly different drift offsets
    frames = []
    for shft in candidateShifts:
        dxp = shft[0]
        dyp = shft[1]

        # Adjust for drift and shift
        rectF = np.copy(rect)
        rectF[0] -= dxp
        rectF[1] -= dxp
        rectF[2] -= dyp
        rectF[3] -= dyp
        frames.append(frame[rectF[0] : rectF[1], rectF[2] : rectF[3]])

    # Compare all these candidate shifted images against each of the candidate reference frame(s) in turn
    # Our aim is to find the best-matching shift from within the search space
    best = 1e200
    for r in ref_index_to_consider:
        sad = jps.sad_with_references(refs[r][rect[0] : rect[1], rect[2] : rect[3]], frames)
        smallest = np.min(sad)
        if (smallest < best):
            bestShiftPos = np.argmin(sad)
            best = smallest

    return (candidateShifts[bestShiftPos][0],
            candidateShifts[bestShiftPos][1])
# !SECTION

# SECTION: Plotter class
class BasicOpticalGatingPlotter():
    # TODO: Add plot settings
    # Clean up histogram plots - maybe move them into one?

    def __init__(self, bogs, names = None) -> None:
        """
        Plots instances of the basic optical gating class

        Args:
            bogs (object or list): single or list of BOG objects
            names (string, optional): Names of the BOG objects. Defaults to None.
        """        
        # Initialise the logger
        self.logger = Logger("OGP")
        self.logger.set_quiet()

        # Fill in our arrays of basic optical gating instances and names
        if type(bogs) is list:
            self.bogs = bogs
        else:
            self.bogs = [bogs]

        if names == None:
            self.names = [f"{i}" for i in range(len(self.bogs))]
        elif type(names) is list:
            self.names = names
        else:
            self.names = [names]

        # Initialise vars
        self.n = len(self.bogs)
        self.figsize = (14,8)
        
    def _check_if_run(self, check_name = None):
        # Checks if the relevant functions have been run within the basic optical gating instances

        not_run_count = 0
        for i in range(self.n):
            if self.bogs[i].has_run[check_name] == False:
                self.logger.print_message("ERROR", f"Run {check_name} in basic optical gating with name \"{self.names[i]}\" before plotting")
                not_run_count += 1
        if not_run_count >= 1:
            raise Exception(f"Not all basic optical gating instances have been run with {check_name}")

    def _begin_plot(self, plot_title):
        # Printo to log
        self.logger.print_message("INFO", f"Plotting {plot_title}")

        # Set theme
        #plt.style.use("seaborn-v0_8-poster")
        #plt.style.use("rose-pine-dawn")

        # Setup our figure
        plt.figure(figsize = self.figsize)
        plt.title(plot_title)

    def set_figsize(self, figsize):
        """
        Set the figure size for the plots

        Args:
            figsize (tuple): tuple of width and height. e.g (14,8)
        """        
        self.figsize = figsize

    def plot_sads(self, frame_number = None):
        """
        Plot the SADs for a given frame

        Args:
            frame_number (int, optional): The frame to plot SADs for. Defaults to None.
        """        
        self._check_if_run("get_sads")
        self._check_if_run("get_phases")

        if frame_number == None:
            frame_number = 0
            self.logger.print_message("WARNING", "No frame number specified, using 0")

        # Plot the SADs for a given frame
        self._begin_plot(f"SADs against frame number for frame: {frame_number}")
        for i in range(self.n):
            plt.scatter(range(self.bogs[i].sads[frame_number].shape[0]), self.bogs[i].sads[frame_number], label = self.names[i], c = f"C{i}")
            argmin = np.argmin(self.bogs[i].sads[frame_number][2:-2]) + 2
            plt.scatter([argmin], [self.bogs[i].sads[frame_number][argmin]], c = f"C{i}", alpha = 0.5, s = 100)
            plt.axvline(self.bogs[i].phases[frame_number] + 2, c = f"C{i}", ls = "--")
        plt.axvline(2, c = "black", ls = "--")
        plt.axvline(self.bogs[0].sads[frame_number].shape[0] - 3, c = "black", ls = "--")
        plt.xticks(range(self.bogs[i].sads[frame_number].shape[0]))
        plt.xlabel("Frame number")
        plt.ylabel("SAD")
        plt.legend()
        plt.show()

    def plot_phases(self):
        """ Plots the phases against frame number"""        
        self._check_if_run("get_phases")

        # Plot the phases
        self._begin_plot("Phases against frame number")
        for i in range(self.n):
            plt.scatter(self.bogs[i].unwrapped_phases, self.bogs[i].phases, label = self.names[i])
        plt.xlabel("Frame number")
        plt.ylabel("Phase")
        plt.legend()
        plt.show()

    def plot_delta_phases(self):
        """ Plots the delta phases against frame number"""        
        self._check_if_run("get_delta_phases")

        self._begin_plot("Delta phases against frame number")
        for i in range(self.n):
            plt.scatter(self.bogs[i].unwrapped_phases[0:-1], self.bogs[i].delta_phases, label = self.names[i], s = 2)
        plt.xlabel("Unwrapped phase")
        plt.ylabel("Delta phase")
        plt.legend()
        plt.show()

    def plot_delta_phases_phases(self):
        """ Plots the delta phases against phases"""
        self._check_if_run("get_delta_phases")

        self._begin_plot("Delta phases against phases")
        for i in range(self.n):
            plt.scatter(self.bogs[i].phases[0:-1], self.bogs[i].delta_phases, label = self.names[i], s = 5)
        plt.axhline(1, color = "black", linestyle = "--", label = "Expected delta phase")
        plt.xlabel("Phase")
        plt.ylabel("Delta phase")
        plt.legend()
        plt.show()

    def plot_bias_correction(self, subtract = False):
        """
        Plots the bias correction against frame number

        Args:
            subtract (bool, optional): Whether to subtract a linear bias correction. Defaults to False.
        """       
        self._check_if_run("get_bias_correction")

        self._begin_plot("Bias correction against frame number")
        xs = np.linspace(0, len(self.bogs[0].sads[0]), 1000)
        for i in range(self.n):
            if subtract:
                plt.plot(xs, self.bogs[i].cs(xs) - xs, label = self.names[i])
            else:
                plt.plot(xs, self.bogs[i].cs(xs), label = self.names[i])
        plt.xticks(range(self.bogs[0].reference_sequence.shape[0]))
        plt.xlabel("Frame number")
        plt.ylabel("Non-normalised bias correction")
        plt.legend()
        plt.show()

    def plot_unwrapped_phases(self, subtract = False):
        """
        Plots the unwrapped phases against frame number

        Args:
            subtract (bool, optional): Whether to subtract a linear unwrapped phase from the unwrapped phases. Defaults to False.
        """        
        # TODO - Add a proper linear fit to the unwrapped phases
        self._check_if_run("get_unwrapped_phases")

        self._begin_plot("Unwrapped phases against frame number")
        for i in range(self.n):
            xs = range(self.bogs[i].unwrapped_phases.shape[0])
            if subtract:
                plt.scatter(xs, self.bogs[i].unwrapped_phases - xs, label = self.names[i], s = 5)
            else:
                plt.scatter(xs, self.bogs[i].unwrapped_phases, label = self.names[i], s = 5)
        plt.xlabel("Frame number")
        plt.ylabel("Unwrapped phase")
        plt.legend()
        plt.show()

    def plot_subframe_histogram(self, frame_number = None, bins = 20, skip = 0):
        """
        _summary_

        Args:
            frame_number (_type_, optional): Frame to plot, if None then plots every frame on one plot. Defaults to None.
            bins (int, optional): Number of bins for the histogram. Defaults to 20.
            skip (int, optional): Number of frames to skip. Useful if we want to ignore reference frames. Defaults to 0.
        """        
        self._check_if_run("get_phases")

        for i in range(self.n):
            if frame_number != None:
                subframes = []
                #for j, frame_minima in enumerate(self.bogs[i].frame_minimas):
                for j in range(skip, self.bogs[i].frame_minimas.shape[0]):
                    if self.bogs[i].frame_minimas[j] == frame_number:
                        subframes.append(self.bogs[i].frame_minimas[j] - self.bogs[i].phases[j])
            else:
                subframes = self.bogs[i].phases - self.bogs[i].frame_minimas

            self._begin_plot(f"Subframe histogram for {self.names[i]}")
            plt.hist(subframes, bins = bins)
            plt.xlabel("Subframe")
            plt.ylabel("Count")
            plt.show()

    def plot_subframe_histogram_full(self, bins = 100):
        """
        Plot the subframe histogram across every phase

        Args:
            bins (int, optional): _description_. Defaults to 100.
        """        
        self._check_if_run("get_phases")
        for i in range(len(self.bogs)):
            self._begin_plot(f"Subframe histogram for {self.names[i]}")
            plt.hist(self.bogs[i].phases, bins = bins)
            plt.xlabel("Frame")
            plt.ylabel("Counts")
            plt.show()

        """self._begin_plot("Subframe histogram for all")
        hists = []
        for i in range(len(self.bogs)):
            hists.append(self.bogs[i].phases)
        plt.hist(hists, bins = bins, label = self.names, histtype = "barstacked")
        plt.axhline(self.bogs[0].phases.shape[0] / bins, c = "black", linestyle = "--", label = "Expected count")
        plt.xlabel("Frame")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()"""

    def plot_video(self):
        self._check_if_run("set_sequence")

        for i in range(self.n):
            x, y, width, height = self.bogs[i].roi
            plt.imshow(self.bogs[i].sequence[0,x:x + width,y:y + height])
            plt.show()

    def plot_all(self, subtract = False):
        """
        Plots all the plots

        Args:
            subtract (bool, optional): Whether to do linear fir on bias correction and unwrapped phases. Defaults to False.
        """        
        # TODO: Linear fir for both unwrapped and bias correction as seperate vars
        self.plot_sads(0)
        self.plot_phases()
        self.plot_delta_phases()
        self.plot_delta_phases_phases()
        self.plot_unwrapped_phases(subtract)
        self.plot_bias_correction(subtract)
        self.plot_subframe_histogram_full()


    def __repr__(self) -> str:
        return_string = "Basic optical gating plotter\n"
        return_string += f"\tfigsize: {self.figsize}\n"
        return_string += f"\tBOGs:\n"
        for i, name in enumerate(self.names):
            return_string += f"\t\t{i} : {name}\n"

        return return_string
# !SECTION

# SECTION: Example usage

"""# Create an instance of our optical gating class
og = BasicOpticalGating()
# Load our sequence and reference sequence
og.set_sequence("")
og.set_reference_sequence("")
og.set_reference_period(35.77851226661945)
og.run()
#Plot
ogp = BasicOpticalGatingPlotter([og], ["OG"])
ogp.plot_all()"""
