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

# Logger
from .logger import Logger, Colour

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
    if y_1 < y_2 or y_3 < y_2:
        return 0, 0

    if y_1 > y_3:
        x = 0.5 * (y_1 - y_3) / (y_1 - y_2)
        y = y_2 - x * (y_1 - y_2)
    else:
        x = 0.5 * (y_1 - y_3) / (y_3 - y_2)
        y = y_2 + x * (y_3 - y_2)


    return x, y

def quadratic_v(x, I_1, I_2, x_n, c):
    return np.abs(I_1 * (x - x_n) + 0.5 * I_2 * (x**2 - x_n**2)) + c

def u_fitting(y_1, y_2, y_3):
    # Fit using a symmetric 'U' function, to find the interpolated minimum for three datapoints y_1, y_2, y_3,
    # which are considered to be at coordinates x=-1, x=0 and x=+1
    c = y_2
    a = (y_1 + y_3 - 2*y_2) / 2
    b = y_3 - a - c
    x = -b / (2*a)
    y = a*x**2 + b*x + c
    return x, y, a, b, c

def u_fittingN(y):
    from scipy.optimize import curve_fit
    # Quadratic best fit to N datapoints, which [** inconsistently with respect to u/v_fitting() **]
    # are considered to be at coordinates x=0, 1, ...
    length = y.shape[0]
    x = np.arange(length)
    def quadratic(x, a, b, c):
        return a*x**2 + b*x + c
    (a, b, c), cov = curve_fit(quadratic, x, y, p0=[0, 0, np.average(y)])
    x = -b / (2*a)
    y = quadratic(x, a, b, c)
    #    return x, y, *popt
    return x - length / 2, y, a, b, c

# !SECTION

# SECTION: Basic optical gating
# Simplified version of the optical gating class for testing and implementing new features
class BasicOpticalGating(): 
    # TODO: Add option to scale frame values to a range so different sequences can be compared
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
        self.logger.set_normal()

        # Initialise vars
        self.drift = (0, 0)
        self.reference_period = None
        self.framerate_correction = False
        self.roi = None
        self.frame_history = []
        self.period_history = []
        self.reference_indices = [None, None]
        
        self.has_run = {
            "set_sequence" : False,
            "set_reference_sequence" : False,
            "get_sads" : False,
            "get_phases" : False,
            "get_delta_phases" : False,
            "get_unwrapped_phases" : False,
            "set_region_of_interest" : False
        }
        self.settings = {
            "matching_method" : "jSAD", # jSAD, pSAD, SSD
            "drift_correction" : False,
            "clear_memory" : False,
            "padding" : 2,
            "buffer_length" : 600,
            "skip_frames" : 1,
            "padding_frames" : 2,
            "subframe_method" : "v-fitting",
            "u_fitting_points" : 2,
            "reference_framerate_reduction" : 1
        }

    def __repr__(self): 
        return_string = ""
        return_string += "Basic optical gating class\n"
        return_string += "Run:\n"
        for key, value in self.has_run.items():
            return_string += f"\t{key}:"
            if value:
                return_string += f"{Colour.GREEN} True{Colour.END}\n"
            else:
                return_string += f"{Colour.RED} False{Colour.END}\n"
        return_string += "Settings:\n"
        for key, value in self.settings.items():
            return_string += f"\t{key}:"
            if type(value) is str:
                return_string += f"{Colour.BLUE} {value}{Colour.END}\n"
            elif type(value) is not bool:
                return_string += f"{Colour.BLUE} {value}{Colour.END}\n"
            elif value:
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

        self.logger.print_message("INFO", "Loading sequence")
        
        self.sequence = self._load_data(sequence_src, frames)

        if self.sequence.shape[0] <= 1:
            self.logger.print_message("ERROR", "Sequence must have at least one frame")
            raise Exception("Sequence must have at least one frame")
        
        self.has_run["set_sequence"] = True
        self.logger.print_message("SUCCESS", f"Sequence loaded with {Colour.BLUE}{self.sequence.shape[0]}{Colour.END} frames")

    def set_reference_sequence(self, reference_sequence_src):
        """ Set the reference sequence to use for optical gating

                Args:
                    reference_sequence_src (str): Path to the reference sequence
        """        
        # Load the reference sequence
        self.logger.print_message("INFO", "Loading reference sequence")

        #self.reference_sequence = tf.imread(reference_sequence_src)
        self.reference_sequence = self._load_data(reference_sequence_src)

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
        
    def get_reference_sequence(self):
        self.logger.print_message("INFO", "Generating reference sequence from input sequence")
        for frame in self.sequence[::self.settings["reference_framerate_reduction"]]:
            refget = self.establish_period_from_frames(frame)
            if refget[0] != None:
                break
        self.reference_sequence = np.array(refget[0])
        self.reference_period = refget[1]
        self.reference_indices = refget[2]
        self.has_run["set_reference_sequence"] = True
        
    def establish_period_from_frames(self, pixel_array):
        """ Attempt to establish a period from the frame history,
            including the new frame represented by 'pixel_array'.

            Returns: True/False depending on if we have successfully identified a one-heartbeat reference sequence
        """
        # Add the new frame to our history buffer
        self.frame_history.append(pixel_array)

        # Impose an upper limit on the buffer length, to protect against performance degradation
        # in cases where we are not succeeding in identifying a period.
        # That limit is defined in terms of how many seconds of frame data we have,
        # relative to the minimum heart rate (in Hz) that we are configured to expect.
        # Note that this logic should work when running in real time, and with file_optical_gater
        # in force_framerate=False mode. With force_framerate=True (or indeed in real time) we will
        # have problems if we can't keep up with the framerate frames are arriving at.
        # We should probably be monitoring for that situation...
        ref_buffer_duration = len(self.frame_history)
        while (ref_buffer_duration > self.settings["buffer_length"]):
            # I have coded this as a while loop, but we would normally expect to only trim one frame at a time
            self.logger.print_message("DEBUG", f"Trimming buffer from duration {ref_buffer_duration}s to {self.settings['buffer_length']} frames")
            del self.frame_history[0]
            ref_buffer_duration = len(self.frame_history)

        return self.establish(self.frame_history, self.period_history)
    
    def establish(self, sequence, period_history, require_stable_history=True):
        """ Attempt to establish a reference period from a sequence of recently-received frames.
            Parameters:
                sequence        list of PixelArray objects  Sequence of recently-received frame pixel arrays (in chronological order)
                period_history  list of float               Values of period calculated for previous frames (which we will append to)
                ref_settings    dict                        Parameters controlling the sync algorithms
                require_stable_history  bool                Do we require a stable history of similar periods before we consider accepting this one?
            Returns:
                List of frame pixel arrays that form the reference sequence (or None).
                Exact (noninteger) period for the reference sequence
        """
        start, stop, periodToUse = self.establish_indices(sequence, period_history, require_stable_history)
        if (start is not None) and (stop is not None):
            referenceFrames = sequence[start:stop]
        else:
            referenceFrames = None

        return referenceFrames, periodToUse, [start, stop]

    def establish_indices(self, sequence, period_history, require_stable_history=True):
        """ Establish the list indices representing a reference period, from a given input sequence.
            Parameters: see header comment for establish(), above
            Returns:
                List of indices that form the reference sequence (or None).
        """
        if len(sequence) > 1:
            frame = sequence[-1]
            pastFrames = sequence[:-1]

            # Calculate Diffs between this frame and previous frames in the sequence
            diffs = jps.sad_with_references(frame, pastFrames)

            # Calculate Period based on these Diffs
            period = calculate_period_length(diffs, 5, 0.5, 0.75)
            if period != -1:
                period_history.append(period)

            # If we have a valid period, extract the frame indices associated with this period, and return them
            # The conditions here are empirical ones to protect against glitches where the heuristic
            # period-determination algorithm finds an anomalously short period.
            # JT TODO: The three conditions on the period history seem to be pretty similar/redundant. I wrote these many years ago,
            #  and have just left them as they "ain't broke". They should really be tidied up though.
            #  One thing I can say is that the reason for the *two* tests for >6 have to do with the fact that
            #  we are establishing the period based on looking back from the *most recent* frame, but then actually
            #  and up taking a period from a few frames earlier, since we also need to incorporate some extra padding frames.
            #  That logic could definitely be improved and tidied up - we should probably just
            #  look for a period starting numExtraRefFrames from the end of the sequence...
            # TODO: JT writes: logically these tests should probably be in calculate_period_length, rather than here
            history_stable = (len(period_history) >= (5 + (2 * self.settings["padding_frames"]))
                                and (len(period_history) - 1 - self.settings["padding_frames"]) > 0
                                and (period_history[-1 - self.settings["padding_frames"]]) > 6)
            if (
                period != -1
                and period > 6
                and ((require_stable_history == False) or (history_stable))
            ):
                # We pick out a recent period from period_history.
                # Note that we don't use the very most recent value, because when we pick our reference frames
                # we will pad them with numExtraRefFrames at either end. We pick the period value that
                # pertains to the range of frames that we will actually use
                # for the central "unpadded" range of our reference frames.
                periodToUse = period_history[-1 - self.settings["padding_frames"]]
                self.logger.print_message("SUCCESS", f"Found a period I'm happy with: {Colour.BLUE}{periodToUse}{Colour.END}")

                # DevNote: int(x+1) is the same as np.ceil(x).astype(np.int)
                numRefs = int(periodToUse + 1) + (2 * self.settings["padding_frames"])

                # return start, stop, period
                self.logger.print_message("INFO", f"Start index: {Colour.BLUE}{len(pastFrames) - numRefs}{Colour.END}; Stop index: {Colour.BLUE}{len(pastFrames)}{Colour.END}; Period {Colour.BLUE}{periodToUse}{Colour.END}",)
                return len(pastFrames) - numRefs, len(pastFrames), periodToUse

        self.logger.print_message("DEBUG", "I didn't find a period I'm happy with!")
        return None, None, None

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
            if self.settings["matching_method"] == "jSAD":
                return self.reference_sequence[:,y:y+height, x:x+width]
            else:
                return self.reference_sequence[:,y:y+height, x:x+width].astype(np.int32)
        else:
            if self.settings["matching_method"] == "jSAD":
                return self.reference_sequence
            else:
                return self.reference_sequence.astype(np.int32)
        
    def get_sad(self, frame, reference_sequence):
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
        # NOTE: Drift correction copied from OOG
        if self.settings["drift_correction"]:
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

        # Get our SADs using the specified method
        # TODO: Find a neater way to do this
        # Ideally we'd use switch statements but python doesn't support these
        if self.settings["matching_method"] == "jSAD":
            # Fast SAD using JPS
            sad = jps.sad_with_references(frame_cropped, reference_frames_cropped)
        elif self.settings["matching_method"] == "pSAD":
            # Slower SAD using python
            sad = []
            for i in range(len(reference_frames_cropped)):
                diff = np.abs(np.subtract(frame_cropped, reference_frames_cropped[i], casting = "unsafe"))
                sad.append(np.sum(diff))
            sad = np.array(sad)
        elif self.settings["matching_method"] == "SSD":
            # Sum of square differences
            sad = []
            for i in range(len(reference_frames_cropped)):
                # Divide to avoid overflow
                diff = np.square(np.subtract(frame_cropped, reference_frames_cropped[i], dtype = np.float64) / 10000)
                #diff = np.abs(frame_cropped.astype(np.int64) - reference_frames_cropped[i].astype(np.int64))**2
                sad.append(np.sum(diff))
            sad = np.array(sad)
        elif self.settings["matching_method"] == "SGTD":
            # Sum of greater than differences
            sad = []
            for i in range(len(reference_frames_cropped)):
                diff = np.abs(np.subtract(frame_cropped, reference_frames_cropped[i], casting = "unsafe"))
                diff[diff < 0] = 0
                sad.append(np.sum(np.abs(diff)))
        elif self.settings["matching_method"] == "SLTD":
            # Sum of less than differences
            sad = []
            for i in range(len(reference_frames_cropped)):
                diff = np.abs(np.subtract(frame_cropped, reference_frames_cropped[i], casting = "unsafe"))
                diff[diff > 0] = 0
                sad.append(np.sum(np.abs(diff)))
        else:
            self.logger.print_message("WARNING", "Matching method not recognised, defaulting to SAD")
            sad = jps.sad_with_references(frame_cropped, reference_frames_cropped)


        if self.settings["drift_correction"]:
            dx, dy = self.drift
            self.drift = update_drift_estimate(frame, reference_sequence[np.argmin(sad)], (dx, dy))
            self.drifts.append(self.drift)

        return sad

    def get_sads(self): 
        """ Get the SADs for every frame in our sequence"""       
        if self._check_if_run("set_sequence") == True and self._check_if_run("set_reference_sequence") == True:    
            self.logger.print_message("INFO", f"Calculating similarity metric using method: {Colour.BLUE}{self.settings['matching_method']}{Colour.END}")      
            self.sads = []
            self.drifts = []

            # Calculate SADs
            reference_sequence = self._preprocess_reference_sequence()
            for i in range(self.sequence[::self.settings["skip_frames"]].shape[0]):
                frame = self._preprocess_frame(i * self.settings["skip_frames"])
                self.sads.append(self.get_sad(frame, reference_sequence))

            self.sads = np.array(self.sads)

            self.has_run["get_sads"] = True
            self.logger.print_message("SUCCESS", "SADs calculated")

            return self.sads
        else:
            self.logger.print_message("ERROR", "Cannot calculate SADs without a sequence and reference sequence")
    
    def get_phases(self):
        """ Get the phase estimates for our sequence"""        
        if self._check_if_run("get_sads") == True:
            self.logger.print_message("INFO", "Calculating phases")
            self.phases = []
            self.frame_minimas = []

            # Track frames outside the reference period
            outside_errors = []
            
            # Get the frame estimates
            for i, sad in enumerate(self.sads):           
                # Get the frame estimate
                frame_minima = np.argmin(sad[self.settings["padding_frames"]:-self.settings["padding_frames"]]) + self.settings["padding_frames"]

                if self.settings["subframe_method"] == "u-fitting":
                    y_1 = sad[frame_minima - 1]
                    y_2 = sad[frame_minima]
                    y_3 = sad[frame_minima + 1]
                    subframe_minima = u_fitting(y_1, y_2, y_3)[0]
                    if abs(subframe_minima) > 0.5:
                        self.logger.print_message("WARNING", f"Subframe minima outside range {subframe_minima} for frame {i}")
                elif self.settings["subframe_method"] == "n-u-fitting":
                    subframe_minima = u_fittingN(sad[frame_minima - self.settings["u_fitting_points"]:frame_minima + self.settings["u_fitting_points"] + 1])[0]
                elif self.settings["subframe_method"] == "minima":
                    subframe_minima = 0
                else:
                    # Default to v-fitting
                    y_1 = sad[frame_minima - 1]
                    y_2 = sad[frame_minima]
                    y_3 = sad[frame_minima + 1]
                    subframe_minima = v_fitting_standard(y_1, y_2, y_3)[0]
                    if abs(subframe_minima) > 0.5:
                        self.logger.print_message("WARNING", f"Subframe minima outside range {subframe_minima} for frame {i}")

                self.phases.append(frame_minima - self.settings["padding_frames"] + subframe_minima)

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
            self.logger.print_message("INFO", "Calculating delta phases")
            # If reference period isn't set then use the length of our reference sequence as a guide
            if self.reference_period == None:
                self.logger.print_message("WARNING", "No reference period specified, using reference sequence frames as period.")
                self.reference_period = self.reference_sequence.shape[0] - 4
                self.logger.print_message("INFO", f"Reference period set to {self.reference_period}")
            # Otherwise check values are reasonable
            elif self.reference_period <= 0:
                self.logger.print_message("ERROR", "Reference period must be greater than zero.")
                raise ValueError("Reference period must be greater than.")
            elif (self.reference_period > self.reference_sequence.shape[0] - 4 + 1) or (self.reference_period < self.reference_sequence.shape[0] - self.settings["padding_frames"] * 2 - 1):
                self.logger.print_message("WARNING", f"Reference period significantly different to reference sequence length: {self.reference_sequence.shape[0] - self.settings['padding_frames'] * 2} vs {self.reference_period}.")

            # Initialise the delta frames
            self.delta_phases = []

            # Get our delta frames
            for i in range(1, self.phases.shape[0]):
                delta_frame = ((self.phases[i] - self.phases[i - 1]) % self.reference_period) - self.reference_period
                
                # Correct for wraparound
                while delta_frame < -self.reference_period / 2:
                    delta_frame += self.reference_period

                # Append the delta frame
                self.delta_phases.append(delta_frame)

            self.delta_phases = np.array(self.delta_phases)

            self.has_run["get_delta_phases"] = True
            self.logger.print_message("SUCCESS", "Delta phases calculated")
    
    def get_unwrapped_phases(self):
        """ Get the unwrapped phases """        
        if self._check_if_run("get_delta_phases") == True:
            self.logger.print_message("INFO", "Unwrapping phases")
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
            self.logger.print_message("INFO", "Clearing memory")
            try:
                del self.sequence
                self.has_run["set_sequence"] = False
                gc.collect()
                return True
            except:
                self.logger.print_message("ERROR", "Failed to clear memory.")
                return False
    
    def run(self, clear_memory = False):
        """ Run the full optical gating on our sequence.
            Outputs the phases, delta phases and unwrapped phases.
            Optionally run bias correct and clear memory after completion

            Args:
                bias_correct (bool, optional): Run with bias correction. Defaults to False.
                clear_memory (bool, optional): Delete our sequence from memory. Defaults to False.
        """        
        if not self.has_run["set_reference_sequence"]:
            self.get_reference_sequence()
        self.get_sads()
        self.get_phases()
        self.get_delta_phases()
        self.get_unwrapped_phases()

        if self.settings["clear_memory"]:
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

# SECTION Generate reference sequence
# NOTE: Work in progress
# TODO: Update logger so it works with my code
# TODO: Implement into BOG
#           Save indices of reference sequence
#           Use saved indices to generate reference sequence
#           Use these references for our low framerate generation

logger = Logger("DRS")
logger.set_normal()

def calculate_period_length(diffs, minPeriod=5, lowerThresholdFactor=0.5, upperThresholdFactor=0.75):
    """ Attempt to determine the period of one heartbeat, from the diffs array provided. The period will be measured backwards from the most recent frame in the array
        Parameters:
            diffs    ndarray    Diffs between latest frame and previously-received frames
        Returns:
            Period, or -1 if no period found
    """

    # Calculate the heart period (with sub-frame interpolation) based on a provided list of comparisons between the current frame and previous frames.
    bestMatchPeriod = None

    # Unlike JTs codes, the following currently only supports determining the period for a *one* beat sequence.
    # It therefore also only supports determining a period which ends with the final frame in the diffs sequence.
    if diffs.size < 2:
        logger.print_message("DEBUG", "Not enough diffs, returning -1")
        return -1

    # initialise search parameters for last diff
    score = diffs[diffs.size - 1]
    minScore = score
    maxScore = score
    totalScore = score
    meanScore = score
    minSinceMax = score
    deltaForMinSinceMax = 0
    stage = 1
    numScores = 1
    got = False

    for d in range(minPeriod, diffs.size+1):
        score = diffs[diffs.size - d]
        # got, values = gotScoreForDelta(score, d, values)

        totalScore += score
        numScores += 1

        lowerThresholdScore = minScore + (maxScore - minScore) * lowerThresholdFactor
        upperThresholdScore = minScore + (maxScore - minScore) * upperThresholdFactor

        if score < lowerThresholdScore and stage == 1:
            stage = 2

        if score > upperThresholdScore and stage == 2:
            # TODO: speak to JT about the 'final condition'
            stage = 3
            got = True
            break

        if score > maxScore:
            maxScore = score
            minSinceMax = score
            deltaForMinSinceMax = d
            stage = 1
        elif score != 0 and (minScore == 0 or score < minScore):
            minScore = score

        if score < minSinceMax:
            minSinceMax = score
            deltaForMinSinceMax = d

        # Note this is only updated AFTER we have done the other processing (i.e. the mean score used does NOT include the current delta)
        meanScore = totalScore / numScores

    if got:
        bestMatchPeriod = deltaForMinSinceMax

    if bestMatchPeriod is None:
        logger.print_message("DEBUG","I didn't find a whole period, returning -1")
        return -1

    bestMatchEntry = diffs.size - bestMatchPeriod

    interpolatedMatchEntry = (bestMatchEntry + v_fitting_standard(diffs[bestMatchEntry - 1], diffs[bestMatchEntry], diffs[bestMatchEntry + 1])[0])

    return diffs.size - interpolatedMatchEntry

# !SECTION