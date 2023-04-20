import numpy as np
import tifffile as tf
from perlin_numpy import generate_perlin_noise_2d
import math

from .logger import Logger, Colour

class Drawer():
    def __init__(self, beats, reference_period, dimensions):
        self.beats = beats
        self.reference_period = reference_period
        self.dimensions = dimensions

        # Initialise our arrays
        self.sequence = np.zeros((int(math.ceil(self.beats * self.reference_period)), *self.dimensions), dtype = np.uint8)
        self.reference_sequence = np.zeros((int(math.ceil(self.reference_period) + 4), *self.dimensions), dtype = np.uint8)
        self.canvas = np.zeros(self.dimensions, dtype = np.float32)

        # Set draw mode
        self.draw = np.add

        # Initialise our logger
        self.logger = Logger("DRW")

        self.background = np.abs(generate_perlin_noise_2d((self.dimensions[0], self.dimensions[1]), (1, 1)) * 32)
        self.show_background = True
 
    def clear_canvas(self):
        self.canvas = np.zeros_like(self.canvas)

        if self.show_background == True:
            self.canvas += self.background


    def get_canvas(self):
        self.canvas[self.canvas < 0] = 0
        self.canvas[self.canvas > 255] = 255
        #self.canvas += np.random.normal(0, 4, self.canvas.shape)
        #noise = np.random.poisson(self.canvas / 32)
        #self.canvas += noise
        self.canvas[self.canvas < 0] = 0
        self.canvas[self.canvas > 255] = 255
        return self.canvas.astype(np.uint8)
    
    def draw_to_canvas(self, new_canvas):
        return self.draw(self.canvas, new_canvas)
    
    def set_drawing_method(self, draw_mode):
        self.draw = draw_mode

    # Gaussian
    def circular_gaussian(self, _x, _y, _mean_x, _mean_y, _sdx, _sdy, _theta, _super):
        # Takes an array of x and y coordinates and returns an image array containing a 2d rotated Gaussian
        _xd = (_x - _mean_x)
        _yd = (_y - _mean_y)
        _xdr = _xd * np.cos(_theta) - _yd * np.sin(_theta)
        _ydr = _xd * np.sin(_theta) + _yd * np.cos(_theta)
        return np.exp(-((_xdr**2 / (2 * _sdx**2)) + (_ydr**2 / (2 * _sdy**2)))**_super)

    def draw_circular_gaussian(self, _mean_x, _mean_y, _sdx, _sdy, _theta, _super, _br):
        """
        _summary_

        Args:
            _mean_x (_type_): _description_
            _mean_y (_type_): _description_
            _sdx (_type_): _description_
            _sdy (_type_): _description_
            _theta (_type_): _description_
            _super (_type_): _description_
            _br (_type_): _description_
        """
        # Draw a 2d gaussian
        xx, yy = np.meshgrid(range(self.canvas.shape[0]), range(self.canvas.shape[1]))
        new_canvas = self.circular_gaussian(xx, yy, _mean_x, _mean_y, _sdx, _sdy, _theta, _super)
        new_canvas = _br * new_canvas / np.max(new_canvas)
        self.canvas = self.draw_to_canvas(new_canvas)

    def draw_frame_at_phase(self, phase):
        """
        Draws a frame at a given phase. Subclass this to redefine what our sequence looks like.
        Currently draws two Gaussian blobs with a phase difference of pi/2

        Args:
            phase (float): Phase to draw the frame at
        """        
        phase += 1
        self.clear_canvas()
        self.set_drawing_method(np.add)
        self.draw_circular_gaussian(64, 64, 6 + 12 * (1 + np.sin(phase)), 6 + 12 * (1 + np.sin(phase)), 0, 1, 500)
        self.set_drawing_method(np.subtract)
        self.draw_circular_gaussian(64, 64, 3 + 12 * (1 + np.sin(phase)), 3 + 12 * (1 + np.sin(phase)), 0, 1, 500)
        self.set_drawing_method(np.add)
        self.draw_circular_gaussian(192, 192, 6 + 12 * (1 + np.cos(phase)), 6 + 12 * (1 + np.cos(phase)), 0, 1, 500)
        self.set_drawing_method(np.subtract)
        self.draw_circular_gaussian(192, 192, 3 + 12 * (1 + np.cos(phase)), 3 + 12 * (1 + np.cos(phase)), 0, 1, 500)

    def generate_reference_sequence(self):
        phase_per_frame = 2 * np.pi / self.reference_period
        phase_min = 0
        phase_max = (self.reference_sequence.shape[0]) * phase_per_frame

        phases = np.arange(phase_min, phase_max, phase_per_frame)
        #phases += np.random.normal(0, 0.002, phases.shape[0])
        #phases -= phases[2]
        #phases[13] += 0.01

        for i, phase in enumerate(phases):
            self.draw_frame_at_phase(phase)
            self.reference_sequence[i] = self.get_canvas()

        self.reference_phases = [phase_per_frame]
        self.reference_phases.extend(np.diff(phases))
        self.reference_phases = np.cumsum(self.reference_phases)

    def generate_sequence(self):
        # Generate a sequence of frames
        phase_per_frame = 2 * np.pi / self.reference_period
        phase_min = 0
        phase_max = self.sequence.shape[0] * phase_per_frame


        phases = np.arange(phase_min, phase_max, phase_per_frame)
        """import matplotlib.pyplot as plt
        plt.plot(phases)
        plt.show()
        #phases += np.random.normal(0, 0.001, phases.shape[0])

        phases = np.zeros(phases.shape)
        for i in range(1, phases.shape[0]):
            if i < 500:
                phases[i] = phases[i - 1] + 0.05
            elif i < 750:
                phases[i] = phases[i - 1] + 0.025
            else:
                phases[i] = phases[i - 1] + 0.025 + (i - 750) / 1000
        phases += np.random.normal(0, 0.001, phases.shape[0])
        plt.plot(phases)
        plt.show()"""

        for i, phase in enumerate(phases):
            self.draw_frame_at_phase(phase)
            self.sequence[i] = self.get_canvas()