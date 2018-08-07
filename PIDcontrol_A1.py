# This code implements a PID control algorithm. The user inputs parameters Kp, Ki, and Kd, the setpoint

import matplotlib.pyplot as plt
import numpy as np
import time


class PIDcontrol():
    def __init__(self, kp, ki, kd):
        # Initialize the PID control class
        self.processval = 0.0
        self.controlvar = 0.0
        self.setpoint = 0.0
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error = 0.0
        self.prop_term = 0.0
        self.inte_term = 0.0
        self.deri_term = 0.0
        self.lasterr = 0.0
        self.samplingtime = 0.01  # seconds
        self.currenttime = time.time()
        self.lasttime = self.currenttime
        self.winduplim = 20.0

    def main(self):
        # Setup plots
        timesteps = 200
        timepoints = []
        outputs = []
        setpoints = []
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.show()
        fig.tight_layout()
        fig.canvas.draw()
        axrange = [0, timesteps * self.samplingtime, 0, 2]

        for nn in range(timesteps):
            ax.clear()
            if self.setpoint > 0:
                self.processval = self.processval + (self.controlvar - (1.0 / nn))
            if nn > 9:
                self.setpoint = 1
            time.sleep(self.samplingtime)
            outputs.append(self.processval)
            timepoints.append(nn * self.samplingtime)
            setpoints.append(self.setpoint)
            ax.plot(timepoints, outputs)
            ax.plot(timepoints, setpoints)
            ax.axis(axrange)
            fig.canvas.draw()
            self.update()

    def update(self):
        # Generate output using PID control algorithm
        self.error = self.setpoint - self.processval
        self.currenttime = time.time()
        dt = self.currenttime - self.lasttime
        if dt > self.samplingtime:
            self.prop_term = self.kp * self.error
            self.inte_term = self.inte_term + self.ki * self.error * dt
            if self.inte_term < -self.winduplim:
                self.inte_term = -self.winduplim
            elif self.inte_term > self.winduplim:
                self.inte_term = self.winduplim
            self.deri_term = self.kd * (self.error - self.lasterr) / dt
            self.controlvar = self.prop_term + self.inte_term + self.deri_term
            self.lasterr = self.error
            self.lasttime = self.currenttime


pid = PIDcontrol(1, 0.01, 0.01)
pid.main()
