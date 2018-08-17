import numpy as np
import time
from bokeh.layouts import row, widgetbox
from bokeh.models.widgets import Slider, Select, Div
from bokeh.plotting import figure
from bokeh.server.server import Server

timesteps = 100
samplingtime = 0.2
maxsetpoint = 1.0
timepoints = []
outputs = []
setpoints = []
errors = []

timepoints = np.zeros(timesteps)
outputs = np.zeros(timesteps)
setpoints = np.zeros(timesteps)
errors = np.zeros(timesteps)

class PID:
    def __init__(self, P, I, D):
        self.kp = P
        self.ki = I
        self.kd = D

    def initialize(self):
        # Initialize
        global timepoints
        timepoints.fill(np.nan)
        global outputs
        outputs.fill(np.nan)
        global setpoints
        setpoints.fill(np.nan)
        global errors
        errors.fill(np.nan)
        self.processval = 0.0
        self.controlvar = 0.0
        self.setpoint = 0.0
        self.error = 0.0
        self.prop_term = 0.0
        self.inte_term = 0.0
        self.deri_term = 0.0
        self.lasterr = 0.0
        self.currenttime = time.time()
        self.lasttime = self.currenttime

    def controlLoop(self):
        # PID control logic
        self.error = self.setpoint - self.processval
        self.currenttime = time.time()
        dt = self.currenttime - self.lasttime
        if dt > samplingtime:
            self.prop_term = self.kp * self.error
            inte_term = self.inte_term + self.ki * self.error * dt
            self.deri_term = self.kd * (self.error - self.lasterr) / dt
            self.controlvar = self.prop_term + self.inte_term + self.deri_term
            self.lasterr = self.error
            self.lasttime = self.currenttime
        time.sleep(samplingtime)

def modify_doc(doc):    

    # Setup plots
    p1 = figure(plot_width=400, plot_height=300, x_axis_label='time',
                x_range=[0, timesteps * samplingtime], title='PID controller input and output')
    p2 = figure(plot_width=400, plot_height=300, x_axis_label='time',
                x_range=[0, timesteps * samplingtime], title='PID controller error')
    r1 = p1.line(timepoints, setpoints, line_color='cornflowerblue', legend='input',
                 line_width=2)
    r2 = p1.line(timepoints, outputs, line_color='indianred', legend='output',
                 line_width=2)
    r3 = p2.line(timepoints, errors, line_color='indigo', legend='error',
                 line_width=2)
    p1.legend.location = 'top_left'
    p2.legend.location = 'top_left'

    # Setup widgets
    Kp = Slider(title='Kp', value=0.75, start=0.0, end=1.5, step=0.15)
    Ki = Slider(title='Ki', value=0.5, start=0.0, end=1.0, step=0.1)
    Kd = Slider(title='Kd', value=0.05, start=0.0, end=0.1, step=0.01)
    InputFunction = Select(title='Input Function', value='step', options=['step', 'square', 'sine'])
    TextDisp = Div(text='''<b>Note:</b> Wait for the plots to stop updating before changing inputs.''')

    def runPID(attrname, old, new):
        # Get current widget values
        kp = Kp.value
        ki = Ki.value
        kd = Kd.value
        inputfunc = InputFunction.value
        # Create PID class object
        pid = PID(kp, ki, kd)
        pid.initialize()
        for nn in range(1, timesteps, 1):
            # Inputs
            if inputfunc == 'step':
                # step function
                if nn > 30:
                    pid.setpoint = maxsetpoint
            elif inputfunc == 'square':
                # square wave
                if nn > 30:
                    pid.setpoint = maxsetpoint * np.sign(np.sin(4 * np.pi * (nn - 30) / timesteps))
            elif inputfunc == 'sine':
                # sine wave
                if nn > 30:
                    pid.setpoint = maxsetpoint * np.sin(4 * np.pi * (nn - 30) / timesteps)
            # Outputs    
            if nn > 30:
                pid.processval = pid.processval + (pid.controlvar - (1.0 / nn))
            pid.controlLoop()
            global outputs
            outputs[nn] = pid.processval
            global timepoints
            timepoints[nn] = nn * samplingtime
            global setpoints
            setpoints[nn] = pid.setpoint
            global errors
            errors[nn] = pid.error
            # Update plots
            r1.data_source.data['x'] = timepoints
            r1.data_source.data['y'] = setpoints
            r2.data_source.data['x'] = timepoints
            r2.data_source.data['y'] = outputs
            r3.data_source.data['x'] = timepoints
            r3.data_source.data['y'] = errors
    
    # Setup callbacks
    for d in [Kp, Ki, Kd, InputFunction]:
        d.on_change('value', runPID)
    
    # Setup layout and add to document
    wInputs = widgetbox(InputFunction, Kp, Ki, Kd, TextDisp)
    
    doc.add_root(row(wInputs, p1, p2, width=1000))
 
server = Server({'/': modify_doc}, num_procs=1)
server.start()

if __name__=='__main__':
    print('Opening Bokeh application on http://localhost:5006/')
    server.io_loop.add_callback(server.show, '/')
    server.io_loop.start()
