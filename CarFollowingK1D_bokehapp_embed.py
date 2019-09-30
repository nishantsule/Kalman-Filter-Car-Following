import numpy as np
import time
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from bokeh.plotting import figure 
from bokeh.layouts import row, column, widgetbox
from bokeh.models.widgets import Slider, Div, Button
from bokeh.events import ButtonClick
from bokeh.server.server import Server
from bokeh.models import Label

# Initial values
dist_size = 30
car_size = 1
car1_initpos = 3
car2_initpos = 0
car1_vel = 4
car2_initvel = 4
car1_acc = 0
car2_initacc = 0
car2_maxacc = 2
car2_maxbrake = 40
dt = 0.015
timesteps = int((dist_size - car1_initpos) / car1_vel / dt)
dist_sep = 3 * car_size

timepoints = np.zeros(timesteps)
poscar1 = np.zeros(timesteps)
poscar1_est = np.zeros(timesteps)
poscar1_meas = np.zeros(timesteps)
poscar2 = np.zeros(timesteps)
car_sep = np.zeros(timesteps)
car2_acc = np.zeros(timesteps)
poscar2_kf = np.zeros(timesteps)
car_sep_kf = np.zeros(timesteps)
car2_acc_kf = np.zeros(timesteps)
noiserange = np.zeros(timesteps)
pdf_proc = np.zeros(timesteps)
pdf_meas = np.zeros(timesteps)


def modify_doc(doc):

    # Initializing arrays for plotting
    def InitPlotArrays():    
        timepoints.fill(np.nan)
        poscar1.fill(np.nan)
        poscar1_est.fill(np.nan)
        poscar1_meas.fill(np.nan)
        poscar2.fill(np.nan)
        car_sep.fill(np.nan)
        car2_acc.fill(np.nan)
        poscar2_kf.fill(np.nan)
        car_sep_kf.fill(np.nan)
        car2_acc_kf.fill(np.nan)
        noiserange.fill(np.nan)
        pdf_proc.fill(np.nan)
        pdf_meas.fill(np.nan)

    # Setup plots
    InitPlotArrays()
    
    # Adding legends
    p1 = figure(plot_width = 300, plot_height = 120, x_range=[-0.5, 20], y_range=[-1, 5])
    colors = ['darkslategray', 'gold', 'crimson']
    p1.rect([0.2, 0.2, 0.2], [4, 2, 0], width=1, height=1, color=colors, 
                  width_units='data', height_units='data')
    label1 = Label(x=1, y=3.3, x_units='data', y_units='data', text='Leading Car')
    label2 = Label(x=1, y=1.3, x_units='data', y_units='data', text='Following Car')
    label3 = Label(x=1, y=-0.7, x_units='data', y_units='data', text='Following Car with Kalman Filter')
    p1.add_layout(label1)
    p1.add_layout(label2)
    p1.add_layout(label3)
    p1.yaxis.visible = False
    p1.ygrid.visible = False
    p1.xaxis.visible = False
    p1.xgrid.visible = False
    p1.toolbar.logo = None
    p1.toolbar_location = None

    # Figure 1
    p2 = figure(plot_width = 400, plot_height = 200, title='Noise Distributions')
    r21 = p2.line(noiserange, pdf_proc, line_color='seagreen', line_width=4, legend='Process')
    r22 = p2.line(noiserange, pdf_meas, line_color='greenyellow', line_width=4, legend='Measurement')
    p2.toolbar.logo = None
    p2.toolbar_location = None
    p2.legend.location = 'top_left'

    # Figure 2
    p3 = figure(plot_width=1000, plot_height=150, x_axis_label='position', title='Car Following Animation',
                x_range=[0, dist_size], y_range=[-0.4, 0.4])
    colors = ['darkslategray', 'gold', 'crimson']
    r31 = p3.rect(x=[car1_initpos, car2_initpos, car2_initpos], y=[0.0, -0.2, 0.2], 
                  width=car_size, height=8, color=colors, 
                  width_units='data', height_units='screen')
    r32 = p3.line(x=[car1_initpos - car_size / 2, car1_initpos - car_size / 2],
                  y=[-0.4, 0.4])
    r33 = p3.line(x=[car1_initpos + car_size / 2 - dist_sep, car1_initpos + car_size / 2 - dist_sep], 
                  y=[-0.4, 0.4])
    p3.yaxis.visible = False
    p3.ygrid.visible = False
    p3.toolbar.logo = None
    p3.toolbar_location = None

    # Figure 3
    p4 = figure(plot_width=500, plot_height=400, x_axis_label='time', y_axis_label='position',
                title='Car Positions', x_range=[0, timesteps * dt], y_range=[0, dist_size])
    r41 = p4.line(timepoints, poscar1, color='darkslategray', line_width=4)
    r42 = p4.line(timepoints, poscar1_meas, color='royalblue', 
                 legend='Measured position', line_width=4)
    r43 = p4.line(timepoints, poscar1_est, color='skyblue', 
                 legend='KF estimated position', line_width=4)
    r44 = p4.line(timepoints, poscar2, color='gold', line_width=4)
    r45 = p4.line(timepoints, poscar2_kf, color='crimson', line_width=4)
    p4.legend.location = 'bottom_right'
    p4.toolbar.logo = None
    p4.toolbar_location = None

    # Figure 4
    p5 = figure(plot_width=500, plot_height=200, x_axis_label='time', y_axis_label='separation',
                title='Car Separations', x_range=[0, timesteps * dt], y_range=[0, 3 * dist_sep])
    r51 = p5.line(timepoints, car_sep, color='gold', line_width=4)
    r52 = p5.line(timepoints, car_sep_kf, color='crimson', line_width=4)
    p5.toolbar.logo = None
    p5.toolbar_location = None
    
    # Figure 5
    p6 = figure(plot_width=500, plot_height=200, x_axis_label='time', y_axis_label='acceleration',
                title='Following Car Accelerations', x_range=[0, timesteps * dt], 
                y_range=[-car2_maxbrake / 2, car2_maxacc])
    r61 = p6.line(timepoints, car2_acc, color='gold', line_width=4)
    r62 = p6.line(timepoints, car2_acc_kf, color='crimson', line_width=4)
    p6.toolbar.logo = None
    p6.toolbar_location = None

    # Setup widgets
    procnoise_slider = Slider(title='Process Noise', value=0.5, start=0.1, end=1.5, step=0.1)
    measnoise_slider = Slider(title='Measurement Noise', value=0.5, start=0.1, end=1.5, step=0.1)
    TextDisp = Div(text='''<b>Note:</b> Wait for the plots to stop updating before hitting Start.''')
    TextDesc = Div(text='''This simulation shows how Kalman Filtering can be used to improve autonomous car following. 
                   The red car uses a Kalman filter to filter out noise inherent in the detection of the car in the front.
                   You can increase or decrease the noise level by dragging the sliders.
                   The yellow car uses the raw sensor data. Both cars aim to achieve a fixed separation between itself and
                   (vertical blue line in the Car Following Animation) the leading car. 
                   The plots below show car positions, separations, and accelarations.''', width=1000)
    textrel = Div(text='''Learn more about this app works, Kalman filters, and their applications in <b>ES/AM 115</b> ''', width=1000)
    TextTitle = Div(text='''<b>KALMAN FILTER FOR AUTONOMOUS CAR FOLLOWING</b>''', width=1000)
    StartButton = Button(label='Start', button_type="success")

    #def RunCarFollowing(attr, old, new):        
    def RunCarFollowing(event):        
        # Get current widget values
        PN = procnoise_slider.value
        MN = measnoise_slider.value
        # Setup initial values
        InitPlotArrays()
        pos1real = car1_initpos
        pos2 = car2_initpos
        pos2_kf = car2_initpos
        vel2 = car2_initvel
        vel2_kf = car2_initvel
        acc2 = car2_initacc
        acc2_kf = car2_initacc
        tt = 0
        realt = 0

       # Update noise distributions
        nlim = max(PN, MN)
        global noiserange, pdf_proc, pdf_meas
        noiserange = np.linspace(-4 * nlim, 4 * nlim, timesteps)
        pdf_proc = 1 / (PN * np.sqrt(2 * np.pi)) * np.exp(-noiserange**2 / (2 * PN**2))
        r21.data_source.data['x'] = noiserange
        r21.data_source.data['y'] = pdf_proc
        pdf_meas = 1 / (MN * np.sqrt(2 * np.pi)) * np.exp(-noiserange**2 / (2 * MN**2))
        r22.data_source.data['x'] = noiserange
        r22.data_source.data['y'] = pdf_meas
        
        # Initialize Kalman Filter
        f = KalmanFilter(dim_x=2, dim_z=1)  # initialize Kalman filter object
        f.x = np.array([[car1_initpos],  # position
                        [car1_vel]])  # velocity
        f.F = np.array([[1.0, 1.0],  # state transition matrix
                        [0.0, 1.0]])
        f.H = np.array([[1.0, 0.0]])  # measurement function
        f.P = np.array([[1000.0, 0.0],  # aprioori state covariance
                        [0.0, 1000.0]])
        f.R = np.array([[5.0]])  # measurement noise covariance
        f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=PN**2)  # Process noise
        
        # Loop for updating positions
        for tt in range(timesteps):        
    
            # Real position of leading car
            pos1real = pos1real + car1_vel * dt
    
            # Measurement of position of leading car with noise
            z = np.random.normal(pos1real, MN)
    
            # Kalman Filter predict and update steps
            f.predict()  # predict next state using Kalman filter state propagation equation
            f.update(z)  # add new measurement (z) to the Kalman filter 
   
            # Following car update without Kalman Filter
            dx = z - pos2
            err = (dx - dist_sep) / dist_sep
            if err > 1.0: err = 1.0
            # Acceleration model
            if err > 0:
                acc2 = err * car2_maxacc
            else:
                acc2 = err * car2_maxbrake
            vel2 = vel2 + acc2 * dt
            # crash protection 
            if vel2 < 0:
                vel2 = 0
                acc2 = 0
            # update position
            pos2 = pos2 + vel2 * dt + 0.5 * acc2 * dt**2
    
            # Following car update with Kalman Filter
            dx_kf = f.x[0, 0] - pos2_kf
            err_kf = (dx_kf - dist_sep) / dist_sep
            if err_kf > 1.0: err_kf = 1.0
            # Acceleration model
            if err_kf > 0:
                acc2_kf = err_kf * car2_maxacc
            else:
                acc2_kf = err_kf * car2_maxbrake
            vel2_kf = vel2_kf + acc2_kf * dt
            # crash protection
            if vel2_kf < 0:
                vel2_kf = 0
                acc2_kf = 0
            # update position
            pos2_kf = pos2_kf + vel2_kf * dt + 0.5 * acc2_kf * dt**2

            # Update arrays for plotting
            global timepoints, poscar1, poscar2, poscar1_est, car_sep
            timepoints[tt] = tt * dt
            poscar1[tt] = pos1real
            poscar1_est[tt] = f.x[0, 0]
            poscar1_meas[tt] = z
            poscar2[tt] = pos2
            poscar2_kf[tt] = pos2_kf
            car_sep[tt] = pos1real - pos2
            car_sep_kf[tt] = pos1real - pos2_kf  
            car2_acc[tt] = acc2
            car2_acc_kf[tt] = acc2_kf

            # Update plots
            #first row
            r31.data_source.data['x'] = [pos1real, pos2, pos2_kf]
            r32.data_source.data['x'] = [pos1real - car_size / 2, pos1real - car_size / 2]
            r33.data_source.data['x'] = [pos1real + car_size / 2 - dist_sep, pos1real + car_size / 2 - dist_sep]
            #second row
            r41.data_source.data['x'] = timepoints
            r41.data_source.data['y'] = poscar1
            r42.data_source.data['x'] = timepoints
            r42.data_source.data['y'] = poscar1_meas
            r43.data_source.data['x'] = timepoints
            r43.data_source.data['y'] = poscar1_est
            r44.data_source.data['x'] = timepoints
            r44.data_source.data['y'] = poscar2
            r45.data_source.data['x'] = timepoints
            r45.data_source.data['y'] = poscar2_kf
            r51.data_source.data['x'] = timepoints
            r51.data_source.data['y'] = car_sep
            r52.data_source.data['x'] = timepoints
            r52.data_source.data['y'] = car_sep_kf
            r61.data_source.data['x'] = timepoints
            r61.data_source.data['y'] = car2_acc
            r62.data_source.data['x'] = timepoints
            r62.data_source.data['y'] = car2_acc_kf
    
            time.sleep(0.05)

    # Setup callbacks
    StartButton.on_event(ButtonClick, RunCarFollowing)
    # Setup layout and add to document
    wInputs = widgetbox(procnoise_slider, measnoise_slider, TextDisp, StartButton)    

    doc.add_root(column(TextTitle, TextDesc, row(p1, p2, wInputs), p3, row(p4, column(p5, p6)), textrel))

server = Server({'/': modify_doc}, num_procs=1)
server.start()
 
if __name__=='__main__':
    print('Opening Bokeh application on http://localhost:5006/')
    server.io_loop.add_callback(server.show, '/')
    server.io_loop.start()

