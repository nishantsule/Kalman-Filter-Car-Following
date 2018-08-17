import numpy as np
import time
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from bokeh.plotting import figure 
from bokeh.layouts import row, column
from bokeh.models.widgets import Button
from bokeh.server.server import Server

# Initial values
ProcNoise = 0.4
MeasNoise = 0.8
dist_size = 100
car_size = 2
car1_initpos = 10
car2_initpos = 6
car1_vel = 4
car2_initvel = 4
car1_acc = 0
car2_initacc = 0
car2_maxacc = 2
car2_maxbrake = 40
dt = 0.01
dist_sep = 2 * car_size
arr_size = 50

# Initializing arrays for plotting
timepoints = np.zeros(arr_size)
timepoints.fill(np.nan)
poscar1 = np.zeros(arr_size)
poscar1.fill(np.nan)
poscar1_est = np.zeros(arr_size)
poscar1_est.fill(np.nan)
poscar1_meas = np.zeros(arr_size)
poscar1_meas.fill(np.nan)
poscar2 = np.zeros([arr_size, 2])
poscar2.fill(np.nan)
car_sep = np.zeros([arr_size, 2])
car_sep.fill(np.nan)
car2_acc = np.zeros([arr_size, 2])
car2_acc.fill(np.nan)

def modify_doc(doc):
# Setup plots

    # Figure1
    p1 = figure(plot_width=300, plot_height=400, x_axis_label='time', y_axis_label='position',
                title='Car positions')
    r1 = p1.circle(timepoints, poscar1, color='teal', legend='car1',
                line_width=2)
    r2 = p1.circle(timepoints, poscar1_meas, color='olivedrab', legend='car1 meas.',
                line_width=2)
    r3 = p1.circle(timepoints, poscar1_est, color='greenyellow', legend='car1 est.',
                line_width=2)
    r4 = p1.circle(timepoints, poscar2[:, 0], color='gold', legend='car2 no kf',
                line_width=2)
    r5 = p1.circle(timepoints, poscar2[:, 1], color='darkslategray', legend='car2 with kf',
                line_width=2)
    p1.legend.location = 'center_right'
    p1.toolbar.logo = None
    p1.toolbar_location = None
    
    # Figure 2
    p2 = figure(plot_width=300, plot_height=400, x_axis_label='time', y_axis_label='separation',
                title='Car separations', y_range=[0, 4 * dist_sep])
    r6 = p2.circle(timepoints, car_sep[:, 0], color='gold', line_width=2, legend='no KF')
    r7 = p2.circle(timepoints, car_sep[:, 1], color='darkslategray', line_width=2, legend='with KF')
    p2.legend.location = 'center_right'
    p2.toolbar.logo = None
    p2.toolbar_location = None
    
    # Figure 3
    p3 = figure(plot_width=300, plot_height=400, x_axis_label='time', y_axis_label='acceleration',
                title='Car2 acceleration')
    r8 = p3.circle(timepoints, car2_acc[:, 0], color='gold', line_width=2, legend='no KF')
    r9 = p3.circle(timepoints, car2_acc[:, 1], color='darkslategray', line_width=2, legend='with KF')
    p3.legend.location = 'bottom_center'
    p3.toolbar.logo = None
    p3.toolbar_location = None
    
    # Figure 4
    p4 = figure(plot_width=900, plot_height=200, x_axis_label='position', title='Car Following',
                x_range=[0, dist_size])
    colors = ['teal', 'gold', 'darkslategray']
    r10 = p4.rect(x=[car1_initpos, car2_initpos, car2_initpos], y=[0, 0, 0], 
                  width=2, height=10, color=colors, alpha=0.6,
                  width_units='data', height_units='screen')
    p4.yaxis.visible = False
    p4.ygrid.visible = False
    p4.toolbar.logo = None
    p4.toolbar_location = None

    # Setup widgets
    StartButton = Button(label='Start', width=10)

    def RunCarFollowing(attr, old, new):        
        # Setup initial values
        pos1real = car1_initpos
        pos2 = car2_initpos
        pos2_kf = car2_initpos
        vel2 = car2_initvel
        vel2_kf = car2_initvel
        acc2 = car2_initacc
        acc2_kf = car2_initacc
        tt = 0
        realt = 0
        
        # Initialize Kalman Filter
        f = KalmanFilter(dim_x=2, dim_z=1)
        f.x = np.array([[car1_initpos],  # position
                        [car1_vel]])  # velocity
        f.F = np.array([[1.0, 1.0],
                        [0.0, 1.0]])
        f.H = np.array([[1.0, 0.0]])
        f.P = np.array([[1000.0, 0.0],
                        [0.0, 1000.0]])
        f.R = np.array([[5.0]])
        f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=ProcNoise)  # Process noise
        
        # Loop for updating positions
        while True:
            tt = np.mod(tt, arr_size - 1)
    
            # Real position of leading car
            pos1real = pos1real + car1_vel * dt
    
            # Applying periodic boundary 
            pos1real = np.mod(pos1real, dist_size)
            pos2 = np.mod(pos2, dist_size)
            pos2_kf = np.mod(pos2_kf, dist_size)
    
            # Measurement of position of leading car with noise
            z = np.random.normal(pos1real, MeasNoise)
    
            # Kalman Filter predict and update steps
            f.predict()
            f.update(z)   
    
            # Following car update without Kalman Filter
            dx = z - pos2
            # Applying minimum image criteria for periodic boundary
            while dx > dist_size / 2: dx = dx - dist_size
            while dx < -dist_size / 2: dx = dx + dist_size
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
            # Applying minimum image criteria for periodic boundary
            while dx_kf > dist_size / 2: dx_kf = dx_kf - dist_size
            while dx_kf < -dist_size / 2: dx_kf = dx_kf + dist_size
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
    
            # update counter
            tt = tt + 1
            realt = realt + 1
    
            # Update arrays for plotting
            global timepoints, poscar1, poscar2, poscar1_est, car_sep
            timepoints[tt] = realt * dt
            poscar1[tt] = pos1real
            poscar1_est[tt] = f.x[0, 0]
            poscar1_meas[tt] = z
            poscar2[tt, 0] = pos2
            poscar2[tt, 1] = pos2_kf
            car_sep[tt, 0] = pos1real - pos2
            car_sep[tt, 1] = pos1real - pos2_kf  
            car2_acc[tt, 0] = acc2
            car2_acc[tt, 1] = acc2_kf
    
            # Update plots
            #first row
            r10.data_source.data['x'] = [pos1real, pos2, pos2_kf]
            #second row
            r1.data_source.data['x'] = timepoints
            r1.data_source.data['y'] = poscar1
            r2.data_source.data['x'] = timepoints
            r2.data_source.data['y'] = poscar1_meas
            r3.data_source.data['x'] = timepoints
            r3.data_source.data['y'] = poscar1_est
            r4.data_source.data['x'] = timepoints
            r4.data_source.data['y'] = poscar2[:, 0]
            r5.data_source.data['x'] = timepoints
            r5.data_source.data['y'] = poscar2[:, 1]
            r6.data_source.data['x'] = timepoints
            r6.data_source.data['y'] = car_sep[:, 0]
            r7.data_source.data['x'] = timepoints
            r7.data_source.data['y'] = car_sep[:, 1]
            r8.data_source.data['x'] = timepoints
            r8.data_source.data['y'] = car2_acc[:, 0]
            r9.data_source.data['x'] = timepoints
            r9.data_source.data['y'] = car2_acc[:, 1]

            time.sleep(0.1)

    StartButton.on_click(RunCarFollowing)
    doc.add_root(column(StartButton, p4))
    doc.add_root(row(p1, p2, p3))

server = Server({'/': modify_doc}, num_procs=1)
server.start()
 
if __name__=='__main__':
    print('Opening Bokeh application on http://localhost:5006/')
    server.io_loop.add_callback(server.show, '/')
    server.io_loop.start()

