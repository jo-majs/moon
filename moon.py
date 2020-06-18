# -*- coding: utf-8 -*-

"""The program simulates the movement of Sun, Earth, Moon and n=BODIES planetoids
   It plots:
   the initial positions of all bodies
   the initial positions of n=5 first bodies (S, E, M and 2 planetoids)
   the trajectories of n=5 first bodies
   It detects collisions when a platetoid is closer than DIST_COLLISION to Moon
   It then plots
   a histogram of the angles of collisions
   """

import time
import math
import multiprocessing as mp
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
#CONSTANTS
#------------------------------------------------------------------------------
M_SUN = 1.989 * 10**30#kg
M_EARTH = 5.972 * 10**24#kg
M_MOON = 7.342 * 10**22#kg
M_PLANETOID = 1.#kg

D_SUN_EARTH = 1.49597870 * 10**11#m = 1 AU
D_EARTH_MOON = 3.84400 * 10**8#m
D_SUN_MARS = 1.524 * D_SUN_EARTH#m
R_MOON = 1.7381 * 10**6#m

G = 6.67 * 10 ** (-11)#m^3/kg*s gravitational constant

#------------------------------------------------------------------------------
#SIMULATION PARAMETERS
#------------------------------------------------------------------------------
BODIES = 100#number of bodies, minimum : 3 - sun, earth, moon
DIST_COLLISION = 50.1 * D_EARTH_MOON#when distance between any planetoid 
#and moon smaller than this, a collision detected
ANGLE_RANGE = 0.4 * 2 * math.pi#planetoids get random positions within it
V_ANGLE_RANGE = 0.4 * 2 * math.pi#planetoids get random velocities within it
V_INIT_FRAC = 0.5#initial velocity of planetoids as fraction of v of Mars
SIMULATION_DURATION_YEARS = 1#earth years
TIME_STEP = 3600#earth seconds
SIMULATION_DURATION_SECONDS = int(SIMULATION_DURATION_YEARS * 365 * 24 * 3600)
NUMBER_TIME_STEPS = int(SIMULATION_DURATION_SECONDS / TIME_STEP)

v_cosmic1_sun_earth = np.sqrt(G * M_SUN / D_SUN_EARTH)
v_cosmic1_earth = np.sqrt(G * M_EARTH / D_EARTH_MOON)
v_cosmic1_sun_mars = np.sqrt(G * M_SUN / D_SUN_MARS)
v_angular_earth_on_mars = v_cosmic1_sun_earth  * D_SUN_MARS / D_SUN_EARTH
V_RANGE = 0.5 * v_cosmic1_sun_mars#planetoids' random velocities within v_range
N = mp.cpu_count()

#------------------------------------------------------------------------------
#FUNCTIONS
#------------------------------------------------------------------------------

#measuring time of functions' execution
times_sum = {}
calls = []

def measure_time(func):
    """ time measuring decorator """
    times_sum[func.__name__] = 0.
    def new_func(*args, **kwargs):
        t1 = time.time()
        calls.append(func.__name__)
        result = func(*args, **kwargs)
        calls.pop()
        t2 = time.time()
        times_sum[func.__name__] += (t2-t1)
        if calls:
            times_sum[calls[-1]] -= (t2-t1)
        return result
    return new_func

@measure_time
def subtract_vector(v_row):
    """ takes a row vector v with dim=n
    returns an n x n matrix with differences of every two elements of v"""
    v_col = v_row[:, np.newaxis]
    return v_row - v_col

@measure_time
def equation(state_vector, t, masses):
    """ differential equation of motion
    for n=BODIES bodies acting on each other with gravitational force"""
    x = state_vector[::4]
    y = state_vector[1::4]
    vx = state_vector[2::4]
    vy = state_vector[3::4]
    
    r_x = subtract_vector(x)
    r_y = subtract_vector(y)
    r = np.array([r_x, r_y])
    norm_r = np.linalg.norm(r, axis=0)
    norm_r_positive = np.where(norm_r==0, 1, norm_r)#zeros are replaced
    #by ones to avoid dividing by zero

    #2 x n x n matrix of x and y accelerations between any two bodies
    GBA = np.nan_to_num(G * masses * r / norm_r_positive**3)
    #2 x n vector of x and y accelerations working on each body
    a = np.sum(GBA, axis=2) #[x or y][acceleration of i-th body]

    solutions = np.empty((4 * BODIES), dtype=float)
    solutions[::4] = vx
    solutions[1::4] = vy
    solutions[2::4] = a[0]
    solutions[3::4] = a[1]
    return solutions

@measure_time
def find_collisions(all_collisions, angles, solution):
    """takes two empty numpy arrays and one array with diff.eq. solutions
    returns array of indices of bodies which collided with moon
    and array of angles of collision"""
    # print("indices of bodies closer to the moon than DIST_COLLISION")
    for i in range(0, NUMBER_TIME_STEPS-1):
        xs = solution[i][12::4]
        ys = solution[i][13::4]
        moon_x = solution[i][8]
        moon_y = solution[i][9]
        d_xs = xs - moon_x
        d_ys = ys - moon_y
        rs = np.sqrt(d_xs**2 + d_ys**2)
        new_collisions = np.nonzero(rs <= DIST_COLLISION)[0] + 3

        new_collisions = np.setdiff1d(new_collisions, all_collisions)
        if new_collisions.size:
            # print("indices of colliding bodies", new_collisions)
            all_collisions = np.append(all_collisions, new_collisions)
            for j in new_collisions:
                # j = int(j)
                # print("index of colliding body", j)
                earth = np.array([solution[i][4], solution[i][5]])
                planetoid = np.array([solution[i][j*4], solution[i][j*4+1]])
                moon = np.array([moon_x, moon_y])

                moon_ea = earth - moon
                moon_pl = planetoid - moon

                num = np.dot(moon_ea, moon_pl)
                denom = np.linalg.norm(moon_ea) * np.linalg.norm(moon_pl)
                angle = np.arccos(num/denom)#radians
                angles = np.append(angles, angle)
    return all_collisions, angles

@measure_time
def plot_angles_distribution(angles, n_bins=16):
    """ takes a list of collisions' angles
    plots a histogram of collisions' angles, 
    where 0 is the angle from Earth to Moon"""

    bin_size = 2 * math.pi / n_bins
    a, b = np.histogram(angles, bins=np.arange(0, 2*math.pi+bin_size, bin_size))
    centers = (b//2)[:-1] + b[:-1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.bar(centers, a, width=bin_size, bottom=0.0)
    ax.set_title("angular distribution of collisions")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    plt.show()

@measure_time
def plot_trajectories(n=5, n_days=15):
    """ plots trajectories of n first bodies during SIMULATION_DURATION 
    each new color shows a state after n_days days"""
    plt.plot(l_solution[0][0:n*4:4], l_solution[0][1:n*4:4], 'o')
    plt.xlim([-3 * 10**11, 3 * 10**11])
    plt.ylim([-3 * 10**11, 3 * 10**11])
    plt.title("Initial positions of n first bodies")
    plt.show()
    for i in range(0, NUMBER_TIME_STEPS-1, 24*n_days):
        plt.plot(l_solution[i][0:n*4:4], l_solution[i][1:n*4:4], 'o')
        plt.xlim([-3 * 10**11, 3 * 10**11])
        plt.ylim([-3 * 10**11, 3 * 10**11])
    plt.title("Trajectories of n first bodies")
    plt.show()
    
#------------------------------------------------------------------------------
#INITIAL CONDITIONS
#------------------------------------------------------------------------------

#times at which solutions to differential equation will be calculated
times = np.linspace(0, SIMULATION_DURATION_SECONDS, NUMBER_TIME_STEPS)

#masses vector
m_row = np.ones(BODIES) * M_PLANETOID
m_row[0] = M_SUN
m_row[1] = M_EARTH
m_row[2] = M_MOON

#random planetoid positions
angles_planetoids = np.random.rand(BODIES - 3) * ANGLE_RANGE - ANGLE_RANGE / 2
x_planetoids = D_SUN_MARS * np.cos(angles_planetoids)
y_planetoids = D_SUN_MARS * np.sin(angles_planetoids)

# #planetoid velocities - angular velocity equal to Earth's angular velocity
# vx_planetoids = v_angular_earth_on_mars * np.sin(-angles_planetoids)
# vy_planetoids = v_angular_earth_on_mars * np.cos(angles_planetoids)
#planetoid velocities - velocity equal to a fraction of velocity of Mars
vx_planetoids = V_INIT_FRAC * v_cosmic1_sun_mars * np.sin(-angles_planetoids)
vy_planetoids = V_INIT_FRAC * v_cosmic1_sun_mars * np.cos(angles_planetoids)

#random planetoid velocities
v_rand_pl = np.random.rand(BODIES - 3) * V_RANGE
v_angle_rand_pl = np.random.rand(BODIES - 3) * V_ANGLE_RANGE - V_ANGLE_RANGE / 2
v_x_rand_pl = v_rand_pl * -np.cos(angles_planetoids + v_angle_rand_pl)
v_y_rand_pl = v_rand_pl * -np.sin(angles_planetoids + v_angle_rand_pl)
vx_planetoids += v_x_rand_pl
vy_planetoids += v_y_rand_pl

#initial state vector
state_vector0 = np.zeros(BODIES * 4)
state_vector0[4] = D_SUN_EARTH# x coordinate of Earth
state_vector0[7] = v_cosmic1_sun_earth# vy of Earth
state_vector0[8] = D_SUN_EARTH + D_EARTH_MOON# x coordinate of Moon
state_vector0[11] = v_cosmic1_sun_earth + v_cosmic1_earth# vy of Moon
state_vector0[12::4] = x_planetoids
state_vector0[13::4] = y_planetoids
state_vector0[14::4] = vx_planetoids
state_vector0[15::4] = vy_planetoids

#------------------------------------------------------------------------------
#RESULTS
#------------------------------------------------------------------------------

#plotting initial position
plt.plot(state_vector0[::4], state_vector0[1::4], 'o')
plt.title("Initial positions of all bodies")
plt.xlim([-3 * 10**11, 3 * 10**11])
plt.ylim([-3 * 10**11, 3 * 10**11])
plt.show()

#solving the differential equation
l_solution = odeint(equation, state_vector0, times, args=(m_row,))
# print(l_solution)

#plotting trajectories of n first bodies every n_days days, n>=3
plot_trajectories(n=5, n_days=15)

#finding collisions
l_all_collisions = np.array([])
l_angles = np.array([])#radians
l_collisions_found = find_collisions(l_all_collisions, l_angles, l_solution)
l_all_collisions = l_collisions_found[0]
l_angles = l_collisions_found[1]
# print("all collisions indices", l_all_collisions)
# print("angles", l_angles)
print("collisions detected: ", l_all_collisions.size)

#plotting angles distribution
plot_angles_distribution(l_angles)

l_times = [(time, name) for name, time in times_sum.items()]
l_times.sort()

print("\nfunctions times:")
for time, name in l_times:
    print(name, " : ", time)
