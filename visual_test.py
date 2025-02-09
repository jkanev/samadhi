
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# Create sum of sine waves of different frequencies
dt = 0.005
t0 = np.arange(0, 5*np.pi, dt)
f = [(abs(np.sin(1*t0))),
     (abs(np.sin(2*t0))),
     (abs(np.sin(3*t0))),
     (abs(np.sin(4*t0))),
     (abs(np.sin(5*t0))),
     ((np.sin(2*t0)+1)*10.0)]

# Create 10 circles of different lengths
M = 40 # number of circles
N = 200 # points per circle
lines = [None] * M
ylim = 32
thickness = 0.5
ax = None
fig = None
freq_start = 1.0
freq_step = 0.005
r = [[]] * M  # the interpolated circle data, zeros for now, consisting of M*N circles s[M][1..5], all the same length
s = [[]] * M  # the raw circle data, zeros for now, consisting of M*N circles s[M][1..5]
t = [[]] * M  # the base to plot against, will go from 0 to 2pi, t[M]
for m in range(0, M):
    s[m] = [[]]*5
    freq = freq_start + m*freq_step  # our circle frequency
    print(freq)
    for n in range(0, 5):
        s[m][n] = np.zeros(int((2 / freq) * np.pi / dt))
    #t[m] = [m for m in np.arange(0, 2 * np.pi * (1 + 2 / len(s[m][0])), 2 * np.pi / (len(s[m][0])))[0:len(s[m][0])]]
    t[m] = [m for m in np.arange(0, 2 * np.pi*(1.0 + 1.0/N), 2 * np.pi / (N-1.0))]

# Average over all circles - pre-calculate parts of the sum of sines
for m in range(0, M):
    for n in range(0, 5):
        # only plot if you can fill the entire circle
        for i in range(0, (len(f[n]) // len(s[m][n])) * len(s[m][n])):
            value = f[n][i] - 0.5
            index = i % len(s[m][n])
            s[m][n][index] += value  # add to front
            s[m][n][-(index + 1)] += value  # add to back, so beginning and end match and the circle stays closed

# downsample so all circles have the same length (N samples)
for m in range(0, M):
    r[m] = [[]]*5
    for n in range(0, 5):
        r[m][n] = np.zeros(N)
        for i in range(0, N-1):
            index = int(float(i)*(len(s[m][n])-1)/(N-1.0))
            r[m][n][i] = s[m][n][index]
        r[m][n][N-1] = s[m][n][0]    # close the circle

first = True
s = 0    # index into speed vector
p = 0.0
q = 1.0
k = 0
for f0 in np.arange(0, 1000, 0.01):
    c1 = np.square(np.sin(2*f0))
    c2 = np.square(np.sin(3*f0))
    c3 = np.square(np.sin(4*f0))
    c4 = np.square(np.sin(5*f0))
    c5 = np.square(np.sin(6*f0))
    norm = c1 + c2 + c3 + c4 + c5
    print("Frequency bands: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(c1, c2, c3, c4, c5), end='\r')

    # initialise plot if we're here for first time
    if first:
        #plt.plot(t[0], s[0][0])
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_theta_zero_location("N")
        ax.set_ylim(0.0, ylim)
        ax.set_facecolor('k')
        fig.set_facecolor('k')
        fig.canvas.toolbar.pack_forget()
        plt.subplots_adjust(top=1, bottom=0)
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show(block=False)

    # plot curves
    last_data = []
    for m in range(0, M):
        #offset = np.sqrt((M-m))
        offset = M-m
        offset = np.sqrt(np.sqrt(offset*offset*offset))  # x^(3/4)
        data = c1 * np.roll(r[m][0], k) + c2 * np.roll(r[m][1], 0) + c3 * np.roll(r[m][2], -k) + c4 * np.roll(r[m][3], 0) + c5 * np.roll(r[m][4], k) + 1
        data /= data.max()
        data *= offset
        data += offset
        if first:
            try:
                if len(last_data):
                    [lines[m]] = ax.plot(t[m], p*last_data + q*data, '.', linewidth=thickness, color=(m/M, 0.5*m/M, 1-m/M))
            except BaseException as e:
                print("Plot exception {} for curve n={}".format(e, m))
        else:
            try:
                if len(last_data):
                    lines[m].set_ydata(p*last_data + q*data)
                    if m == 1:
                        lines[m].set_color((c1*q*m/M, c2*q*0.5*m/M, (1.0 - 0.5*c3)*q*(1 - m/M)))
                    elif m == M-1:
                        lines[m].set_color((c1*p*m/M, c2*p*0.5*m/M, (1.0 - 0.5*c3)*p*(1 - m/M)))
                    else:
                        lines[m].set_color((c1*m/M, c2*0.5*m/M, (1.0 - 0.5*c3)*(1-m/M)))
            except BaseException as e:
                print("Plot exception {} for curve n={}".format(e, m))
        last_data = data
    plt.pause(0.001)
    first = False
    s += 1
    if s > len(f[5]):
        s = 0
    p = f[5][s] % 1
    q = 1.0 - p
    k += 1
