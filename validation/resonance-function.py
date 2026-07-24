%matplotlib widget
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
from numpy.fft import fft, fftfreq, fftshift
from tqdm.notebook import tqdm
from numba import njit
from joblib import Parallel, delayed


# NSTEPS = 512*64
# NPAD = NSTEPS
LAMDA_RES = 32
DT = 2 * np.pi / LAMDA_RES

@njit
def efield(t, x, y, z, w0, sigma_lz, tpow = 1.0, beta_ff = 0):
    # All the coordiantes and lengths are normalized to \lambda / 2 / pi
    # Propagates in the -z direction
    z_rayleigh = w0 * w0 * ( 1.0 + beta_ff ) / 2
    
    r_sq = x**2 + y**2
    curv = z / (z**2 + z_rayleigh**2) # Sign of the wavefront curvature does not depend on the propagation direction
    gouy = np.arctan(z / z_rayleigh)
    phase = z + t + r_sq / 2 * curv - gouy
    tenv = np.exp( - np.power(( phase / sigma_lz )**2 / 2, tpow) )
    sigma_l_sq = w0 * w0 * ( 1.0 + ( z - beta_ff * t )**2 / z_rayleigh**2 )

    return np.exp( - r_sq / sigma_l_sq + 1j * phase ) * w0 / np.sqrt(sigma_l_sq) * tenv

def calculate_radiation(ef, wr, dphi0, gtheta_sq):
    # This function calculates the shape of a resonance function defined as
    # R(w) = wr * | \int E(wr*x) exp(i*x*(w-wr)) dx |^2 / 2 \pi \int |E|^2(x) dx
    # Where wr = 2 / ( 1 - \vb{v} \cdot \vb{n} )
    # It also returns actual radiation spectrum to view angle defined by gtheta_sq
    dur = np.sum(np.abs(ef)**2) * dphi0
    dphi = dphi0 / wr
    norm = wr / 2 / np.pi / dur
    f_pad = np.pad(ef, 2*ef.shape[0], constant_values=0.0)
    spec = np.abs(fftshift(fft(f_pad) * dphi))**2
    # The spectrum of the radiation is proportional to the spectrum of the experienced field with a time scaled by a factor wr
    freq = 2*np.pi*fftshift(fftfreq(f_pad.shape[0], dphi)) - wr # -wr corresponds to shifting the fourier spectrum to get frequency samples for the resonance function
    res = norm * spec # this is the resonance function
    rad = spec * wr**2 / 2 / np.pi / ( 1 + gtheta_sq ) # this is actual spectrum
    return freq, res, dur, rad

GAUSS_WIDTH   = 6.0 * 2.0
LORENTZ_WIDTH = 12.0 * 2.0
def t_int(z0, sigmaz, w, beta):
    # returns start and end of interaction of a particle moving as z = z0 + t with the laser pulse
    zr = w**2 * (1 + beta) / 2
    sigma_tau = GAUSS_WIDTH * sigmaz
    sigma_raileigh  = LORENTZ_WIDTH * zr
    t0 = (max(-sigma_tau, ( - z0 * ( 1 + beta ) - 2 * sigma_raileigh ) / ( 1 - beta ) ) - z0) / 2
    t1 = (min( sigma_tau, ( - z0 * ( 1 + beta ) + 2 * sigma_raileigh ) / ( 1 - beta ) ) - z0) / 2
    return t0, t1

def process_particle(i, x, y, z, vx, vy, vz, w, sigma, betaff, theta_n, w_interp, tpow):
    xi, yi, zi = x[i], y[i], z[i]
    vxi, vyi, vzi = vx[i], vy[i], vz[i]

    vmod_sq = vxi**2 + vyi**2 + vzi**2
    vmod = np.sqrt(vmod_sq)
    gamma = 1.0 / np.sqrt(1 - vmod_sq)

    t0, t1 = t_int(zi, sigma, w, betaff)
    ts = np.arange(t0, t1, DT)
    ef = efield(ts, xi + vxi * ts, yi + vyi * ts, zi + vzi * ts, w, sigma, tpow, betaff)
    wr = 2 / ( 1 - vzi * np.cos(theta_n) - vxi * np.sin(theta_n) )
    dphi0 = DT * ( 1 + vzi )
    freq, res, dur, rad = calculate_radiation(ef, wr, dphi0, gamma**2 * (theta_n + vxi/vmod)**2 + (vyi/vmod)**2 )
    return res, np.interp(w_interp, freq, rad), dur

lamda = 1e-4
betaff = 0.0
w0 = 2*np.pi*1.3 # distance at which the *field* decreases to 1/e measured in 1 / k
sigma = 2*np.pi*60.0
print(f"Z_R = {2*w0**2}, dur = {sigma}")
zr = w0**2 * (1 + betaff) / 2
tpow = 4.0
sigmaw_las = 1.0 / sigma

gamma0 = 2000 / 0.511
nemit_x = 5e-4
nemit_y = 5e-4

emit_x = nemit_x / gamma0
emit_y = nemit_y / gamma0

sx = 3.0*w0# 2*np.pi*3.0
sy = 3.0*w0# 2*np.pi*3.0
sz = 3.0*w0# 2*np.pi*3.0

svx = emit_x / (lamda*sx/2/np.pi)
svy = emit_y / (lamda*sy/2/np.pi)

cov = np.diag((sx**2, sy**2, sz**2, svx**2, svy**2))

delta_z = 2*np.pi*0

N_INTERP = 512*16
vmod = np.sqrt(1 - 1 / gamma0**2)
theta_n = 0 / gamma0
wr0 = 2 / ( 1 - vmod * np.cos(theta_n) )
w_interp = np.linspace(-0.15*wr0,0.15*wr0,N_INTERP)

Npar = 2048*8
x, y, z, vx, vy = np.random.multivariate_normal([0.0, 0.0, delta_z, 0.0, 0.0], cov, Npar).T
vz = np.sqrt(vmod**2 - vx**2 - vy**2)
dt0 = z / vz
x += vx * dt0 # making particle position self consistent
y += vy * dt0

N_THREADS = 14
results = Parallel(n_jobs=N_THREADS, verbose=4)(
    delayed(process_particle)(
        i, x, y, z, vx, vy, vz, w0, sigma, betaff, theta_n, w_interp, tpow,
    ) for i in range(Npar)
)

res = np.array([r[0] for r in results])
interps = np.array([r[1] for r in results])
weights = np.array([r[2] for r in results])

fig, ax = plt.subplots()

theta0 = theta_n

t0, t1 = t_int(delta_z, sigma, zr, betaff)
ts = np.arange(t0, t1, DT)
print(np.diff(ts)[0]*2*np.pi)

dphi0 = np.diff(ts)[0] * ( 1 + vmod )
e_axis = efield(ts, 0.0, 0.0, delta_z + ts*vmod, w0, sigma, tpow, betaff)
freq0, res0, dur0, rad0 = calculate_radiation(e_axis, wr0, dphi0, gamma0**2 * theta0**2)

spmax = interps.max()
nres = 256
idxs = np.argsort(weights)[:nres][::-1]
# for i in range(nres):
#     rad = interps[idxs][i]
#     ax.plot(w_interp/wr0, rad,alpha=0.4)

# rad_sum = (interps * weights[:,np.newaxis]).sum(axis=0) / weights.sum()
rad_sum = (interps).sum(axis=0)

ax.plot(freq0/wr0, rad0/rad0.max(),'--r',lw=1.5)
ax.plot(w_interp/wr0, rad_sum/rad_sum.max(),'-k',lw=2)
# ax.plot(w_interp/wr0, res_av/res_av.max(),'--r',lw=1.5)
# ax.plot(w_interp/wr0, np.exp(-(w_interp/wr0)**2/2/sigmaw_las**2),'--g',lw=1.5)
ax.set_xlim(-0.15,0.15)