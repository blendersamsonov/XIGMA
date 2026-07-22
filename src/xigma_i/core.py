import numpy as np
import cupy as cp
from cupyx import jit
from cupyx.scipy.special import erf as cp_erf
from cupyx.scipy.special import erfinv as cp_erfinv
from scipy.special import erfcx

hbar = 1.054571800e-27               # постоянная Планка
me = 9.10938356e-28                  # масса электрона
c = 2.99792458e+10                   # скорость света
el = 4.803204673e-10                 # заряд электрона (в статкулонах)
elC = 1.602176634e-19                # заряд электрона в Кулонах
rel = el ** 2 / (me * c ** 2)        # классический радиус электрона
sigma_T = 8.0 * np.pi / 3.0 * rel**2 # Томсоновское сечение рассеяния
alpha = el ** 2 / (hbar * c)         # постоянная тонкой структуры
PHI = 1.618033988749894848           # golden ratio

SINGLE_PRECISION = True

CP_FLOAT  = cp.float32 if SINGLE_PRECISION else cp.float64
CP_UINT   = cp.uint32  if SINGLE_PRECISION else cp.uint64
CP_INT    = cp.int32   if SINGLE_PRECISION else cp.int64
CP_PI     = CP_FLOAT(cp.pi)
CP_TWO_PI = CP_FLOAT(2.0 * cp.pi)
CP_ONE    = CP_FLOAT(1.0)
CP_ZERO   = CP_FLOAT(0.0)

X_THREADS = 128

N_RINGS_MIN = 32
MAX_RINGS = 32
MAX_ARCS = 4 * MAX_RINGS
ARC_STRIDE = 3
RING_STRIDE = 9
RINGS_SIZE = CP_UINT(RING_STRIDE * MAX_RINGS)
INVAL = CP_FLOAT(9999.)

PHI_EDGES = 32
PHI_CELLS = PHI_EDGES - 1
WEIGHTS_SIZE = MAX_ARCS * PHI_CELLS
WEIGHTS_REPEAT = ( WEIGHTS_SIZE + X_THREADS - 1 ) // X_THREADS
CUM_WEIGHTS_SIZE = MAX_ARCS * PHI_EDGES

CDF_PHI_RESOLUTION = 32
CDF_PHI_REPEAT = ( CDF_PHI_RESOLUTION + X_THREADS - 1 ) // X_THREADS
CDF_SIZE = CDF_PHI_RESOLUTION * MAX_ARCS

SAMPLES_MIN = 16
SAMPLES_TOTAL = 256
SAMPLES_REPEAT = ( SAMPLES_TOTAL + X_THREADS - 1) // X_THREADS
THREAD_STRIDE = 3 * SAMPLES_REPEAT + 1

R_MAX_NUDGE = 128

GAUSS_WIDTH   = CP_FLOAT(3)
LORENTZ_WIDTH = CP_FLOAT(8)

N_STEPS = CP_UINT(X_THREADS)

@jit.rawkernel()
def particle_kernel(intersect, envelope, particles, THX, THY, f_th, w0, beta_ff, sigma_lz, t0, t1, dvx, dvy):#, debug, xy_debug): 
    # Make sure arguments and arrays are of type CP_FLOAT
    z_rayleigh = 2 * w0 * w0 * ( CP_ONE + beta_ff )
    
    step_idx = jit.threadIdx.x
    xy_idx = jit.blockIdx.y
    p_idx = jit.blockIdx.x
    
    weight = f_th[xy_idx]
    n_particles = CP_UINT( cp.ceil( weight * particles.shape[0] ) )
    if p_idx < n_particles:
        x0  = particles[p_idx, 0]
        y0  = particles[p_idx, 1]
        z0  = particles[p_idx, 2]

        t_start = particles[p_idx, 3]
        t_end   = particles[p_idx, 4]
        
        vx = THX[xy_idx]
        vy = THY[xy_idx]

        reg = p_idx / n_particles
        fib = cp.remainder(CP_ONE / 2 + p_idx / PHI, CP_ONE)
        vx += ( 2 * reg - CP_ONE ) * dvx / 2 # Spreading velocities uniformly across the cell
        vy += ( 2 * fib - CP_ONE ) * dvy / 2
        vz = cp.sqrt( CP_ONE - vx**2 - vy**2)
        dt0 = z0 / vz

        dt = ( t_end - t_start ) / N_STEPS

        t = step_idx * dt + t_start
        x = x0 + vx * ( t + dt0 )
        y = y0 + vy * ( t + dt0 )
        z = z0 + vz * t

        # Laser propagates along -z axis
        env = cp.exp( -( ( z + t ) / sigma_lz )**2 / 2 ) / cp.sqrt( CP_TWO_PI ) / sigma_lz
        sigma_l_sq = w0 * w0 * ( CP_ONE + ( z - beta_ff * t )**2 / z_rayleigh**2 )
        f_cur = cp.exp( - ( x**2 + y**2 ) / sigma_l_sq / 2 ) / CP_TWO_PI / sigma_l_sq * env
        f_cur *= dt * weight / n_particles

        if t >= t0 and t < t1:
            env_fac = envelope.shape[0] * ( t - t0 ) / ( t1 - t0 )
            env_idx = CP_UINT( cp.floor( env_fac ) )
            env_fac = cp.remainder(env_fac, CP_ONE)
            # if  xy_idx == xy_debug:
            #     debug[p_idx, step_idx, 0] = f_cur
            #     debug[p_idx, step_idx, 1] = x
            #     debug[p_idx, step_idx, 2] = y
            #     debug[p_idx, step_idx, 3] = z
            jit.atomic_add(envelope, env_idx,     f_cur * (CP_ONE - env_fac))
            jit.atomic_add(envelope, env_idx + 1, f_cur * env_fac)

        jit.atomic_add(intersect, xy_idx , f_cur)

@jit.rawkernel()
def spectrum_kernel(output, params_Arr, collision, sigma_g, gamma0, dx, dy, phi_pol, subsampling, debug_arr, debug_arcs, debug_cdf, debug_thread, debug_weight, debug_idx):
    # Calculates number of photons with energies defined by s_Arr and direction x0, y0.
    # Requires arcs array, calculated by arcs_kernel
    thread_idx = jit.threadIdx.x
    out_idx = jit.blockIdx.x

    # Shared memory allocations
    inv_cdf = jit.shared_memory(CP_FLOAT, CDF_SIZE)
    # We can use large inv_cdf array for temporary storage, untill actuall inv_cdf is required
    TMP_FLOAT_ARRAY = inv_cdf

    n_arcs_shared = jit.shared_memory(CP_UINT, 1)
    arcs  = jit.shared_memory(CP_FLOAT, ARC_STRIDE * MAX_ARCS)
    cum_cell_weights = jit.shared_memory(CP_FLOAT, CUM_WEIGHTS_SIZE)
    thread_samples = jit.shared_memory( CP_UINT, X_THREADS*THREAD_STRIDE )

    x0 = params_Arr[out_idx, 0]
    y0 = params_Arr[out_idx, 1]
    s  = params_Arr[out_idx, 2]

    rmin_g = cp.sqrt(cp.maximum( CP_ZERO, CP_ONE / s - CP_ONE / ( gamma0 - 3 * sigma_g )**2 ) )
    rmax_g = cp.sqrt(cp.maximum( CP_ZERO, CP_ONE / s - CP_ONE / ( gamma0 + 3 * sigma_g )**2 ) )
    
    rmin_r = cp.sqrt(max(cp.abs(x0) - dx, CP_ZERO)**2 + max(cp.abs(y0) - dy, CP_ZERO)**2)

    diam = 2 * cp.sqrt(dx**2 + dy**2)
    xm = dx + cp.abs(x0)
    ym = dy + cp.abs(y0)
    rmax_r = cp.sqrt( xm**2 + ym**2 ) - diam / R_MAX_NUDGE

    rmin = max(rmin_g, rmin_r)
    rmax = min(rmax_g, rmax_r)

    skip = rmin >= rmax
    if not skip:
        r_inside = max( CP_ZERO, min(dx - cp.abs(x0), dy - cp.abs(y0)) )
        
        n_rings = max(N_RINGS_MIN, CP_UINT( MAX_RINGS * ( rmax - rmin ) / diam ) )

        dr = ( rmax - rmin ) / n_rings

        rings = TMP_FLOAT_ARRAY # need RING_STRIDE elements for each ring to temporary store arcs
        phi_cur = TMP_FLOAT_ARRAY # need 2 variables for each thread, indicating start and end of the current arc. Using elements after rings elements
        if thread_idx < n_rings:
            phi_cur[RINGS_SIZE + 2*thread_idx + 0] = -INVAL
            phi_cur[RINGS_SIZE + 2*thread_idx + 1] =  INVAL
            
            r_idx = thread_idx
            r = rmin + dr * ( CP_FLOAT(r_idx) + CP_FLOAT(0.5) )
            
            n_arcs = CP_UINT(0)
            
            if r < r_inside:
                rings[r_idx*RING_STRIDE + 0] = CP_ONE
                rings[r_idx*RING_STRIDE + 1] = CP_ZERO
                rings[r_idx*RING_STRIDE + 2] = CP_TWO_PI

            else:
                for q_idx in jit.range(4):
                    sin_pos = CP_UINT((q_idx // 2))
                    # 1 | 1
                    # --+--
                    # 0 | 0
                    cos_pos = CP_UINT(((q_idx+1)//2)%2)
                    # 0 | 1
                    # --+--
                    # 0 | 1

                    sin_sign = CP_INT(2 * sin_pos - 1)
                    # + | +
                    # --+--
                    # - | -
                    cos_sign = CP_INT(2 * cos_pos - 1)
                    # - | +
                    # --+--
                    # - | +

                    cos_0 = (  dx - x0 ) / r
                    sin_0 = cp.sqrt(CP_ONE - cos_0**2)
                    
                    cos_1 = (- dx - x0 ) / r
                    sin_1 = cp.sqrt(CP_ONE - cos_1**2)

                    sin_2 = (  dy - y0 ) / r
                    cos_2 = cp.sqrt(CP_ONE - sin_2**2)

                    sin_3 = (- dy - y0 ) / r
                    cos_3 = cp.sqrt(CP_ONE - sin_3**2)

                    if cos_sign * cos_0 > 0 and cp.abs(y0 + r * sin_0 * sin_sign) < dy: # Right edge
                        phi_cur[RINGS_SIZE + 2*thread_idx + (1 - sin_pos)] = cp.arctan2(sin_0 * sin_sign, cos_0)

                    if cos_sign * cos_1 > 0 and cp.abs(y0 + r * sin_1 * sin_sign) < dy: # Left edge
                        phi_cur[RINGS_SIZE + 2*thread_idx + (sin_pos)] = cp.arctan2(sin_1 * sin_sign, cos_1)

                    if sin_sign * sin_2 > 0 and cp.abs(x0 + r * cos_2 * cos_sign) < dx: # Top edge
                        phi_cur[RINGS_SIZE + 2*thread_idx + (cos_pos)] = cp.arctan2(sin_2, cos_2 * cos_sign)

                    if sin_sign * sin_3 > 0 and cp.abs(x0 + r * cos_3 * cos_sign) < dx: # Bottom edge
                        phi_cur[RINGS_SIZE + 2*thread_idx + (1 - cos_pos)] = cp.arctan2(sin_3, cos_3 * cos_sign)

                    if phi_cur[RINGS_SIZE + 2*thread_idx + 1] < 1000.:
                        rings[r_idx*RING_STRIDE + 1 + 2*n_arcs + 0] = phi_cur[RINGS_SIZE + 2*thread_idx + 0]
                        rings[r_idx*RING_STRIDE + 1 + 2*n_arcs + 1] = phi_cur[RINGS_SIZE + 2*thread_idx + 1]

                        phi_cur[RINGS_SIZE + 2*thread_idx + 0] = -INVAL
                        phi_cur[RINGS_SIZE + 2*thread_idx + 1] =  INVAL

                        n_arcs += CP_UINT(1)

                if phi_cur[RINGS_SIZE + 2*thread_idx + 0] > -1000. and rings[r_idx*RING_STRIDE + 1] < -1000.: # Need to merge 4th and 1st quadrants
                    rings[r_idx*RING_STRIDE + 1] = phi_cur[RINGS_SIZE + 2*thread_idx + 0] - CP_TWO_PI

                rings[r_idx*RING_STRIDE + 0] = CP_FLOAT(n_arcs)

    jit.syncthreads()

    if not skip:
        n_arcs = CP_UINT(0)
        if thread_idx == 0:
            for i in jit.range(CP_INT(n_rings)):
                n_ring_arcs = CP_INT(rings[i*RING_STRIDE+0])
                for j in jit.range(n_ring_arcs):
                    arcs[n_arcs*ARC_STRIDE + 0] = rmin + dr * ( CP_FLOAT(i) + CP_FLOAT(0.5) )
                    arcs[n_arcs*ARC_STRIDE + 1] = rings[i*RING_STRIDE + 1 + 2*j + 0]
                    arcs[n_arcs*ARC_STRIDE + 2] = rings[i*RING_STRIDE + 1 + 2*j + 1]

                    # if out_idx == debug_idx:
                    #     debug_arcs[n_arcs, 0] = arcs[n_arcs*ARC_STRIDE + 0]
                    #     debug_arcs[n_arcs, 1] = arcs[n_arcs*ARC_STRIDE + 1]
                    #     debug_arcs[n_arcs, 2] = arcs[n_arcs*ARC_STRIDE + 2]

                    n_arcs += CP_UINT(1)
            n_arcs_shared[0] = n_arcs

    jit.syncthreads()
    
    if not skip:
        n_arcs = n_arcs_shared[0]
        
        ## Calculating weight of each arc first, by sparsely sampling the collision array and approximate specturm shape
        cell_weights = cum_cell_weights
        
        weights_size = n_arcs * PHI_EDGES
        weights_repeat = CP_INT( ( weights_size + X_THREADS - 1 ) // X_THREADS )
        
        for i in jit.range(weights_repeat):
            sample_idx = CP_UINT(i * X_THREADS) + thread_idx
            phi_idx = sample_idx %  PHI_EDGES # calculating indecies as if the stride is PHI_EDGES to have the same memory layout as for cum_cell_weight
            arc_idx = sample_idx // PHI_EDGES
            
            if arc_idx < n_arcs and phi_idx < PHI_CELLS:
                r       = arcs[arc_idx*ARC_STRIDE + 0]
                phi_min = arcs[arc_idx*ARC_STRIDE + 1]
                phi_max = arcs[arc_idx*ARC_STRIDE + 2]

                g = cp.sqrt( CP_ONE / ( CP_ONE / s - r**2 ) )
                fr = cp.exp(-( g - gamma0 )**2 / 2 / sigma_g**2 )
                
                phi = phi_min + ( ( phi_idx + CP_ONE / 2 ) / PHI_CELLS ) * ( phi_max - phi_min )
                x = x0 + r * cp.cos(phi)
                y = y0 + r * cp.sin(phi)
                
                xi = CP_UINT( cp.floor( ( collision.shape[0] - 1 ) * ( x + dx ) / dx / 2 ) )
                yj = CP_UINT( cp.floor( ( collision.shape[1] - 1 ) * ( y + dy ) / dy / 2 ) )

                if (x > -dx and x < dx and y > -dy and y < dy):
                    w = collision[xi, yj]
                else:
                    w = CP_ZERO
                
                w *= fr
                cell_weights[sample_idx] = w * ( phi_max - phi_min ) * r

                # if debug_idx == out_idx:
                #     debug_weight[arc_idx, phi_idx] = cell_weights[sample_idx]

    jit.syncthreads()

    # Calculating cumulative weight of cells at the cell edges
    if not skip:
        if thread_idx < n_arcs:
            total = CP_ZERO
            tmp = CP_ZERO
            for i in jit.range(PHI_EDGES):
                tmp = cell_weights[thread_idx*PHI_EDGES + CP_UINT(i)]
                cum_cell_weights[thread_idx*PHI_EDGES + CP_UINT(i)] = total
                total += tmp
    jit.syncthreads()

    # Calculating total weight of all arcs
    if not skip:
        if thread_idx == 0:
            TMP_FLOAT_ARRAY[0] = CP_ZERO
            for i in jit.range(CP_INT(n_arcs)):
                TMP_FLOAT_ARRAY[0] += cum_cell_weights[CP_UINT(i*PHI_EDGES) + ( PHI_EDGES - 1 )]

    jit.syncthreads()

    if not skip:
        total_weight = TMP_FLOAT_ARRAY[0]
        # Distributing samples accross the threads according to their weight
        thread_samples[thread_idx * THREAD_STRIDE] = CP_UINT(0) # total number of samples in current thread
        if thread_idx == 0:
            cur_thread = CP_UINT(0)
            for k in jit.range(CP_INT(n_arcs)):
                arc_weight = cum_cell_weights[k*PHI_EDGES + ( PHI_EDGES - 1 )]
                s_add = CP_UINT( cp.floor( SAMPLES_TOTAL * arc_weight / total_weight ) )
                for j in jit.range( CP_INT(s_add) ):
                    n_samples = thread_samples[cur_thread*THREAD_STRIDE + 0]
                    thread_samples[cur_thread*THREAD_STRIDE + 1 + 3 * n_samples + 0] = CP_UINT(k)     # index of an arc the sample belongs to
                    thread_samples[cur_thread*THREAD_STRIDE + 1 + 3 * n_samples + 1] = CP_UINT(j)     # index of a sample within the arc
                    thread_samples[cur_thread*THREAD_STRIDE + 1 + 3 * n_samples + 2] = CP_UINT(s_add) # total number of samples per arc
                    thread_samples[cur_thread*THREAD_STRIDE + 0] += CP_UINT(1)
                    cur_thread = ( cur_thread + CP_UINT(1) ) % X_THREADS
    
    jit.syncthreads()

    if not skip:
        # Tabulating inverse_cdf for each arc
        for arc_idx in jit.range(n_arcs):
            phi_min = arcs[arc_idx*ARC_STRIDE + 1]
            phi_max = arcs[arc_idx*ARC_STRIDE + 2]
            dphi = ( phi_max - phi_min ) / PHI_CELLS
            for k in jit.range(CDF_PHI_REPEAT):
                r_idx = CP_UINT(k * X_THREADS) + thread_idx
                if r_idx < CDF_PHI_RESOLUTION:
                    r = cum_cell_weights[arc_idx*PHI_EDGES + ( PHI_EDGES - 1 )] * r_idx / ( CDF_PHI_RESOLUTION - 1 )
                    left  = CP_UINT(0)
                    right = CP_UINT(PHI_EDGES - 1)
                    while right - left > 1:
                        mid = ( left + right ) // 2
                        if cum_cell_weights[arc_idx*PHI_EDGES + mid] <= r:
                            left = mid
                        else:
                            right = mid
                    
                    cdf_i   = cum_cell_weights[arc_idx*PHI_EDGES + (left + 0)]
                    cdf_ip1 = cum_cell_weights[arc_idx*PHI_EDGES + (left + 1)]
                    fac = ( r - cdf_i ) / ( cdf_ip1 - cdf_i )
                    inv_cdf[arc_idx*CDF_PHI_RESOLUTION + r_idx] = phi_min + ( CP_FLOAT(left) + fac ) * dphi

                    # if debug_idx == out_idx:
                    #     debug_cdf[r_idx, arc_idx] = inv_cdf[arc_idx*CDF_PHI_RESOLUTION + r_idx]
        
    jit.syncthreads()

    if not skip:
        # Now we evaluate the function at each point and sum them with account of both their statistical weight and geometrical weight
        f_tot = CP_ZERO
        n_thread_samples  = thread_samples[thread_idx*THREAD_STRIDE + 0]
        for thread_sample_idx in jit.range(CP_UINT(SAMPLES_REPEAT)):
            if thread_sample_idx < n_thread_samples:
                arc_idx        = thread_samples[thread_idx*THREAD_STRIDE + 1 + 3 * thread_sample_idx + 0]
                arc_sample_idx = thread_samples[thread_idx*THREAD_STRIDE + 1 + 3 * thread_sample_idx + 1]
                n_arc_samples  = thread_samples[thread_idx*THREAD_STRIDE + 1 + 3 * thread_sample_idx + 2]

                arc_r   = arcs[arc_idx*ARC_STRIDE + 0]
                phi_min = arcs[arc_idx*ARC_STRIDE + 1]
                phi_max = arcs[arc_idx*ARC_STRIDE + 2]
                arc_total_weight = cum_cell_weights[arc_idx*PHI_EDGES + ( PHI_EDGES - 1 )]
                arc_area = ( phi_max - phi_min ) * arc_r * dr

                for di in jit.range(subsampling):
                    subsample_idx = arc_sample_idx * subsampling + di # index of the subsample within the current arc
                    
                    reg = ( subsample_idx + 0.5 ) / n_arc_samples / subsampling # regularly distributed among all samples in the current arc from 0 to 1
                    fib = cp.remainder( subsample_idx * PHI, 1.0 )

                    theta_min = arc_r - dr / 2
                    theta_max = theta_min + dr

                    theta_sq = theta_min**2 + fib * ( theta_max**2 - theta_min**2 ) # To achieve uniform distribution in a disk using Fibonacci algorithm, *squares* of radii should be distributed unformly, not the radii themselves!    
                    theta = cp.sqrt(theta_sq)

                    # theta = arc_r
                    # theta_sq = arc_r**2
                    
                    il = CP_UINT( cp.floor( reg * ( CDF_PHI_RESOLUTION - 1 ) ) )
                    fac = reg * ( CDF_PHI_RESOLUTION - 1 ) - CP_FLOAT(il)
                    phi = inv_cdf[arc_idx*CDF_PHI_RESOLUTION + il] * ( CP_ONE - fac ) + inv_cdf[arc_idx*CDF_PHI_RESOLUTION + (il + 1)] * fac

                    phi_idx = min( PHI_CELLS - 1, CP_UINT( PHI_CELLS * ( phi - phi_min ) / ( phi_max - phi_min ) ) ) # closest cell index
                    cell_weight = cum_cell_weights[arc_idx*PHI_EDGES + phi_idx + 1] - cum_cell_weights[arc_idx*PHI_EDGES + phi_idx] # Cell weight is difference between neighbours of cumulative weight
                    sample_area = arc_area / n_arc_samples / subsampling * arc_total_weight / cell_weight # Area in theta_x theta_y plane "occupied" by each sample
                    
                    x = x0 + theta * cp.cos(phi)
                    y = y0 + theta * cp.sin(phi)
                    
                    g_sq = CP_ONE / ( CP_ONE / s - theta_sq )
                    if g_sq >= 0: # Heavyside

                        Xi = (collision.shape[0] - 1) * ( x + dx ) / dx / 2
                        Yj = (collision.shape[1] - 1) * ( y + dy ) / dy / 2
                        
                        xi = CP_UINT(cp.floor(Xi))
                        yj = CP_UINT(cp.floor(Yj))

                        Xi = cp.remainder(Xi, CP_ONE)
                        Yj = cp.remainder(Yj, CP_ONE)

                        if (x > -dx and x < dx and y > -dy and y < dy): # occasionally a sample might be completely outside the collision rectangle
                            # linear interpolation
                            col = collision[xi    , yj    ] * (CP_ONE - Yj) * (CP_ONE - Xi) \
                                + collision[xi + 1, yj    ] * (CP_ONE - Yj) * (Xi) \
                                + collision[xi    , yj + 1] * (Yj)          * (CP_ONE - Xi) \
                                + collision[xi + 1, yj + 1] * (Yj)          * (Xi) 
                        else:
                            col = CP_ZERO
                                
                        g = cp.sqrt(g_sq)
                        ffac = cp.exp(-(g - gamma0)**2 / 2 / sigma_g**2) / cp.sqrt(CP_TWO_PI*sigma_g**2)

                        gth_sq_inv = CP_ONE / ( CP_ONE + theta_sq * g_sq )**2

                        cos_pol = cp.cos( phi_pol - phi )**2
                        a_fac = CP_ONE - 4 * cos_pol * theta_sq * g_sq * gth_sq_inv
                        
                        f = ffac * a_fac * col * g**5 * gth_sq_inv

                        f_tot += f * sample_area

                        # if debug_idx == out_idx:
                        #     debug_arr[arc_idx, subsample_idx, 0] = x
                        #     debug_arr[arc_idx, subsample_idx, 1] = y
                        #     debug_arr[arc_idx, subsample_idx, 2] = f
                            # debug_arr[arc_idx, subsample_idx, 3] = CP_FLOAT(n_arc_samples)

        jit.atomic_add(output, out_idx, f_tot / s**2)

## TODO: Kernels warmup

class Compton:
    # Electron parameters
    chargeNC = None
    # sigma_gamma = None
    emit_x = None
    emit_y = None
    sigma_ex = None
    sigma_ey = None
    sigma_ez = None
    N_e = None
    sigma_l = None

    # Laser parameters
    WL = None
    lambda_l = None
    sigma_lr0 = None
    sigma_lz = None
    omega_las = None
    Wph = None

    # Foci displacement
    delta_x = 0.0
    delta_y = 0.0
    delta_z = 0.0

    # Runtime variables
    device = None
    dtheta = None
    dtheta_x = None
    dtheta_y = None
    THX = None
    THY = None
    intersection = None

    def estimate_yield(self):
        sb_av = np.sqrt(self.sigma_ex * self.sigma_ey / self.beta_x / self.beta_y)
        sigma0 = np.sqrt(self.sigma_ex**2 + self.sigma_lr0**2)
        nu = np.sqrt(2) * sigma0 / np.sqrt(self.sigma_ez**2 + self.sigma_lz**2) / np.sqrt(sb_av**2 + self.lambda_l**2 / np.pi**2 / self.sigma_lr0**2)
        return self.N_e * self.N_l * sigma_T / 2 / np.sqrt(np.pi) / sigma0**2 * nu * erfcx(nu)
    
    def estimate_spectrum_width(self, gamma0, sigma_gamma, theta_col):
        emit_width = np.sqrt(self.sigma_thx*self.sigma_thy)
        return 0.5*2.355*np.sqrt((gamma0 * theta_col)**4 + (gamma0 * emit_width)**4 + (sigma_gamma / gamma0)**2 + (0.5*self.a0**2)**2)

    def set_electron_parameters(self, chargeNC, emit_x, emit_y, sigma_ex, sigma_ey, sigma_ez):
        self.chargeNC = chargeNC
        self.emit_x = emit_x
        self.emit_y = emit_y
        self.sigma_ex = sigma_ex
        self.sigma_ey = sigma_ey
        self.sigma_ez = sigma_ez
        self.N_e = self.chargeNC * 1e-9 / elC

        self.beta_x = self.sigma_ex**2 / self.emit_x
        self.beta_y = self.sigma_ey**2 / self.emit_y
        # try:
        #     self.sigma_l = np.sqrt(self.sigma_ez**2 + self.sigma_lz**2)
        # except:
        #     pass

        self.intersection = None

    def set_laser_parameters(self, WL, lambda_l, sigma_lr0, sigma_lz, beta_ff = 0.0, ellipticity = 0.0):
        self.WL = WL * 1e7 # Энергия пучка в эргах
        self.lambda_l = lambda_l
        self.beta_ff = beta_ff
        self.ellipticity = ellipticity # laser polarisation ellipticity; 0 = linear, +-1 = circular. Used by particles.push_and_sample's TrXi/2 = (1+ellipticity**2)/2 (see CLAUDE.md); not used by the legacy kernels.
        self.sigma_lr0 = sigma_lr0 # NOTE: This is the RMS radius of the *photon density* distribution. Using this Rayleigh range is 2 * sigma_lr0**2 * omega (compare to sigma**2 * omega / 2 for sigma at which the field amplitude is e times smaller than at the maximum)
        self.sigma_lz = sigma_lz
        self.omega_las = 2 * np.pi*c / self.lambda_l
        self.k0_las = self.omega_las / c
        Wph = hbar * self.omega_las # Энергия фотона в эргах
        self.Wph = Wph * 1e-6 / ( elC * 1e7 ) # Энергия фотона в МэВ
        self.N_l = self.WL / Wph
        self.a0 = 4 * rel**2 * lambda_l / alpha * self.N_l / (np.power(np.pi, 3/2) * sigma_lr0**2 * sigma_lz)
        # try:
        #     self.sigma_l = np.sqrt(self.sigma_ez**2 + self.sigma_lz**2)
        # except:
        #     pass

        self.intersection = None

    def set_foci_displacement(self, delta_x, delta_y, delta_z):
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_z = delta_z

        self.intersection = None

    def calculate_intersection(self, theta_num = 128, particles_amount = 4096, debug_idx = 0):

        sigma_thx = CP_FLOAT(self.emit_x / self.sigma_ex)
        sigma_thy = CP_FLOAT(self.emit_y / self.sigma_ey)

        nx = theta_num
        ny = theta_num

        self.sigma_thx = sigma_thx
        self.sigma_thy = sigma_thy

        thxs = cp.linspace(-3.0*sigma_thx, 3.0*sigma_thx, nx, dtype=CP_FLOAT)
        thys = cp.linspace(-3.0*sigma_thy, 3.0*sigma_thy, ny, dtype=CP_FLOAT)

        self.theta_extent = [thxs[0].get(), thxs[-1].get(), thys[0].get(), thys[-1].get()]
        self.THX, self.THY = cp.meshgrid(thxs, thys, indexing='ij')

        self.dtheta_x = float(thxs[1] - thxs[0])
        self.dtheta_y = float(thys[1] - thys[0])
        self.dtheta   = float(cp.sqrt(self.dtheta_x**2 + self.dtheta_y**2))

        f_th = cp.exp(-((self.THX/sigma_thx)**2 + (self.THY/sigma_thy)**2) / 2) 

        N = particles_amount
        reg = (cp.arange(0, N)/N).astype(CP_FLOAT)
        fib = cp.remainder(N * reg * PHI, 1.0).astype(CP_FLOAT)

        p_xs = reg
        p_ys = fib

        sigx = CP_FLOAT(min(self.sigma_lr0, self.sigma_ex) / self.sigma_ex)
        dsx = cp_erf(GAUSS_WIDTH * sigx)
        sigy = CP_FLOAT(min(self.sigma_lr0, self.sigma_ey) / self.sigma_ey)
        dsy = cp_erf(GAUSS_WIDTH * sigy)
        dsx *= CP_FLOAT(1 - 1e-6)
        dsy *= CP_FLOAT(1 - 1e-6)

        p_xs = cp_erfinv((2*p_xs-CP_ONE) * dsx)*CP_FLOAT(self.k0_las*self.sigma_ex)*cp.sqrt(2*CP_ONE)
        p_ys = cp_erfinv((2*p_ys-CP_ONE) * dsy)*CP_FLOAT(self.k0_las*self.sigma_ey)*cp.sqrt(2*CP_ONE)

        z0 = CP_FLOAT(self.k0_las * self.delta_z)
        zR = CP_FLOAT((self.k0_las * self.sigma_lr0)**2 * ( 1.0 + self.beta_ff ) * 2)
        zT = CP_FLOAT(self.k0_las * self.sigma_lz)

        sigma_tau = GAUSS_WIDTH * zT
        sigma_raileigh  = LORENTZ_WIDTH * zR
        zmx = ( ( 1.0 - self.beta_ff ) * sigma_tau + 2 * sigma_raileigh ) / ( 1 + self.beta_ff )
        
        z_min = CP_FLOAT(max(- GAUSS_WIDTH * self.sigma_ez * self.k0_las, -zmx - z0))
        z_max = CP_FLOAT(min(  GAUSS_WIDTH * self.sigma_ez * self.k0_las,  zmx - z0))

        p_zs = cp.random.rand(particles_amount, dtype=CP_FLOAT)
        sigz = CP_FLOAT(self.k0_las*self.sigma_ez)
        pz_min = cp_erf(z_min/sigz/cp.sqrt(2*CP_ONE))
        pz_max = cp_erf(z_max/sigz/cp.sqrt(2*CP_ONE))
        z_weight = ( pz_max - pz_min ) / 2
        p_zs = z0 + cp_erfinv( pz_min + p_zs * (pz_max - pz_min) ) * sigz * cp.sqrt(2*CP_ONE) # Only getting particles that will be inside the laser pulse at some point

        p_t0 = (cp.maximum(-sigma_tau, ( - p_zs * CP_FLOAT( 1 + self.beta_ff ) - 2 * sigma_raileigh ) / CP_FLOAT( 1 - self.beta_ff ) ) - p_zs) / 2
        t_start = CP_FLOAT(cp.min(p_t0).get())

        p_t1 = (cp.minimum( sigma_tau, ( - p_zs * CP_FLOAT( 1 + self.beta_ff ) + 2 * sigma_raileigh ) / CP_FLOAT( 1 - self.beta_ff ) ) - p_zs) / 2
        t_end = CP_FLOAT(cp.max(p_t1).get())

        particles = cp.stack((p_xs,p_ys,p_zs,p_t0,p_t1),axis=1)
        cp.random.shuffle(particles)

        dvx = CP_FLOAT(self.dtheta_x)
        dvy = CP_FLOAT(self.dtheta_y)
        
        self.intersection  = cp.zeros((nx, ny), dtype=CP_FLOAT).flatten()
        self.time_envelope = cp.zeros((N_STEPS,), dtype=CP_FLOAT)
        #debug =  cp.zeros((particles_amount, N_STEPS,4), dtype=CP_FLOAT) * cp.nan
        finish = cp.cuda.Event()
        particle_kernel[(particles_amount, nx*ny), N_STEPS](self.intersection, self.time_envelope, particles, self.THX.flatten(), self.THY.flatten(), f_th.flatten(), CP_FLOAT(self.k0_las * self.sigma_lr0), CP_FLOAT(self.beta_ff), zT, t_start, t_end, dvx, dvy)#, debug, CP_UINT(debug_idx))
                
        finish.record()
        finish.synchronize()
        v_rel = 2.0
        
        coef = CP_FLOAT( sigma_T * self.k0_las**2 * v_rel * self.N_e * self.N_l * (z_weight * dsx * dsy).get() / ( 2.0 * np.pi * sigma_thx * sigma_thy ) )
        self.intersection *= coef
        self.intersection = self.intersection.reshape((nx, ny))
        self.time_envelope *= coef * N_STEPS / ( t_end - t_start )
        
        ts = cp.linspace(t_start, t_end, N_STEPS, dtype=CP_FLOAT)

        self.env_ts = ts / self.omega_las
        self.particles = particles
        return f_th.flatten()
    
    def calculate_total(self):
        if self.intersection is None:
            self.calculate_intersection()
        return self.intersection.sum().get() * self.dtheta_x * self.dtheta_y 

    def calculate_spectrum(self, s, gamma_0, sigma_gamma_0, gamma_num = 128, emulate_nonlinearity = True):
        if emulate_nonlinearity:
            gamma = gamma_0 / np.sqrt(1.0 + self.a0**2/8)
            sigma_gamma = np.sqrt(sigma_gamma_0**2 + gamma**2 * self.a0**4/16)
        else:
            gamma = gamma_0
            sigma_gamma = sigma_gamma_0
        
        gs = gamma + cp.linspace(-3.0 * sigma_gamma, 3.0 * sigma_gamma, gamma_num)[cp.newaxis, :]
        dg = gs[0, 1] - gs[0, 0]

        y = s[:, cp.newaxis] / gs**2
        spec = 1.5 * ( 1.0 - 2.0 * y * ( 1.0 - y ) )
        spec = cp.where(cp.logical_or(y < 0, y > 1), 0.0, spec)
        spec *= cp.exp(- (gs - gamma)**2 / 2.0 / sigma_gamma**2) / cp.sqrt(2.0 * cp.pi * sigma_gamma**2) / gs**2
        return (spec.sum(axis=1) * dg * self.calculate_total() / ( 4.0 * self.Wph )).get()
    
    def calculate_angular_spectrum(self, s, theta_x, theta_y, gamma_0, sigma_gamma_0, phi_pol, weight_threshold = 0.05, samples_per_point = 32, debug_idx = 0, emulate_nonlinearity = True):
        if self.intersection is None:
            self.calculate_intersection()

        coef = 3.0 / ( 4.0 * cp.pi**4 * self.Wph * 4.0)
        
        params = cp.stack(cp.meshgrid(theta_x, theta_y, s, indexing='ij'), 3).reshape(-1, 3).astype(CP_FLOAT)
        grid_x = theta_x.size * theta_y.size * s.size

        dx = CP_FLOAT(3.0 * self.emit_x / self.sigma_ex)
        dy = CP_FLOAT(3.0 * self.emit_y / self.sigma_ey)

        debug = cp.zeros((MAX_ARCS, SAMPLES_TOTAL * samples_per_point, 3), dtype=CP_FLOAT) * cp.nan

        if emulate_nonlinearity:
            gamma = gamma_0 / np.sqrt(1.0 + self.a0**2/8)
            sigma_gamma = np.sqrt(sigma_gamma_0**2 + gamma**2 * self.a0**4/16)
        else:
            gamma = gamma_0
            sigma_gamma = sigma_gamma_0

        spec = cp.zeros((grid_x,), dtype=CP_FLOAT)
        start  = cp.cuda.Event()
        # mid  = cp.cuda.Event()
        finish = cp.cuda.Event()
        start.record()
        # arcs_kernel[grid_x, X_THREADS](arcs, n_arcs, xy, dx, dy)
        # mid.record()
        # mid.synchronize()
        spectrum_kernel[grid_x, X_THREADS](spec, params, self.intersection, CP_FLOAT(sigma_gamma), CP_FLOAT(gamma), dx, dy, CP_FLOAT(phi_pol), CP_UINT(samples_per_point), debug, debug, debug, debug, debug, CP_UINT(debug_idx))
        finish.record()
        finish.synchronize()
        dt = cp.cuda.get_elapsed_time(start, finish) * 1e-3
        
        return (coef*spec).reshape((theta_x.size, theta_y.size, s.size)).get(), dt, debug#, arcs, n_arcs, debug