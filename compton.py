import numpy as np
import cupy as cp
from cupyx import jit
from cupyx.scipy.special import erf as cp_erf
from cupyx.scipy.special import erfinv as cp_erfinv

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
N_ARCS = X_THREADS
POINTS_RESOLUTION = 512
POINTS_REPEAT = POINTS_RESOLUTION // X_THREADS
IDX_STRIDE = 3
INVAL = CP_FLOAT(9999.)
R_NUDGE = 32

GAUSS_WIDTH   = CP_FLOAT(3)
LORENTZ_WIDTH = CP_FLOAT(8)

N_STEPS = CP_UINT(X_THREADS)

@jit.rawkernel()
def particle_kernel(intersect, envelope, particles, THX, THY, f_th, w0, beta_ff, sigma_lz, t0, t1, dvx, dvy):#, debug, xy_debug): 
    # Make sure arguments and arrays are of type CP_FLOAT
    z_rayleigh = w0 * w0 * ( CP_ONE + beta_ff ) / 2
    
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
def arcs_kernel(arcs_Arr, n_arcs, xy_Arr, dx, dy):
    # This kernel populates *arcs* array of shape (M, N_ARCS, 3) for flattened array of M pairs (x0, y0)
    # It is then used to calculate cdf based on the energy distribution and calculate angular spectrum
    # Each arc is processed concurrently. If arc doesn't intersect with rectangle, all values are equal to INVAL
    # dx and dy are *HALF* the width and height of the rectangle
    # n_arcs is a M element in-out array. The kernel sets it to the actual number of arcs (may be smaller than N_ARCS) of givern (x0, y0)
    thread_idx = jit.threadIdx.x
    xy_idx     = jit.blockIdx.x

    rs_Arr = jit.shared_memory(CP_FLOAT, 4*4)

    x0 = xy_Arr[xy_idx, 0]
    y0 = xy_Arr[xy_idx, 1]

    areas = jit.shared_memory(CP_FLOAT, 4) # areas of each quadrant

    if thread_idx // 4 == 0: # First 4 threads, one for each quadrant
        q_idx = thread_idx % 4
        # 3 | 2
        # --+--
        # 0 | 1
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
        
        # Calculating areas of each quadrant
        xm = dx - cos_sign * x0
        ym = dy - sin_sign * y0

        skip = xm < CP_ZERO or ym < CP_ZERO # The whole quarter circle is outside the rectangle
        if skip:
            areas[q_idx] = CP_ZERO
        else:
            areas[q_idx] = cp.sqrt(min( 2 * dx, xm ) * min( 2 * dy, ym ))
        
    jit.syncthreads()
    if thread_idx == 0:
        total_area = CP_ZERO
        for i in jit.range(4):
            total_area += areas[i]
        for i in jit.range(4):
            areas[i] /= total_area
        
    jit.syncthreads()
    if thread_idx // 4 == 0:
        inside = ( cp.abs(x0) < dx ) and ( cp.abs(y0) < dy )    
        r_max = cp.sqrt( xm**2 + ym**2 ) - cp.sqrt(dx**2 + dy**2) / R_NUDGE # distance from to (x0, y0) the corner of a quadrant
        dr = CP_ZERO
        n_rings_q = CP_UINT(0) # Total number of arcs in quadrant
        n_start_q = CP_UINT(0) # Index of the first arc in this quarant
        n_in = CP_UINT(0) # Number of arcs fully inside quadrant
        
        for _qi in jit.range(q_idx):
            n_start_q += CP_UINT( cp.floor( N_ARCS * areas[_qi] ) ) 
    
        if skip:
            r_min = -INVAL
        else:
            n_rings_q = CP_UINT( cp.floor( N_ARCS * areas[q_idx] ) )     
            if inside:
                r_min = min( xm, ym )
                n_in  = CP_UINT( cp.ceil( ( n_rings_q ) * r_min / r_max ) )
                n_out = n_rings_q - n_in
                dr = ( r_max - r_min ) / ( n_out )

            else:
                r_min = cp.sqrt( max( CP_ZERO, xm - 2 * dx )**2 + max( CP_ZERO, ym - 2 * dy )**2 ) # min distance to rectangle dx, dy
                dr = ( r_max - r_min ) / ( n_rings_q - 1 )
                r_min = r_min + dr / R_NUDGE
       
        jit.atomic_add(n_arcs, xy_idx, n_rings_q)
        rs_Arr[q_idx*4 + 0] = r_min
        rs_Arr[q_idx*4 + 1] = dr
        rs_Arr[q_idx*4 + 2] = CP_FLOAT(n_in)
        rs_Arr[q_idx*4 + 3] = CP_FLOAT(n_start_q)

    jit.syncthreads()

    # Reading from populated arrays
    if thread_idx < n_arcs[xy_idx]:
        q_idx = CP_UINT(0)
        r_idx = CP_UINT(0)
        for qt_idx in jit.range(4):
            n_start = CP_UINT(rs_Arr[qt_idx*4 + 3])
            if thread_idx >= n_start:
                q_idx = CP_UINT(qt_idx)
                r_idx = thread_idx - n_start

        r_min = rs_Arr[q_idx*4 + 0]
        dr    = rs_Arr[q_idx*4 + 1]
        n_in  = CP_UINT(rs_Arr[q_idx*4 + 2])

        r  = CP_ZERO # Radius of the current arc
        r1 = CP_ZERO # Radius of the "previous" arc
        
        phi_start = (CP_PI * q_idx - CP_TWO_PI) / 2 # Starting angle of each quadrant. Since q_idx is unsigned (q_idx - 2) would overflow to UINT_MAX for q_idx = 0 or 1, so we expand the brackets
        phi_end = phi_start + CP_PI / 2
        
        if r_idx < n_in: # Fully inside quadrant
            _dr = r_min / n_in
            r = ( r_idx + 1 ) * _dr
            r1 = r_idx * _dr

            arcs_Arr[xy_idx, thread_idx, 2] = phi_start
            arcs_Arr[xy_idx, thread_idx, 3] = phi_end

        else:
            r = r_min + (r_idx + 1 - n_in) * dr
            r1 = max(CP_ZERO, r - dr)

            sin_pos = CP_UINT((q_idx // 2))
            # 1 | 1
            # --+--
            # 0 | 0
            cos_pos = CP_UINT(((q_idx+1)//2)%2)
            # 0 | 1
            # --+--
            # 0 | 1

            sin_sign = CP_INT(2 * sin_pos - 1)
            # + +
            # - -
            cos_sign = CP_INT(2 * cos_pos - 1)
            # - +
            # - +

            phi_cur = jit.shared_memory(CP_FLOAT, X_THREADS * 2) # 2 variables for each thread, indicating start and end of the current arc
            phi_cur[2*jit.threadIdx.x + 0] = -INVAL
            phi_cur[2*jit.threadIdx.x + 1] =  INVAL

            cos_0 = (  dx - x0 ) / r
            sin_0 = cp.sqrt(CP_ONE - cos_0**2)
            
            cos_1 = (- dx - x0 ) / r
            sin_1 = cp.sqrt(CP_ONE - cos_1**2)

            sin_2 = (  dy - y0 ) / r
            cos_2 = cp.sqrt(CP_ONE - sin_2**2)

            sin_3 = (- dy - y0 ) / r
            cos_3 = cp.sqrt(CP_ONE - sin_3**2)

            if cos_sign * cos_0 > 0 and cp.abs(y0 + r * sin_0 * sin_sign) < dy: # Right edge
                phi_cur[2*jit.threadIdx.x + (1 - sin_pos)] = cp.arctan2(sin_0 * sin_sign, cos_0)

            if cos_sign * cos_1 > 0 and cp.abs(y0 + r * sin_1 * sin_sign) < dy: # Left edge
                phi_cur[2*jit.threadIdx.x + (sin_pos)] = cp.arctan2(sin_1 * sin_sign, cos_1)

            if sin_sign * sin_2 > 0 and cp.abs(x0 + r * cos_2 * cos_sign) < dx: # Top edge
                phi_cur[2*jit.threadIdx.x + (cos_pos)] = cp.arctan2(sin_2, cos_2 * cos_sign)

            if sin_sign * sin_3 > 0 and cp.abs(x0 + r * cos_3 * cos_sign) < dx: # Bottom edge
                phi_cur[2*jit.threadIdx.x + (1 - cos_pos)] = cp.arctan2(sin_3, cos_3 * cos_sign)
                
            arcs_Arr[xy_idx, thread_idx, 2] = max(phi_start, phi_cur[2*jit.threadIdx.x + 0])
            arcs_Arr[xy_idx, thread_idx, 3] = min(phi_end,   phi_cur[2*jit.threadIdx.x + 1])
        
        arcs_Arr[xy_idx, thread_idx, 0] = r
        arcs_Arr[xy_idx, thread_idx, 1] = r1

@jit.rawkernel()
def spectrum_kernel(output, params_Arr, arcs_Arr, n_arcs, collision, sigma_g, gamma0, dx, dy, phi_pol, weight_bias, samples_per_point):#, debug_arr, xy_debug):
    # Calculates number of photons with energies defined by s_Arr and direction x0, y0.
    # Requires arcs array, calculated by arcs_kernel
    thread_idx = jit.threadIdx.x
    xy_idx  = jit.blockIdx.x
    out_idx = jit.blockIdx.x * jit.gridDim.y + jit.blockIdx.y

    x0 = params_Arr[out_idx, 0]
    y0 = params_Arr[out_idx, 1]
    s  = params_Arr[out_idx, 2]

    weights = jit.shared_memory(CP_FLOAT, N_ARCS)
    if thread_idx < n_arcs[xy_idx]:
        # First we need to calculate statistical weight of each arc
        # Each thread processes one arc
        # We also calculate geometrical area of the arc
        area = CP_ZERO

        scale = CP_FLOAT(0.5) / cp.sqrt(CP_FLOAT(2.0)) / sigma_g
        weights[thread_idx] = CP_ZERO
        
        r       = arcs_Arr[xy_idx, thread_idx, 0]
        r1      = arcs_Arr[xy_idx, thread_idx, 1]
        phi_min = arcs_Arr[xy_idx, thread_idx, 2]
        phi_max = arcs_Arr[xy_idx, thread_idx, 3]

        if phi_max - phi_min < CP_TWO_PI:
            area = ( phi_max - phi_min ) * ( r**2 - r1**2 ) 

        g_sq_mx = CP_ONE / ( CP_ONE / s - r * r )
        if g_sq_mx > 0: # Heavyside function effectively
            g_mx = cp.sqrt(g_sq_mx)
            g_mn = CP_ONE / cp.sqrt( CP_ONE / s - r1*r1 ) # guaranteed to be positive and not infinite

            arg1 = ( g_mn - gamma0 ) * scale
            arg2 = ( g_mx - gamma0 ) * scale

            # argmax = 12.

            # if (arg1 > -argmax and arg2 > -argmax) and (arg1 < argmax and arg2 < argmax):
            cp1 = cp.tanh( CP_FLOAT(2.0) / cp.sqrt(CP_FLOAT(2.0)) * ( arg1 + CP_FLOAT(11./123.)*(arg1**3) ) )
            cp2 = cp.tanh( CP_FLOAT(2.0) / cp.sqrt(CP_FLOAT(2.0)) * ( arg2 + CP_FLOAT(11./123.)*(arg2**3) ) )
            weights[thread_idx] = ( cp2 - cp1 ) / ( g_mn**2 * cp.sqrt(r)**3 ) * area
            # weights[thread_idx] = area #cp.power(area, 2) #* cp.power( cp2 - cp1 , 1.0)

    jit.syncthreads()

    # Next on a first thread we basically tabulate inverse cdf of statistical weights to define how many points should be inside each arc
    # The number of elements in this table is POINTS_RESOLUTION. Each point correspond to samples_per_point samples
    # Each element contains 3 (IDX_STRIDE) values: index of an arc the point belongs to, total number of points in this arc, and an index of a first point belonging to that arc
    # The last two are needed to calculate relative index of a current point
    idxs = jit.shared_memory(CP_UINT, POINTS_RESOLUTION * IDX_STRIDE)
    n_points_real = jit.shared_memory(CP_UINT, 1)
    total_weight  = jit.shared_memory(CP_FLOAT, 1)
    if thread_idx == 0:
        total_weight[0] = CP_ZERO
        for i_arc in jit.range(n_arcs[xy_idx]):
            total_weight[0] += weights[i_arc]

    jit.syncthreads()
    if total_weight[0] > 0.0:
        if thread_idx == 0:
            i_pt = CP_UINT(0)
            r_next = CP_UINT(0)
            for k in jit.range(n_arcs[xy_idx]):
                s_add = CP_UINT( cp.floor( POINTS_RESOLUTION * ( weights[k] / total_weight[0] + weight_bias ) / ( CP_ONE + weight_bias * N_ARCS ) ) )
                r_next += s_add
                while i_pt < r_next and i_pt < POINTS_RESOLUTION:
                    idxs[IDX_STRIDE*i_pt + 0] = CP_UINT(k)
                    idxs[IDX_STRIDE*i_pt + 1] = CP_UINT(r_next - s_add)
                    idxs[IDX_STRIDE*i_pt + 2] = CP_UINT(s_add)
                    i_pt += CP_UINT(1)
                
                # if xy_idx == CP_UINT(xy_debug):
                #     if CP_UINT(k) < debug_arr.shape[1]:
                #         debug_arr[jit.blockIdx.y, k, 4] = s_add / POINTS_RESOLUTION
                #         # debug_arr[jit.blockIdx.y, k, 1] = weights[k]
                #         # debug_arr[jit.blockIdx.y, k, 2] = total_weight

            n_points_real[0] = CP_UINT(i_pt)

    jit.syncthreads()
    if total_weight[0] > 0.0:
        # We overwrite value in weights with actual area of the arc, which is needed to properly calculate integral
        if thread_idx < n_arcs[xy_idx]:
            weights[thread_idx] = area
            # if xy_idx == CP_UINT(xy_debug):
            #     if CP_UINT(thread_idx) < debug_arr.shape[1]:
            #         debug_arr[jit.blockIdx.y, thread_idx, 4] = area

        jit.syncthreads()

        # Now we evaluate the function at each point and sum them with account of both their statistical weight and geometrical weight
        f_tot = CP_ZERO
        for rep_i in jit.range(POINTS_REPEAT):
            for di in jit.range(samples_per_point): # For each point we take multiple samples
                # sample_idx is calculated in such a way so that neighbouring samples are calculated by neighbouring threads to maximize GPU occupancy
                sample_idx = CP_UINT(POINTS_RESOLUTION * di) + CP_UINT(X_THREADS * rep_i) + CP_UINT(thread_idx)
                if sample_idx < n_points_real[0] * samples_per_point:
                    idx_idx = sample_idx // samples_per_point # index of the cdf table of the current sample
                    
                    arc_idx = idxs[IDX_STRIDE*idx_idx+0]
                    n_start = idxs[IDX_STRIDE*idx_idx+1]
                    n_len   = idxs[IDX_STRIDE*idx_idx+2]

                    pt_idx_dup = sample_idx - n_start * samples_per_point # index of the sample in a set of samples corresponding to a single arc
                    reg = pt_idx_dup / n_len / samples_per_point # regularly distributed among all samples in the current arc from 0 to 1

                    theta_max = arcs_Arr[xy_idx, arc_idx, 0]
                    theta_min = arcs_Arr[xy_idx, arc_idx, 1]
                    theta_sq = theta_min**2 + reg * ( theta_max**2 - theta_min**2 ) # To achieve uniform distribution in a disk using Fibonacci algorithm, *squares* of radii should be distributed unformly, not the radii themselves!
                    
                    g_sq = CP_ONE / ( CP_ONE / s - theta_sq )
                    if g_sq >= 0: # Heavyside
                        fib = cp.remainder( pt_idx_dup * PHI, 1.0 )
                        phi_min   = arcs_Arr[xy_idx, arc_idx, 2]
                        phi_max   = arcs_Arr[xy_idx, arc_idx, 3]
                        phi = phi_min + fib * ( phi_max - phi_min )
                        
                        theta = cp.sqrt(theta_sq)
                        x = x0 + theta * cp.cos(phi)
                        y = y0 + theta * cp.sin(phi)

                        if (x > -dx and x < dx and y > -dy and y < dy): # occasionally a sample might be completely outside the collision rectangle
                            Xi = (collision.shape[0] - 1) * ( x + dx ) / dx / 2
                            Yj = (collision.shape[1] - 1) * ( y + dy ) / dy / 2
                            
                            xi = CP_UINT(cp.floor(Xi))
                            yj = CP_UINT(cp.floor(Yj))

                            Xi = cp.remainder(Xi, CP_ONE)
                            Yj = cp.remainder(Yj, CP_ONE)

                            # linear interpolation
                            col = collision[xi    , yj    ] * (CP_ONE - Yj) * (CP_ONE - Xi) \
                                + collision[xi + 1, yj    ] * (CP_ONE - Yj) * (Xi) \
                                + collision[xi    , yj + 1] * (Yj)          * (CP_ONE - Xi) \
                                + collision[xi + 1, yj + 1] * (Yj)          * (Xi) 
                        else:
                            col = CP_ZERO
                                
                        g = cp.sqrt(g_sq)
                        ffac = cp.exp(-(g - gamma0)**2 / 2 / sigma_g**2) / cp.sqrt(CP_TWO_PI*sigma_g**2)

                        cos_pol = cp.cos( phi_pol - phi )**2
                        a_fac = CP_ONE - 4 * s * s * cos_pol * theta_sq / g_sq
                        
                        f = ffac * col * a_fac * g

                        ds = weights[arc_idx] / n_len / samples_per_point # Area in theta_x theta_y plane "occupied" by each sample
                        # ds = CP_ONE
                        f_tot += f * ds
                        # if xy_idx == CP_UINT(xy_debug):
                        #     if CP_UINT(sample_idx) < debug_arr.shape[1]:
                        #         debug_arr[jit.blockIdx.y, sample_idx, 0] = x
                        #         debug_arr[jit.blockIdx.y, sample_idx, 1] = y
                        #         debug_arr[jit.blockIdx.y, sample_idx, 2] = ds
                        #         debug_arr[jit.blockIdx.y, sample_idx, 3] = f
        jit.atomic_add(output, out_idx, f_tot)

## Kernels warmup
_nx = 4
_ns = 4
_dx = 1.0
_x0s = cp.linspace(-0.9*_dx, 0.9*_dx, _nx, dtype=CP_FLOAT)
_y0s = cp.linspace(-0.9*_dx, 0.9*_dx, _nx, dtype=CP_FLOAT)
_g0 = 1000.0
_ss = cp.linspace(0.0, _g0**2, _ns, dtype=CP_FLOAT)

_output = cp.zeros((_nx*_nx*_ns,), dtype=CP_FLOAT)
_xy = cp.stack(cp.meshgrid(_x0s, _y0s), 2).reshape(-1, 2)
_params = cp.stack(cp.meshgrid(_x0s, _y0s, _ss), 3).reshape(-1, 3)
_arcs = cp.zeros((_nx*_nx, N_ARCS, 4), dtype=CP_FLOAT)
_narcs = cp.zeros((1,), dtype=CP_FLOAT)
# _rs = cp.zeros((_nx*_nx, 4, 3), dtype=CP_FLOAT)
_debug = cp.zeros((_ns, _nx*_nx*_ns, 5), dtype=CP_FLOAT)
_intersect = cp.zeros((_nx, _nx), dtype=CP_FLOAT)
_env = cp.zeros((N_STEPS,), dtype=CP_FLOAT)
_mean = cp.array([0.0, 0.0, 0.0], dtype=CP_FLOAT)
_cov = cp.zeros((3, 3), dtype=CP_FLOAT)
_cov[0, 0] = CP_ONE
_cov[1, 1] = CP_ONE
_cov[2, 2] = CP_ONE
# particle_kernel[(16,_nx*_nx), 16](_intersect.flatten(), _env, _particles, _THX.flatten(), _THY.flatten(), cp.ones_like(_THX).flatten(), 1.0, 0.0, 1.0, 2.0, 0.1, 0.0, 0.0)
# arcs_kernel[_nx*_nx, X_THREADS](_arcs, _narcs, _xy, CP_ONE, CP_ONE)
# spectrum_kernel[(_nx*_nx, _ns), X_THREADS](_output, _params, _arcs, _narcs, _intersect, 1.0, _g0, _dx, _dx, 0.0, 0.0, 1)#, _debug, 0)

class Compton:
    # Electron parameters
    chargeNC = None
    # sigma_gamma = None
    gamma_0 = None
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

    def set_electron_parameters(self, chargeNC, emit_x, emit_y, sigma_ex, sigma_ey, sigma_ez):
        self.chargeNC = chargeNC
        self.emit_x = emit_x
        self.emit_y = emit_y
        self.sigma_ex = sigma_ex
        self.sigma_ey = sigma_ey
        self.sigma_ez = sigma_ez
        self.N_e = self.chargeNC * 1e-9 / elC
        # try:
        #     self.sigma_l = np.sqrt(self.sigma_ez**2 + self.sigma_lz**2)
        # except:
        #     pass

        self.intersection = None

    def set_laser_parameters(self, WL, lambda_l, sigma_lr0, sigma_lz, beta_ff = 0.0):
        self.WL = WL * 1e7 # Энергия пучка в эргах
        self.lambda_l = lambda_l
        self.beta_ff = beta_ff
        self.sigma_lr0 = sigma_lr0 # / 2.
        self.sigma_lz = sigma_lz
        self.r_l = np.pi * sigma_lr0**2 / self.lambda_l
        self.omega_las = 2 * np.pi*c / self.lambda_l
        self.k0_las = self.omega_las / c
        Wph = hbar * self.omega_las # Энергия фотона в эргах
        self.Wph = Wph * 1e-6 / ( elC * 1e7 ) # Энергия фотона в МэВ
        self.N_l = self.WL / Wph
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

    def calculate_intersection(self, theta_num = 256, particles_amount = 4096, debug_idx = 0):

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
        zR = CP_FLOAT((self.k0_las * self.sigma_lr0)**2 * ( 1.0 + self.beta_ff ) / 2)
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

    def calculate_spectrum(self, s, gamma_0, sigma_gamma, gamma_num = 128):
        gs = gamma_0 + cp.linspace(-3.0 * sigma_gamma, 3.0 * sigma_gamma, gamma_num)[cp.newaxis, :]
        dg = gs[0, 1] - gs[0, 0]

        y = s[:, cp.newaxis] / gs**2
        spec = 1.5 * ( 1.0 - 2.0 * y * ( 1.0 - y ) )
        spec = cp.where(cp.logical_or(y < 0, y > 1), 0.0, spec)
        spec *= cp.exp(- (gs - gamma_0)**2 / 2.0 / sigma_gamma**2) / cp.sqrt(2.0 * cp.pi * sigma_gamma**2) / gs**2
        return (spec.sum(axis=1) * dg * self.calculate_total() / ( 4.0 * self.Wph )).get()
    
    def calculate_angular_spectrum(self, s, theta_x, theta_y, gamma_0, sigma_gamma, phi_pol, weight_bias = 0.05, samples_per_point = 32):#, debug_idx = 0):
        if self.intersection is None:
            self.calculate_intersection()

        coef = 3.0 / ( 2.0 * cp.pi * self.Wph * 4)
        
        xy = cp.stack(cp.meshgrid(theta_x, theta_y, indexing='ij'), 2).reshape(-1, 2).astype(CP_FLOAT)
        params = cp.stack(cp.meshgrid(theta_x, theta_y, s, indexing='ij'), 3).reshape(-1, 3).astype(CP_FLOAT)
        grid_x = theta_x.size * theta_y.size
        arcs = cp.zeros((grid_x, N_ARCS, 4), dtype=CP_FLOAT)
        n_arcs = cp.zeros((grid_x, ), dtype=CP_UINT)

        grid_y = s.size
        dx = CP_FLOAT(3.0 * self.emit_x / self.sigma_ex)
        dy = CP_FLOAT(3.0 * self.emit_y / self.sigma_ey)

        # debug = cp.zeros((grid_y, POINTS_RESOLUTION * samples_per_point, 5), dtype=CP_FLOAT) * cp.nan
        spec = cp.zeros((grid_x * grid_y,), dtype=CP_FLOAT)
        start  = cp.cuda.Event()
        mid  = cp.cuda.Event()
        finish = cp.cuda.Event()
        start.record()
        arcs_kernel[grid_x, X_THREADS](arcs, n_arcs, xy, dx, dy)
        mid.record()
        mid.synchronize()
        spectrum_kernel[(grid_x, grid_y, 1), X_THREADS](spec, params, arcs, n_arcs, self.intersection, CP_FLOAT(sigma_gamma), CP_FLOAT(gamma_0), dx, dy, CP_FLOAT(phi_pol), CP_FLOAT(weight_bias), CP_UINT(samples_per_point))#, debug, debug_idx)
        finish.record()
        finish.synchronize()
        dt = cp.cuda.get_elapsed_time(start, finish) * 1e-3
        
        return (coef*spec).reshape((theta_x.size, theta_y.size, s.size)).get(), dt, arcs, n_arcs#, debug