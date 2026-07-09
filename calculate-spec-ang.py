import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from xigma_i import Compton

import argparse
import tomllib

c = 2.99792458e+10
def main():
    parser = argparse.ArgumentParser(description="Calculate angular and spectral distribution of Compton photons")
    parser.add_argument(
        "-c", "--config",
        default="example-config.toml",
        help="Path to the TOML configuration file (default: example-config.toml)"
    )
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    e = config["electron"]
    l = config["laser"]

    # Electrons
    chargeNC = e["charge"]
    gamma_0 = e["gamma"]
    sigma_gamma_p = e["sigma_gamma_rel"]
    sigma_ex = e["sigma_x"]
    sigma_ey = e["sigma_y"]
    electron_duration = e["duration"]
    norm_emit_x = e["norm_emit_x"]
    norm_emit_y = e["norm_emit_y"]
    focus_x = e["focus_x"]
    focus_y = e["focus_y"]
    focus_z = e["focus_z"]

    emit_x = norm_emit_x / gamma_0
    emit_y = norm_emit_y / gamma_0
    sigma_ez = electron_duration * c

    # Laser
    w_l = l["energy"]
    lambda_l = l["lambda"]
    sigma_lr0 = l["sigma_r"]
    laser_duration = l["duration"]
    pol = l["polarization_angle"]
    sigma_lz = laser_duration * c

    compton = Compton()
    compton.set_electron_parameters(chargeNC = chargeNC, emit_x = emit_x, emit_y = emit_y, sigma_ex = sigma_ex, sigma_ey = sigma_ey, sigma_ez = sigma_ez)
    compton.set_laser_parameters(WL = w_l, lambda_l = lambda_l, sigma_lr0 = sigma_lr0, sigma_lz = sigma_lz)
    compton.set_foci_displacement(focus_x, focus_y, focus_z)
    compton.calculate_intersection(theta_num = 64)

    sigma_gamma = sigma_gamma_p * gamma_0

    thx = cp.linspace(0.0,1.0/gamma_0,127,dtype=cp.float32)
    th_zero = cp.array([0.0],dtype=cp.float32)
    ss = (cp.linspace(0.8,1.1,256)*gamma_0**2).astype(cp.float32)
    spec_kwargs = dict(gamma_0 = gamma_0, sigma_gamma_0 = sigma_gamma, phi_pol = pol*np.pi/180., samples_per_point = 32)
    spec_x, _, _ = compton.calculate_angular_spectrum(ss, thx, th_zero, **spec_kwargs)
    spec_y, _, _ = compton.calculate_angular_spectrum(ss, th_zero, thx, **spec_kwargs)
    ss_tot = (cp.linspace(0.0,1.1,512)*gamma_0**2).astype(cp.float32)
    spec_tot = compton.calculate_spectrum(ss_tot, gamma_0, sigma_gamma)
    print(f"Total yield: {compton.calculate_total():.2e} photons")

    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout="constrained")
    s_scale = 4 * compton.Wph
    
    ax=axes[0]
    ax.plot(ss_tot.get()*s_scale, spec_tot)
    ax.set_xlabel("E, MeV")
    ax.set_ylabel(r"dN/dE, MeV$^{-1}$")
    
    data_y = spec_y.sum(axis=0)*1e-6
    data_x = spec_x.sum(axis=1)*1e-6

    vmin = min(data_y.min(), data_x.min())
    vmax = max(data_y.max(), data_x.max())
    for i, spec in enumerate([data_y, data_x]):
        ax=axes[1+i]
        im=ax.imshow(spec,origin='lower', extent=[ss[0].get()*s_scale,ss[-1].get()*s_scale,thx[0].get()*1e3,thx[-1].get()*1e3],vmin=vmin,vmax=vmax)
        ax.set_aspect('auto')
        ax.set_xlabel("E, MeV")
        ax.set_ylabel(f"θ{'yx'[i]}, mrad")

    fig.colorbar(im, ax=[axes[1], axes[2]], label=r'dN/dEdΩ, MeV$^{-1}$μsr$^{-1}$')
    plt.show()


if __name__ == "__main__":
    main()