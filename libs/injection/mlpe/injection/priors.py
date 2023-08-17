import numpy as np
from bilby.core.prior import Constraint, Cosine, LogUniform, PriorDict, Uniform
from bilby.gw.prior import UniformSourceFrame


def sg_uniform():
    prior_dict = PriorDict()
    prior_dict["quality"] = Uniform(
        name="quality", minimum=2, maximum=100, latex_label="quality"
    )
    prior_dict["frequency"] = Uniform(
        name="frequency", minimum=32, maximum=1024, latex_label="frequency"
    )
    prior_dict["dec"] = Cosine(name="dec", latex_label="dec")

    prior_dict["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, latex_label="ra"
    )
    prior_dict["hrss"] = LogUniform(
        name="hrss", minimum=1e-23, maximum=5e-20, latex_label="hrss"
    )

    prior_dict["eccentricity"] = Uniform(
        name="eccentricity", minimum=0, maximum=1, latex_label="eccentricity"
    )
    prior_dict["psi"] = Uniform(
        name="psi", minimum=0, maximum=np.pi, latex_label="psi"
    )
    prior_dict["phase"] = Uniform(
        name="phase", minimum=0, maximum=2 * np.pi, latex_label="phase"
    )

    return prior_dict


def nonspin_bbh():

    prior_dict = PriorDict()
    prior_dict["mass_1"] = Uniform(
        name="mass_1", minimum=5, maximum=100, unit=r"$M_{\odot}$"
    )
    prior_dict["mass_2"] = Uniform(
        name="mass_2", minimum=5, maximum=100, unit=r"$M_{\odot}$"
    )
    prior_dict["mass_ratio"] = Constraint(
        name="mass_ratio", minimum=0.2, maximum=5.0
    )
    prior_dict["luminosity_distance"] = UniformSourceFrame(
        name="luminosity_distance", minimum=100, maximum=3000, unit="Mpc"
    )
    prior_dict["dec"] = Cosine(name="dec")
    prior_dict["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )
    prior_dict["theta_jn"] = 0
    prior_dict["psi"] = 0
    prior_dict["phase"] = 0
    prior_dict["a_1"] = 0
    prior_dict["a_2"] = 0
    prior_dict["tilt_1"] = 0
    prior_dict["tilt_2"] = 0
    prior_dict["phi_12"] = 0
    prior_dict["phi_jl"] = 0

    return prior_dict
