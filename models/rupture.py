import numpy as np

default_parameters = {
    "mw_power": 1.5,
    "mw_scale": 9.1,
    "a1": 3.0,
    "b1": 6.1,
    "a2": 2.5,
    "hinge_length": 5500.0,
}


def rupture_length(magnitude, pars, par_id=None):
    """
    Step #1 in Bourne et al. [2019].

    Get the expected mean rupture length (log10-scale) for a given magnitude, based on Leonard (2010, 2012) global
    scaling relation between magnitude and rupture length.

    Leonard, M.: Earthquake Fault Scaling: Self-Consistent Relating of Rupture Length, Width, Average Displacement, and
    Moment Release. Bulletin of the Seismological Society of America 100(5A), 1971{1988 (2010). DOI 10.1785/0120090189.
    URL https://www.bssaonline.org/cgi/doi/10.1785/0120090189
    Leonard, M.: Erratum to Earthquake Fault Scaling: Self-Consistent Relating of Rupture Length, Width, Average
    Displacement, and Moment Release. Bull. Seis. Soc. Am. 102(6), 2797 (2012). DOI10.1785/0120120249

    :param magnitude: magnitude of the earthquake
    :param pars: dictionary of parameters, structured array, or array that can be indexed by par_id
    :param par_id: list of parameter names, not needed if pars is a dictionary or structured array
    :return: mean rupture length in km
    """

    c = gen_dict_like(pars, par_id)
    p = c["mw_power"]  # 1.5
    s = c["mw_scale"]  # 9.1 (Kanamori) or 9.05 (Hanks-Kanamori)
    a1 = c["a1"]  # 3.0
    b1 = c["b1"]  # 6.1
    a2 = c["a2"]  # 2.5
    h = c["hinge_length"]  # 5500

    log10moment = p * magnitude + s
    log10moment_hinge = a1 * np.log10(h) + b1
    b2 = log10moment_hinge - (a2 / a1) * (log10moment_hinge - b1)

    rupture_length_km = 10 ** (
        np.where(
            log10moment > log10moment_hinge,
            (log10moment - b2) / a2,
            (log10moment - b1) / a1,
        )
        - 3.0  # to translate from m to km
    )

    return rupture_length_km


def rupture_distance(hypocentral_distance, hypocenter_depth, azimuth, offset):
    """
    Get horizontal distance to the rupture plane for given distance to the epicentre, a (relative) azimuth in degrees
    (angle relative to the fault strike, 0 is along strike, 90 is perpendicular), and an
    offset of the rupture plane relative to the epicentre. If the offset is 0, the rupture plane extends away from
    the observer. If the offset is positive, the rupture plane extends towards the observer. The maximum offset is
    equal to the rupture length (otherwise the hypocentre is not on the rupture plane).
    We assume vertical faults/ruptures and hypocenter at the top of the fault.

    :param hypocentral_distance:
    :param hypocenter_depth:
    :param azimuth: angle in degrees (clockwise angle from north)
    :param offset: the horizontal offset of the rupture plane relative to the hypocentre
    :return:
    """

    vertical_distance = hypocenter_depth
    azimuth_radians = np.pi * azimuth / 180.0

    # normal distance : distance to an infinite line along the rupture trace
    epicentral_distance = np.clip(
        np.sqrt(hypocentral_distance**2 - vertical_distance**2), 0.0, None
    )
    normal_distance = epicentral_distance * np.sin(azimuth_radians)

    # parallel distance: distance to the nearest point on the rupture trace
    # in the direction of the rupture (i.e. projection on the rupture trace)
    parallel_distance = epicentral_distance * np.cos(azimuth_radians) - offset
    parallel_distance = np.clip(parallel_distance, 0.0, None)

    # total distance requires pythagoras
    distance_to_rupture = np.sqrt(
        (normal_distance**2) + (parallel_distance**2) + (vertical_distance**2)
    )

    return distance_to_rupture


def gen_dict_like(c, p):
    if p is None:
        return c
    else:
        return {par: c[..., i] for i, par in enumerate(p)}
