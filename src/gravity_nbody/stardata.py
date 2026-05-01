import numpy as np


ALPHA_CEN_A = (219.90205833170774, -60.83399268831004, 742.12)
ALPHA_CEN_B = (219.89609628987276, -60.83752756558407, 742.12)
PROXIMA_CEN = (217.42894222160578, -62.67949018907555, 768.0665)
DEGREES_TO_RADIANS = np.pi / 180
DEFAULT_STARS = ["Alpha Cen A", "Alpha Cen B", "Proxima Cen"]
SOLAR_MASS_KG = 1.98847e30


def pcalc1():
    from astroquery.simbad import Simbad

    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('plx')
    return custom_simbad.query_objects(DEFAULT_STARS)


def result_crtsn(m, n, r):
    plxarc = r / 1000
    d = (1 / plxarc) * 3.086e16  # in m

    x = d * (np.cos(DEGREES_TO_RADIANS * n)) * np.cos(DEGREES_TO_RADIANS * m)
    y = d * (np.cos(DEGREES_TO_RADIANS * n)) * np.sin(DEGREES_TO_RADIANS * m)
    z = d * np.sin(DEGREES_TO_RADIANS * n)

    return (x, y, z)


def pcalc2():
    pos_a = result_crtsn(ALPHA_CEN_A[0], ALPHA_CEN_A[1], ALPHA_CEN_A[2])
    pos_b = result_crtsn(ALPHA_CEN_B[0], ALPHA_CEN_B[1], ALPHA_CEN_B[2])
    pos_proxima = result_crtsn(PROXIMA_CEN[0], PROXIMA_CEN[1], PROXIMA_CEN[2])

    return np.array([pos_a, pos_b, pos_proxima])


def get_velocity_arrays(stars=None):
    from astroquery.simbad import Simbad

    if stars is None:
        stars = DEFAULT_STARS

    custom = Simbad()
    custom.add_votable_fields('ra', 'dec', 'plx_value', 'pmra', 'pmdec', 'rvz_radvel')
    res = custom.query_objects(stars)
    if res is None:
        raise ValueError(f"No SIMBAD results found for: {stars}")
    res.rename_columns(res.colnames, [c.lower() for c in res.colnames])

    # Conversion constant: mas/yr * pc -> km/s
    k = 4.74047

    def compute_velocity_array(row):
        ra = np.deg2rad(row['ra'])
        dec = np.deg2rad(row['dec'])
        plx = row['plx_value']  # in mas
        pmra = (row['pmra'] if row['pmra'] is not None else 0.0) / 1000
        pmdec = (row['pmdec'] if row['pmdec'] is not None else 0.0) / 1000
        rv = row['rvz_radvel'] if row['rvz_radvel'] is not None else 0.0

        if plx == 0 or plx is None:
            raise ValueError(f"Parallax missing for {row['main_id']}")

        d = 1 / (plx * 1e-3)

        cosd, sind, cosa, sina = np.cos(dec), np.sin(dec), np.cos(ra), np.sin(ra)
        e_r = np.array([cosd * cosa, cosd * sina, sind])
        e_alpha = np.array([-sina, cosa, 0.0])
        e_delta = np.array([-sind * cosa, -sind * sina, cosd])

        v_tan = k * d * (pmra * e_alpha + pmdec * e_delta)
        v_rad = rv * e_r
        return v_tan + v_rad

    velocity_arrays = []
    for row in res:
        try:
            velocity_arrays.append(compute_velocity_array(row))
        except Exception as exc:
            print(f"Skipping {row['main_id']}: {exc}")

    return np.array(velocity_arrays) * 1000  # shape: (N_stars, 3), in m/s


def v_relative():
    masses = np.array([1.1 * SOLAR_MASS_KG, 0.907 * SOLAR_MASS_KG, 0.122 * SOLAR_MASS_KG])
    velocities = get_velocity_arrays()

    barycenter_velocity = np.sum(masses[:, np.newaxis] * velocities, axis=0) / np.sum(masses)
    return velocities - barycenter_velocity
