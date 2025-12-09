import numpy as np
import pandas as pd


def compute_relative_wind(ship_speed, ship_course_deg, wind_speed, wind_dir_deg):
    """
    Compute relative (apparent) wind seen from a moving ship.

    All directions are in degrees using the meteorological convention (degrees clockwise from North, "from").
    - `wind_dir_deg` : wind direction (where the wind is coming from), meteorological convention.
    - `ship_course_deg` : ship heading/course (direction the ship is moving TOWARDS), degrees clockwise from North.

    Returns (rel_speed, rel_dir_from):
    - rel_speed : relative wind speed (same units as inputs)
    - rel_dir_from : relative wind direction (meteorological "from" degrees, 0..360)

    The function accepts scalars, numpy arrays, or pandas Series. If pandas Series are passed
    the returned values will be pandas Series with the same index.

    Vector math:
    - Convert wind (from) to ground-referenced vector (u_w, v_w) in east/north components:
        u_w = -wind_speed * sin(dir_rad)
        v_w = -wind_speed * cos(dir_rad)
      (This follows the meteorological convention where a wind from 0deg blows to the south.)
    - Ship velocity vector (u_s, v_s) for course (towards):
        u_s = ship_speed * sin(course_rad)
        v_s = ship_speed * cos(course_rad)
    - Relative wind (air relative to ship):
        u_rel = u_w - u_s
        v_rel = v_w - v_s
    - Relative speed = sqrt(u_rel^2 + v_rel^2)
    - Relative direction (meteorological "from") = atan2(-u_rel, -v_rel) in degrees, normalized to [0,360).

    Examples
    --------
    >>> compute_relative_wind(5.0, 90, 8.0, 270)
    (13.0, 270.0)  # approx — ship east 5, wind from west 8 -> apparent from west and stronger

    """

    # Work with numpy arrays internally
    is_pandas = False
    if isinstance(ship_speed, pd.Series) or isinstance(ship_course_deg, pd.Series) or isinstance(wind_speed, pd.Series) or isinstance(wind_dir_deg, pd.Series):
        is_pandas = True

    # Convert inputs to numpy arrays
    ship_speed_arr = np.asarray(ship_speed)
    ship_course_arr = np.asarray(ship_course_deg)
    wind_speed_arr = np.asarray(wind_speed)
    wind_dir_arr = np.asarray(wind_dir_deg)

    # Convert degrees to radians
    course_rad = np.deg2rad(ship_course_arr.astype(float))
    wind_dir_rad = np.deg2rad(wind_dir_arr.astype(float))

    # Wind (from -> to): u = -s*sin(dir), v = -s*cos(dir)
    u_w = -wind_speed_arr * np.sin(wind_dir_rad)
    v_w = -wind_speed_arr * np.cos(wind_dir_rad)

    # Ship velocity (towards): u = s*sin(course), v = s*cos(course)
    u_s = ship_speed_arr * np.sin(course_rad)
    v_s = ship_speed_arr * np.cos(course_rad)

    # Relative wind (air relative to ship)
    u_rel = u_w - u_s
    v_rel = v_w - v_s

    # Relative speed
    rel_speed = np.hypot(u_rel, v_rel)

    # Relative wind direction (meteorological FROM): atan2(-u_rel, -v_rel)
    rel_dir_rad = np.arctan2(-u_rel, -v_rel)
    rel_dir_deg = (np.degrees(rel_dir_rad) + 360.0) % 360.0

    if is_pandas:
        # attempt to preserve index of any pandas input (take first Series index found)
        for x in (ship_speed, ship_course_deg, wind_speed, wind_dir_deg):
            if isinstance(x, pd.Series):
                idx = x.index
                break
        rel_speed = pd.Series(rel_speed, index=idx)
        rel_dir_deg = pd.Series(rel_dir_deg, index=idx)

    return rel_speed, rel_dir_deg


if __name__ == "__main__":
    # Quick CLI test
    import doctest
    doctest.testmod()

    # Example: ship heading east (90 deg) at 5 m/s, wind from west (270 deg) at 8 m/s
    s, d = compute_relative_wind(5.0, 90, 8.0, 270)
    print(s, d)
