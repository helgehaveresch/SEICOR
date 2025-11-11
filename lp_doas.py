import pandas as pd
import os

def read_lpdoas(dir, date, filename=None, mode= "ppb"):
    """
    Reads LP-DOAS .dat file for a given date, skipping bad lines.

    Parameters
    ----------
    file_path : str
        The path to the LP-DOAS .dat file.
    date : str
        The date for which to read the data (format: 'YYMMDD').
    mode : str
        The mode for reading the data ('ppb' or 'SC') (default: 'ppb').

    Returns
    -------
    df : pd.DataFrame
        DataFrame with parsed LP-DOAS data.
    """
    if filename is None:
        filename = r"20{}_365_Eval_{}.dat".format(date, mode)
    file_path = os.path.join(dir, filename)
    df = pd.read_csv(
        file_path,
        sep='\t',
        engine='python',
        on_bad_lines='skip'  # skips lines with wrong number of columns
    )
    # Convert 'StartDateAndTime' to datetime and set timezone to UTC
    df['StartDateAndTime'] = pd.to_datetime(df['StartDateAndTime'])
    if df['StartDateAndTime'].dt.tz is None:
        df['StartDateAndTime'] = df['StartDateAndTime'].dt.tz_localize('UTC')
    else:
        df['StartDateAndTime'] = df['StartDateAndTime'].dt.tz_convert('UTC')
    df.set_index("StartDateAndTime", inplace=True)
    df.sort_index(inplace=True)
    return df

def mask_lp_doas_file(df_lp_doas, start_time, end_time, rms_threshold = 0.0005):
    """
    Masks the LP-DOAS DataFrame for a specific time range and RMS threshold.

    Args:
        df_lp_doas (pd.DataFrame): The LP-DOAS DataFrame to mask.
        start_time (pd.Timestamp): The start time for the mask.
        end_time (pd.Timestamp): The end time for the mask.

    Returns:
        pd.DataFrame: The masked LP-DOAS DataFrame.
    """
    mask = (df_lp_doas.index >= start_time) & (df_lp_doas.index <= end_time) & (df_lp_doas['RMS'] < rms_threshold)
    return df_lp_doas[mask]