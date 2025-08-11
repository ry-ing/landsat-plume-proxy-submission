import numpy as np
import xarray as xr
import pandas as pd

# Define the local location of Mankoff 2020 runoff dataset files
p2r = 'C:/Users/s1834371/OneDrive - University of Edinburgh/data/mankoff_runoff/'  # '~/OneDrive - University of Edinburgh/data/mankoff_runoff/'

def deg2rad(deg):
    return deg * (np.pi / 180.0)

def latlon2utm(phi, lambd):
    a = 6378137.0
    e = 0.08181919
    phi_c = 70.0
    lambda_0 = -45.0

    # Convert to radians
    phi = deg2rad(phi)
    phi_c = deg2rad(phi_c)
    lambd = deg2rad(lambd)
    lambda_0 = deg2rad(lambda_0)

    # Switching to South Hemisphere if necessary
    pm = 1
    if phi_c < 0:
        pm = -1
        phi = -phi
        phi_c = -phi_c
        lambd = -lambd
        lambda_0 = -lambda_0
    
    # Perform calculations
    t = np.tan(np.pi / 4 - phi / 2) / ((1 - e * np.sin(phi)) / (1 + e * np.sin(phi)))**(e / 2)
    t_c = np.tan(np.pi / 4 - phi_c / 2) / ((1 - e * np.sin(phi_c)) / (1 + e * np.sin(phi_c)))**(e / 2)
    m_c = np.cos(phi_c) / np.sqrt(1 - e**2 * (np.sin(phi_c))**2)
    rho = a * m_c * t / t_c
    
    x = pm * rho * np.sin(lambd - lambda_0)
    y = -pm * rho * np.cos(lambd - lambda_0)
    
    return x, y

def read_coordinates(p2r):
    with xr.open_dataset(f'{p2r}RACMO.nc') as ds:
        lat = ds['coast_lat'].values
        lon = ds['coast_lon'].values
        z = ds['coast_alt'].values
    return lat, lon, z

def read_runoff_data(p2r):
    try:
        with xr.open_dataset(f'{p2r}RACMO.nc', engine='netcdf4', chunks={'stations': 1000, 'time': 1000}) as ds_RACMO, xr.open_dataset(f'{p2r}MAR.nc', engine='netcdf4', chunks={'stations': 1000, 'time': 1000}) as ds_MAR:
            t0_racmo = pd.to_datetime(ds_RACMO['time'].values, unit='D')
            t0_mar = pd.to_datetime(ds_MAR['time'].values, unit='D')
            
            q0_RACMO = ds_RACMO['discharge'].values
            q0_MAR = ds_MAR['discharge'].values
    except Exception as e:
        print(f"Error reading data: {e}")
        return None, None, None, None
    return t0_racmo, t0_mar, q0_RACMO, q0_MAR

def remove_nans(lat, lon, z, x, y, q0_RACMO, q0_MAR):
    valid_inds = ~np.isnan(q0_MAR[:, 0])
    lat = lat[valid_inds]
    lon = lon[valid_inds]
    z = z[valid_inds]
    x = x[valid_inds]
    y = y[valid_inds]
    q0_RACMO = q0_RACMO[valid_inds, :]
    q0_MAR = q0_MAR[valid_inds, :]
    return lat, lon, z, x, y, q0_RACMO, q0_MAR

def combine_outlets(x, y, z, lat, lon, q_MAR, q_RACMO):
    i = 0
    while i < len(x):
        d = (x - x[i]) ** 2 + (y - y[i]) ** 2
        if np.sum(d == 0) > 1:
            inds = np.where(d == 0)[0]
            q_MAR[inds[0], :] = np.sum(q_MAR[inds, :], axis=0)
            q_RACMO[inds[0], :] = np.sum(q_RACMO[inds, :], axis=0)
            x = np.delete(x, inds[1:])
            y = np.delete(y, inds[1:])
            z = np.delete(z, inds[1:])
            lat = np.delete(lat, inds[1:])
            lon = np.delete(lon, inds[1:])
            q_MAR = np.delete(q_MAR, inds[1:], axis=0)
            q_RACMO = np.delete(q_RACMO, inds[1:], axis=0)
        i += 1
    return x, y, z, lat, lon, q_MAR, q_RACMO

def main():
    lat, lon, z = read_coordinates(p2r)
    x, y = latlon2utm(lat, lon)
    t0_racmo, t0_mar, q0_RACMO, q0_MAR = read_runoff_data(p2r)
    
    if t0_racmo is None or t0_mar is None:
        return
    
    lat, lon, z, x, y, q0_RACMO, q0_MAR = remove_nans(lat, lon, z, x, y, q0_RACMO, q0_MAR)
    
    x, y, z, lat, lon, q_MAR, q_RACMO = combine_outlets(x, y, z, lat, lon, q0_MAR, q0_RACMO)
    
    t = t0_racmo  # Assuming t0_racmo and t0_mar are the same
    q = 0.5 * (q_RACMO + q_MAR[:, :len(t)])
    
    runoff_ds = xr.Dataset(
        {
            'x': (['outlet'], x),
            'y': (['outlet'], y),
            'z': (['outlet'], z),
            'lat': (['outlet'], lat),
            'lon': (['outlet'], lon),
            't': (['time'], t),
            'q': (['outlet', 'time'], q),
            't_MAR': (['time_MAR'], t0_mar),
            'q_MAR': (['outlet', 'time_MAR'], q_MAR),
            't_RACMO': (['time_RACMO'], t0_racmo),
            'q_RACMO': (['outlet', 'time_RACMO'], q_RACMO),
        }
    )
    
    runoff_ds.to_netcdf('runoff.nc')

if __name__ == "__main__":
    main()