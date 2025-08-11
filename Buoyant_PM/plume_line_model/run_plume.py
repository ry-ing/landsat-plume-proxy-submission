import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid

def run_plume(zi, xi, Ta, Sa, Na, Q0, alpha, stop_on_zero_velocity=False):
    """
    ICE SHELF/TIDEWATER GLACIER PLUME MODEL
    FOR ARBITRARY ICE-OCEAN BOUNDARY GEOMETRY
    Donald Slater, uploaded as part of Slater 2022, GRL

    % USEFUL NOTES
    1. model assumes ocean surface at z=0 with z negative below surface
    2. model assumes water emerges at the minimum value in zi
    3. model assumes the ice-ocean boundary is oriented bottom left to top right
    4. model cannot cope with complex in-and-out geometries; i.e. the gradient of zi wrt xi cannot be negative anywhere
    5. model linearly interpolates the shape of the ice-ocean boundary and the ocean conditions between the supplied points

    Parameters:
    -----------
    zi : array-like
        Depth values (should be <= 0).
    xi : array-like
        Horizontal distance values corresponding to zi.
    Ta : array-like
        Ambient temperature values corresponding to zi.
    Sa : array-like
        Ambient salinity values corresponding to zi.
    Na : array-like
        Ambient nutrient concentration values corresponding to zi.
    Q0 : float
        Initial volume flux.
    alpha : float
        Entrainment coefficient.
    stop_on_zero_velocity : bool, optional
        If True, stop the simulation when the vertical velocity of the plume is equal to 0.

    Returns:
    --------
    solution : dict
        Dictionary containing the solution with keys:
        - 'z': Depth values (m).
        - 'b': Plume width (m).
        - 'u': Plume velocity (m/s).
        - 'T': Plume temperature (degC).
        - 'S': Plume salinity (psu).
        - 'N': Plume nutrient concentration (mmol/m^3).
        - 'mdot': Melt rate (m/d).
        - 'rho': Plume density (kg/m^3).
    """

    # Define parameters
    par = {
        'alpha': alpha,
        'g': 9.81,
        'rho0': 1020,
        'l1': -5.73e-2,
        'l2': 8.32e-2,
        'l3': 7.53e-4,
        'cw': 3974,
        'ci': 2009,
        'Lm': 334000,
        'Cd': 0.0097,
        'GT': 1.1e-2,
        'GS': 3.1e-4,
        'Ti': -10,
        'betaS': 7.86e-4,
        'betaT': 3.87e-5,
        'Gamma0': 1,
        'meltdragfeedback': 1,
        'EoS': 1,
    }

    # Initial input checks
    if np.any(zi > 0):
        print('WARNING: z-input should all be <=0')
        
    # Sort input arrays based on zi
    sort_ind = np.argsort(zi)
    zi = np.array(zi)[sort_ind]
    xi = np.array(xi)[sort_ind]
    Ta = np.array(Ta)[sort_ind]
    Sa = np.array(Sa)[sort_ind]
    Na = np.array(Na)[sort_ind]

    # Initial conditions
    T0 = par['l2'] + par['l3'] * min(zi)  # Initial temperature
    S0, N0 = 0, 0  # Initial salinity and nutrient concentration

    # Reduced gravity
    if par['EoS'] == 0:
        g0p = par['g'] * (par['betaS'] * (Sa[0] - S0) - par['betaT'] * (Ta[0] - T0))
    else:
        g0p = (par['g'] / par['rho0']) * (rho(Ta[0], Sa[0], 0) - rho(T0, S0, 0))

    b0 = (par['alpha'] * Q0**2 * par['Gamma0'] / g0p)**(1/3)
    u0 = Q0 / b0

    # Model solves in terms of fluxes, so need initial fluxes
    QFLUX0 = u0 * b0
    MFLUX0 = u0**2 * b0
    TFLUX0 = b0 * u0 * T0
    SFLUX0 = b0 * u0 * S0
    NFLUX0 = b0 * u0 * N0

    # Transform depth input into an along-ice variable
    tantheta = np.gradient(zi, xi)
    li = cumulative_trapezoid(np.sqrt(1 + 1.0 / tantheta**2), zi, initial=0)

    # Calculate sin(theta) from tan(theta)
    sintheta = np.zeros_like(tantheta)
    for ii in range(len(tantheta)):
        if tantheta[ii] == np.inf:
            sintheta[ii] = 1
        else:
            sintheta[ii] = tantheta[ii] / np.sqrt(1 + tantheta[ii]**2)

    def equations_line(l, A):
        b = A[0]**2 / A[1]
        u = A[1] / A[0]
        T = A[2] / A[0]
        S = A[3] / A[0]
        N = A[4] / A[0]

        if par['EoS'] == 0:
            gp = par['g'] * (par['betaS'] * (np.interp(l, li, Sa) - S) - par['betaT'] * (np.interp(l, li, Ta) - T))
        else:
            gp = (par['g'] / par['rho0']) * (rho(np.interp(l, li, Ta), np.interp(l, li, Sa), 0) - rho(T, S, 0))

        z = np.interp(l, li, zi)

        if par['meltdragfeedback'] == 1:
            quad1 = -par['l1'] * par['cw'] * par['GT'] + par['l1'] * par['ci'] * par['GS']
            quad2 = par['cw'] * par['GT'] * (T - par['l2'] - par['l3'] * z) + par['GS'] * (par['ci'] * (par['l2'] + par['l3'] * z - par['l1'] * S - par['Ti']) + par['Lm'])
            quad3 = -par['GS'] * S * (par['ci'] * (par['l2'] + par['l3'] * z - par['Ti']) + par['Lm'])
            Sb = (-quad2 + np.sqrt(quad2**2 - 4 * quad1 * quad3)) / (2 * quad1)
            Tb = par['l1'] * Sb + par['l2'] + par['l3'] * z
            mdot = par['cw'] * np.sqrt(par['Cd']) * par['GT'] * u * (T - Tb) / (par['Lm'] + par['ci'] * (Tb - par['Ti']))
            meltterm_vol = mdot
            meltterm_temp = mdot * Tb - np.sqrt(par['Cd']) * par['GT'] * u * (T - Tb)
            meltterm_sal = mdot * Sb - np.sqrt(par['Cd']) * par['GS'] * u * (S - Sb)
            dragterm = -par['Cd'] * u**2
        else:
            mdot = Tb = Sb = meltterm_vol = meltterm_temp = meltterm_sal = dragterm = 0

        E = par['alpha'] * np.interp(l, li, sintheta)

        return [
            E * u + meltterm_vol,
            b * gp * np.interp(l, li, sintheta) + dragterm,
            E * u * np.interp(l, li, Ta) + meltterm_temp,
            E * u * np.interp(l, li, Sa) + meltterm_sal,
            E * u * np.interp(l, li, Na)
        ]

    if stop_on_zero_velocity: # Stop the simulation when the vertical velocity of the plume is equal to 0
        def event(l, A):
            u = A[1] / A[0]
            return u
    else: # Stop the simulation when the plume width reaches 500 m
        def event(l, A):
            plumewidth = A[0]**2 / A[1]
            b = plumewidth - 500
            return b
    
    event.terminal = True
    event.direction = 0
    
    num_intervals = 1000  # Increase for more refined output (length of output)
    t_eval = np.linspace(li[0], li[-1], num_intervals)

    sol = solve_ivp(
        equations_line, 
        [li[0], li[-1]], 
        [QFLUX0, MFLUX0, TFLUX0, SFLUX0, NFLUX0], 
        method='RK23',  # RK23 originally
        t_eval=t_eval,
        rtol=1e-5, 
        atol=1e-10, 
        events=event
    )

    solution = {key: [] for key in ['z', 'b', 'u', 'T', 'S', 'N', 'Sb', 'Tb', 'mdot', 'rho', 'Ta', 'Sa', 'Na', 'rhoa']}
    
    for i in range(len(sol.t)):
        l = sol.t[i]
        A = sol.y[:, i]
        z = np.interp(l, li, zi)
        b = A[0]**2 / A[1]
        u = A[1] / A[0]
        T = A[2] / A[0]
        S = A[3] / A[0]
        N = A[4] / A[0]

        quad1 = -par['l1'] * par['cw'] * par['GT'] + par['l1'] * par['ci'] * par['GS']
        quad2 = par['cw'] * par['GT'] * (T - par['l2'] - par['l3'] * z) + par['GS'] * (par['ci'] * (par['l2'] + par['l3'] * z - par['l1'] * S - par['Ti']) + par['Lm'])
        quad3 = -par['GS'] * S * (par['ci'] * (par['l2'] + par['l3'] * z - par['Ti']) + par['Lm'])
        Sb = (-quad2 + np.sqrt(quad2**2 - 4 * quad1 * quad3)) / (2 * quad1)
        Tb = par['l1'] * Sb + par['l2'] + par['l3'] * z
        mdot = 86400 * par['cw'] * np.sqrt(par['Cd']) * par['GT'] * u * (T - Tb) / (par['Lm'] + par['ci'] * (Tb - par['Ti']))

        # Append values to the solution dictionary
        solution['z'].append(z)
        solution['b'].append(b)
        solution['u'].append(u)
        solution['T'].append(T)
        solution['S'].append(S)
        solution['N'].append(N)
        solution['Sb'].append(Sb)
        solution['Tb'].append(Tb)
        solution['mdot'].append(mdot)
        solution['rho'].append(rho(T, S, z))
        solution['Ta'].append(np.interp(z, zi, Ta))
        solution['Sa'].append(np.interp(z, zi, Sa))
        solution['Na'].append(np.interp(z, zi, Na))
        solution['rhoa'].append(rho(np.interp(z, zi, Ta), np.interp(z, zi, Sa), z)) 
     
        idx_nb = np.max(np.where(np.array(solution['rho']) < np.array(solution['rhoa'])))
        if idx_nb == -1:
            idx_nb = 0

        # Update the solution dictionary with neutral buoyancy level parameters
        solution.update({
            'rhoNB': solution['rho'][idx_nb],
            'rhoaNB': solution['rhoa'][idx_nb],
            'zNB': solution['z'][idx_nb],
            'TNB': solution['T'][idx_nb],
            'SNB': solution['S'][idx_nb],
            'NNB': solution['N'][idx_nb],
            'QNB': solution['b'][idx_nb] * solution['u'][idx_nb],
            'HNB': par['cw'] * solution['rho'][idx_nb] * (solution['b'][idx_nb] * solution['u'][idx_nb]) * (solution['T'][idx_nb] - (-2))
        })

    return solution

def rho(t, S, depth):
    rho_0 = 1027
    g = 9.81
    P = rho_0 * g * np.abs(depth) * 1e-5  # Convert depth to pressure in bars

    # Secant Bulk Modulus
    kw = 19652.21 + 148.4206 * t - 2.327105 * t**2 + 1.360477e-2 * t**3 - 5.155288e-5 * t**4
    Aw = 3.239908 + 1.43713e-3 * t + 1.16092e-4 * t**2 - 5.77905e-7 * t**3
    Bw = 8.50935e-5 - 6.12293e-6 * t + 5.2787e-8 * t**2

    k0 = kw + (54.6746 - 0.603459 * t + 1.09987e-2 * t**2 - 6.1670e-5 * t**3) * S + (7.944e-2 + 1.6483e-2 * t - 5.3009e-4 * t**2) * S**1.5
    A = Aw + (2.2838e-3 - 1.0981e-5 * t - 1.6078e-6 * t**2) * S + 1.91075e-4 * S**1.5
    B = Bw + (-9.9348e-7 + 2.0816e-8 * t + 9.1697e-10 * t**2) * S
    bulk_modulus = k0 + (A * P) + (B * P**2)

    # One Atmosphere International Equation of State [1980]
    A = 8.24493e-1 - 4.0899e-3 * t + 7.6438e-5 * t**2 - 8.2467e-7 * t**3 + 5.3875e-9 * t**4
    B = -5.72466e-3 + 1.0227e-4 * t - 1.6546e-6 * t**2
    C = 4.8314e-4

    rho_w = 999.842594 + 6.793952e-2 * t - 9.095290e-3 * t**2 + 1.001685e-4 * t**3 - 1.120083e-6 * t**4 + 6.536336e-9 * t**5
    rho_zero = rho_w + A * S + B * S**1.5 + C * S**2

    density_seawater = rho_zero / (1 - (P / bulk_modulus))

    return density_seawater
