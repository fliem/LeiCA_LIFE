def calculate_FD_P(in_file):
    """
    Baed on CPAC function
    https://github.com/FCP-INDI/C-PAC/blob/master/CPAC/generate_motion_statistics/generate_motion_statistics.py#L549
    fixed translation/rotation order for FSL. see:
    https://github.com/FCP-INDI/C-PAC/commit/ebc0d9fc4e683691a0fdfb95489236e841be94a0

    Method to calculate Framewise Displacement (FD) calculations
    (Power et al., 2012)
    
    Parameters
    ----------
    in_file : string
        movement parameters vector file path
    
    Returns
    -------
    FD_ts_file : string
        Frame-wise displacement mat 
        file path
    mean_FD_file : mean FD
    """
    '''
    '''

    import os
    import numpy as np

    FD_ts_file = os.path.join(os.getcwd(), 'FD_P_ts')
    mean_FD_file = os.path.join(os.getcwd(), 'FD_P_mean')
    max_FD_file = os.path.join(os.getcwd(), 'FD_P_max')

    lines = open(in_file, 'r').readlines()
    rows = [[float(x) for x in line.split()] for line in lines]
    cols = np.array([list(col) for col in zip(*rows)])
    
    translations = np.transpose(np.abs(np.diff(cols[3:6, :])))
    rotations = np.transpose(np.abs(np.diff(cols[0:3, :])))

    FD_power = np.sum(translations, axis = 1) + (50*3.141/180)*np.sum(rotations, axis =1)
    
    #FD is zero for the first time point
    FD_power = np.insert(FD_power, 0, 0)

    np.savetxt(FD_ts_file, FD_power)
    np.savetxt(mean_FD_file, np.mean(FD_power).reshape(1,1))
    np.savetxt(max_FD_file, np.max(FD_power).reshape(1,1))

    return FD_ts_file, mean_FD_file, max_FD_file


def calculate_FD_J(in_file):
    # get mean and max FD_Jenkinson from mcflirt's *rel.rms file
    import os
    import numpy as np
    FD_ts_file = os.path.join(os.getcwd(), 'FD_J_ts')
    mean_FD_file = os.path.join(os.getcwd(), 'FD_J_mean')
    max_FD_file = os.path.join(os.getcwd(), 'FD_J_max')
    ts = np.genfromtxt(in_file)

    np.savetxt(FD_ts_file, ts)
    np.savetxt(mean_FD_file, np.mean(ts).reshape(1,1))
    np.savetxt(max_FD_file, np.max(ts).reshape(1,1))
    return FD_ts_file, mean_FD_file, max_FD_file