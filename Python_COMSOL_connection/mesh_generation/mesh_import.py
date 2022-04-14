def load_meshes():

    """
    Runs Matlab code (having the access to COMSOL), which outputs geometries and saves to IRINA/Meshes directory.
    """

    import os
    import matlab.engine
    import pandas as pd

    eng = matlab.engine.connect_matlab(name='matlab')

    '''
    Run COMSOL with Matlab server as administrator.
    '''

    x_valid = pd.DataFrame()
    x_target = pd.DataFrame()

    x_all = x_valid + x_target
    del x_valid

    current_path = os.getcwd()
    upper_directory = '\\'.join(current_path.split('\\')[0:-2])

    x_all.to_csv(upper_directory + '\\Meshes\\all_structures.csv')
    x_target.to_csv(upper_directory + '\\Meshes\\target_structures.csv')


load_meshes()
