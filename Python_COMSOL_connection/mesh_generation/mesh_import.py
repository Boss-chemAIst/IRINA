def load_meshes():

    """
    Runs Matlab code (having the access to COMSOL), which outputs geometries and saves to IRINA/Meshes directory.
    """

    import os
    from oct2py import Oct2Py
    oc = Oct2Py()

    x_valid = oc.run('valid_mesh_generator.m')
    x_target = oc.run('target_mesh_generator.m')

    x_all = x_valid + x_target
    del x_valid

    current_path = os.getcwd()
    upper_directory = '\\'.join(current_path.split('\\')[0:-2])

    x_all.to_csv(upper_directory + '\\Meshes\\all_structures.csv')
    x_target.to_csv(upper_directory + '\\Meshes\\target_structures.csv')
