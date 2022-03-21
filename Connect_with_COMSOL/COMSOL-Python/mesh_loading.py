from oct2py import Oct2Py
oc = Oct2Py()

x_1 = oc.run('valid_mesh_generator.m')
x_2 = oc.run('target_mesh_generator.m')

X = x_1 + x_2
