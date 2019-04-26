import numpy as np

mesh = {}
mesh['next_point_index'] = 0
mesh['next_line_index'] = 0
mesh['next_line_loop_index'] = 0
mesh['next_surface_index'] = 0
mesh['points'] = []
mesh['lines'] = []
mesh['line_loops'] = []
mesh['surfaces'] = []

def generate_region(mesh, filename):
    point_index_start = mesh['next_point_index']
    line_index_start = mesh['next_line_index']
    points = np.loadtxt(filename, delimiter=',')
    points[:, 0] *= -1

    # add points
    for point in points:
        mesh['next_point_index'] += 1
        mesh['points'].append(point)
    point_index_end = mesh['next_point_index'] - 1

    # add lines
    for i in range(point_index_start, point_index_end):
        mesh['next_line_index'] += 1
        mesh['lines'].append((i, i + 1))
    mesh['next_line_index'] += 1
    mesh['lines'].append((point_index_end, point_index_start))
    line_index_end = mesh['next_line_index'] - 1

    # add line loop
    lines = list(range(line_index_start, line_index_end + 1))
    mesh['line_loops'].append(lines)
    mesh['next_line_loop_index'] += 1

generate_region(mesh, 'A.txt')
generate_region(mesh, 'A_hole.txt')
generate_region(mesh, 'T.txt')
generate_region(mesh, 'M.txt')
generate_region(mesh, 'outside.txt')

out = ''
points = mesh['points']
lines = mesh['lines']
line_loops = mesh['line_loops']
surfaces = mesh['surfaces']

# store points
for i in range(len(points)):
    out += 'Point({}) = {{{}, {}, 0, 3}};\n'.format(i, points[i][0], points[i][1])
out += '\n'

# store lines
for i in range(len(lines)):
    out += 'Line({}) = {{{}, {}}};\n'.format(i, lines[i][0], lines[i][1])

# store line loops
out += '\n'
for i in range(len(line_loops)):
    out += 'Line Loop({}) = {{'.format(i)
    for line in line_loops[i]:
        out += '{}, '.format(line)
    out = out[:-2] + '};\n'
out += '\n'

# store surfaces
out += 'Plane Surface(0) = {0, -1};\n'        # block A
out += 'Plane Surface(1) = {-1};\n'            # inside A
out += 'Plane Surface(2) = {2};\n'            # block T
out += 'Plane Surface(3) = {-3};\n'            # block M
out += 'Plane Surface(4) = {0, 2, 3, -4};\n'   # outside
out += '\n'

# store physical surfaces
out += 'Physical Surface(0) = {0, 2, 3};\n'   # block
out += 'Physical Surface(1) = {1, 4};\n'      # outside
out += '\n'

# store physical lines
last = mesh['next_line_index'] - 1
out += 'Physical Line(0) = {{{}, {}, {}, {}}};\n'.format(last - 3, last - 2, last - 1, last)
out += '\n'

out += 'Mesh.Algorithm = 8;\n'
out += 'Mesh.RecombineAll = 1;\n'
out += 'Mesh.SubdivisionAlgorithm = 1;\n'
out += 'Mesh.Smoothing = 20;\n'
with open('atm.geo', 'w') as file:
    file.write(out)
