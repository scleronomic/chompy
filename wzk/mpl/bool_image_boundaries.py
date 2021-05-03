import numpy as np

# Get boundaries (edges/faces) of boolean images


def clean_grid_line(x, n_dim=2, return_indices=False):
    """
    Remove unnecessary intermediate steps on straight lines to compress the representation.

    Input:
    x -> x -> x
              |
              x
              |
              x -> x -> x -> x

    Output
    x ------> x
              |
              |
              |
              x -----------> x


    """

    keep_idx = np.ones(x.shape[0], dtype=bool)
    for i in range(x.shape[0]-2):
        if n_dim-np.logical_and(x[i] == x[i+1], x[i] == x[i+2]).sum() <= 1:
            keep_idx[i+1] = False

    if return_indices:
        return x[keep_idx], keep_idx
    else:
        return x[keep_idx]


# 2D
def get_edges(img):
    """
    Get a list of all edges (where the value changes from 'True' to 'False') in the image.
    Only works for 2D, see 'get_all_boundary_faces()' for 3D.
    Return the list as indices of the image
    """
    shape = img.shape

    ij_edges = []
    ii, jj = np.nonzero(img)
    for i, j in zip(ii, jj):
        # North
        if j == shape[1]-1 or not img[i, j+1]:
            ij_edges.append(np.array([[i, j+1],
                                      [i+1, j+1]]))

        # East
        if i == shape[0]-1 or not img[i + 1, j]:
            ij_edges.append(np.array([[i+1, j],
                                      [i+1, j+1]]))
        # South
        if j == 0 or not img[i, j-1]:
            ij_edges.append(np.array([[i, j],
                                      [i+1, j]]))
        # West
        if i == 0 or not img[i-1, j]:
            ij_edges.append(np.array([[i, j],
                                      [i, j+1]]))

    if not ij_edges:
        return np.zeros((0, 2, 2))
    else:
        return np.array(ij_edges)


def combine_edges(ij_edges, clean=True):
    """
    Connect all edges defined by 'xy_boundary' (result from the function 'get_all_boundary_edges()')
    to closed boundaries around a object.
    If not all edges are part of the surface of one object a list of closed boundaries is returned (one for every
    object).
    """

    combined_edges_list = []
    while ij_edges.size != 0:
        # Current loop
        xy_cl = [ij_edges[0, 0], ij_edges[0, 1]]  # Start with first edge
        ij_edges = np.delete(ij_edges, 0, axis=0)

        while ij_edges.size != 0:
            # Get next boundary edge (edge with common node)
            ij = np.nonzero((ij_edges == xy_cl[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                xy_cl.append(xy_cl[0])
                break

            xy_cl.append(ij_edges[i, (j + 1) % 2, :])
            ij_edges = np.delete(ij_edges, i, axis=0)

        xy_cl = np.array(xy_cl)

        # Clean list
        if clean:
            xy_cl = clean_grid_line(xy_cl)

        combined_edges_list.append(xy_cl)

    return combined_edges_list


def get_combined_edges(img):
    ij_edges = get_edges(img=img)
    return combine_edges(ij_edges=ij_edges)


# 3D
def get_faces(img):
    """
    Get a list of all faces (where the value changes from 'True' to 'False') in the image.
    Only works for 3D, see 'get_edges()' for 2D.
    """

    shape = img.shape

    ijk_faces = []
    ii, jj, kk = np.nonzero(img)
    for i, j, k in zip(ii, jj, kk):
        # East
        if i == shape[0]-1 or not img[i+1, j, k]:
            ijk_faces.append(np.array([[i+1, j, k],
                                       [i+1, j+1, k+1]]))

        # West
        if i == 0 or not img[i-1, j, k]:
            ijk_faces.append(np.array([[i, j, k],
                                       [i, j+1, k+1]]))

        # North
        if j == shape[1]-1 or not img[i, j+1, k]:
            ijk_faces.append(np.array([[i, j+1, k],
                                       [i+1, j+1, k+1]]))

        # South
        if j == 0 or not img[i, j-1, k]:
            ijk_faces.append(np.array([[i, j, k],
                                       [i+1, j, k+1]]))

        # Up
        if k == shape[2]-1 or not img[i, j, k+1]:
            ijk_faces.append(np.array([[i, j, k+1],
                                       [i+1, j+1, k+1]]))
        # Down
        if k == 0 or not img[i, j, k-1]:
            ijk_faces.append(np.array([[i, j, k],
                                       [i+1, j+1, k]]))

    ijk_faces = np.array(ijk_faces)
    return ijk_faces


def face_ll_ur2vertices(ijk_faces):
    """
    Convert the representation of a regular rectangle via the 'lower left' and 'upper right' coordinate to
    representation consisting of a tuple of all 4 vertices.
    """

    n_faces = ijk_faces.shape[0]
    xyz_faces = ijk_faces[:, 0, :][:, np.newaxis, :].repeat(4, axis=1)
    xyz_faces[:, 2, :] = ijk_faces[:, 1, :]

    for i in range(n_faces):
        plane = np.nonzero(xyz_faces[i, 2, :]-xyz_faces[i, 0, :] == 0)[0][0]
        changing_indices = np.sort(list({0, 1, 2}.difference({plane})))
        xyz_faces[i, 1, changing_indices[0]] = xyz_faces[i, 2, changing_indices[0]]
        xyz_faces[i, 3, changing_indices[1]] = xyz_faces[i, 2, changing_indices[1]]

    return xyz_faces


def __get_plane(face):
    """
    Get the plane of the face.
    A surface parallel to the xy(01)-plane lays in the z(2)-plane (direction in which there is no change/
    direction of the surface normal)
    """

    return int(np.nonzero(np.sum(np.sum(face, axis=0) / 4 == face, axis=0))[0][0])


def __is_neighbor(face_plane_1, face_plane_2):
    """
    Check if two surfaces are next to each other (have a common edge) and lay in the same plane.
    -> Use this information, to combine such faces to a bigger connected surface. This makes mpl easier,

    """

    face_1, plane_1 = face_plane_1
    face_2, plane_2 = face_plane_2
    plane_1 = int(plane_1)
    plane_2 = int(plane_2)
    if plane_1 == plane_2 and face_1[0, plane_1] == face_2[0, plane_2]:
        neighbor_direction = list({0, 1, 2}.difference({plane_1}))
        for nd in neighbor_direction:
            if np.all(face_1[:, nd] == face_2[:, nd]):
                free_direction = list(set(neighbor_direction).difference({nd}))[0]
                free_1 = set(face_1[:, free_direction])
                free_2 = set(face_2[:, free_direction])
                if not free_1.isdisjoint(free_2):
                    # print(face_1, '\n', face_2, '\n\
                    return free_direction

        return -1
    else:
        return -1


def combine_faces(face_vtx, verbose=0):
    """
    Combine neighbouring surfaces to bigger surfaces to compress the representation of the object.
    """

    n_faces = face_vtx.shape[0]
    planes = np.zeros(n_faces)
    for i in range(n_faces):
        planes[i] = __get_plane(face_vtx[i, ...])

    if verbose >= 1:
        print('Initial number of faces: ', n_faces)

    while True:
        combine_count = 0
        i = 0
        while i < n_faces:
            j = 1
            while j < n_faces:
                if i == j:
                    j += 1
                    continue

                try:
                    free_direction = __is_neighbor(face_plane_1=(face_vtx[i, ...], planes[i]),
                                                   face_plane_2=(face_vtx[j, ...], planes[j]))
                except IndexError:
                    print(i, j, n_faces)
                    print(face_vtx.shape, planes.shape)
                    raise IndexError

                if free_direction != -1:
                    # Replace the the coordinates of one edge by the edge of the other face to get the combined face
                    fd_coord_set_1 = set(face_vtx[i, :, free_direction])
                    fd_coord_set_2 = set(face_vtx[j, :, free_direction])
                    old_value = list(fd_coord_set_1.difference(fd_coord_set_2))[0]
                    new_value = list(fd_coord_set_2.difference(fd_coord_set_1))[0]
                    common_value = list(fd_coord_set_2.intersection(fd_coord_set_1))[0]
                    face_vtx[i, :, free_direction] = np.where(face_vtx[i, :, free_direction] == common_value,
                                                              x=new_value, y=old_value)

                    # Delete the obsolete faces
                    face_vtx = np.delete(face_vtx, j, axis=0)
                    planes = np.delete(planes, j, axis=0)
                    n_faces = face_vtx.shape[0]
                    combine_count += 1
                    if j < i:
                        i -= 1

                j += 1

            i += 1

        if verbose >= 1:
            print(n_faces, combine_count)
        if combine_count == 0:
            break

    if verbose >= 1:
        print('Final number of faces: ', n_faces)

    return face_vtx


def get_combined_faces(img):
    ijk_faces = get_faces(img=img)
    face_vtx = face_ll_ur2vertices(ijk_faces=ijk_faces)
    return combine_faces(face_vtx=face_vtx)


# Rectangles  # TODO do not fit quite in this method, the rest is so clean and without connections
def rectangles2face_vertices(rect_pos, rect_size):
    """
    Get the faces of the obstacles in 3D. It's cheaper to plot the faces instead of the whole volumes
    Returns a list of faces for all obstacles (6 per obstacle * n_obstacles)

    The used indices are:
       o j k
    0  0 0 0
    1  0 0 1
    2  0 1 0
    3  0 1 1
    4  1 0 0
    5  1 0 1
    6  1 1 0
    7  1 1 1
    """

    n_obstacles = rect_pos.shape[0]
    #                        # 6 faces for each cube             # 4 vertices for each face
    face_vtx = rect_pos.repeat(6, axis=0)[:, np.newaxis, :].repeat(4, axis=1)

    for i in range(n_obstacles):
        # Get the 8 vertices at the nodes of the cube
        ll_vtx = face_vtx[i * 6, 0, :]
        cube_vtx = ll_vtx[np.newaxis, :].repeat(8, axis=0)
        cube_vtx[1, 2] += rect_size[i, 2]
        cube_vtx[2, 1] += rect_size[i, 1]
        cube_vtx[3, [1, 2]] += rect_size[i, [1, 2]]
        cube_vtx[4, 0] += rect_size[i, 0]
        cube_vtx[5, [0, 2]] += rect_size[i, [0, 2]]
        cube_vtx[6, [0, 1]] += rect_size[i, [0, 1]]
        cube_vtx[7, [0, 1, 2]] += rect_size[i, [0, 1, 2]]

        # Get the 6 faces from the corner nodes
        face_vtx[i * 6+0, :, :] = cube_vtx[[0, 2, 3, 1], :]  # West
        face_vtx[i * 6+1, :, :] = cube_vtx[[4, 6, 7, 5], :]  # East
        face_vtx[i * 6+2, :, :] = cube_vtx[[2, 6, 7, 3], :]  # North
        face_vtx[i * 6+3, :, :] = cube_vtx[[0, 4, 5, 1], :]  # South
        face_vtx[i * 6+4, :, :] = cube_vtx[[1, 5, 7, 3], :]  # Up
        face_vtx[i * 6+5, :, :] = cube_vtx[[0, 4, 6, 2], :]  # Down

    return face_vtx
