r"""
.. module:: MeshSlicer
   :synopsis: Generate and process tetrahedral meshes from triangular surface meshes,
              convert between mesh representations, and extract level sets(isosurfaces)
              based on scalar fields defined on tetrahedral meshes.

This module provides:

- :func:`tetrahedralize` to build a tetrahedral mesh using the wildmeshing library.
- :func:`tetrahedral_mesh_to_pyvista` to convert raw tetrahedral data into a PyVista UnstructuredGrid.
- :func:`trimesh_to_o3d` to convert a trimesh.Trimesh object into an Open3D mesh and scene.
- :class:`Mesh` for managing tetrahedral meshes, scalar fields, and extracting level-set surfaces.
"""
import wildmeshing as wm
import open3d as o3d
import numpy as np
import trimesh
import plotly.graph_objects as go
from plotly.colors import sample_colorscale, get_colorscale, unlabel_rgb
import pyvista as pv

def produce_printing_data(mesh_obj, n_levelsets, save_name_isosurfaces, save_name_thickness):
    r'''Given a Mesh object and the desired number of levelsets, produces the data to feed the G-code generator.
    
    Parameters
    ----------
    mesh_obj: Mesh object
    n_levelsets: int
    
    Returns:
    two npz files, isosurfaces.npz and thickness_per_vertex.npz
    When read, isosurfaces is a list of levelsets with vertices and edges information thickness_per_vertex.npz is a list of numpy arrays of floats of the same length as isosurfaces and the i-th element has the same length as the i-th element of the number of vertices of the i-th isosurface, representing the thickness at that point'''
    list_of_levelsets, list_of_distances = mesh_obj.provide_printing_data(n_levelsets)
    
    # save the list_of_levelsets and list_of_distances as two npz files
    # list_of_levelsets contains tuples of (vertices, faces), so we need to save them properly
    isosurfaces_dict = {}
    for i, (vertices, faces) in enumerate(list_of_levelsets):
        isosurfaces_dict[f'vertices_{i}'] = vertices
        isosurfaces_dict[f'faces_{i}'] = faces
    np.savez_compressed(save_name_isosurfaces, **isosurfaces_dict)
    
    # save the thickness data
    thickness_dict = {f'thickness_{i}': dist for i, dist in enumerate(list_of_distances)}
    np.savez_compressed(save_name_thickness, **thickness_dict)
    return f'Data saved as {save_name_isosurfaces} and {save_name_thickness}'


def get_boundary(level_set):
    mesh, _ = pickable_to_o3d_mesh(level_set )
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    level_set = trimesh.Trimesh(vertices=vertices, faces = faces)
    # the following two lines are intended to avoid weird behaviour
    level_set.update_faces(level_set.unique_faces())          # drop exact duplicates
    level_set.update_faces(level_set.nondegenerate_faces())   # drop zero-area faces
    # now we have a trimesh object representing the level set and we can extract the boundary vertices safely
    boundary_vertices = level_set.outline().discrete
    # print('boundary vertices are', len(boundary_vertices))
    if len(boundary_vertices) == 0:
        # this means there is no boundary detected, so we try to reconstruct the levelset
        vertices, faces = reconstruct_levelset(vertices, faces)
        level_set = trimesh.Trimesh(vertices=vertices, faces = faces)
        level_set.update_faces(level_set.unique_faces())          # drop exact duplicates
        level_set.update_faces(level_set.nondegenerate_faces())   # drop zero-area faces
        # now we have a trimesh object representing the level set and we can extract the boundary vertices safely
        boundary_vertices = level_set.outline().discrete
    return boundary_vertices

def reconstruct_levelset(vertices, triangles):
    ''' input is an array of vertices and an array of triangles.
    Performs some manipuilations to reconstruct the levelset. Calls pyvista to do so.
    Returns the new levelset as vertices and triangles'''
    # print("Recosntruction was invoqued")
    faces_pv = np.insert(triangles, 0, 3, axis=1)
    # Create the PyVista PolyData object from our vertices and faces
    # This is the core step for using your own mesh data.
    tri_mesh = pv.PolyData(vertices, faces_pv)

    # 4. Apply Catmull-Clark subdivision to get a quad mesh
    # The process is identical to the previous example.
    new_mesh = tri_mesh.subdivide(2, subfilter='loop')
    faces_with_padding = new_mesh.faces.reshape(-1, 4)
    faces = faces_with_padding[:, 1:]
    return new_mesh.points, faces

def pickable_to_o3d_mesh(data):
    r"""
    Convert raw vertex–triangle arrays into an Open3D mesh **and** a
    ray‑casting scene.

    Parameters
    ----------
    data : tuple(np.ndarray, np.ndarray)
        *(vertices, triangles)* where **vertices** has shape ``(N, 3)``
        and **triangles** has shape ``(M, 3)``.

    Returns
    -------
    tuple(o3d.geometry.TriangleMesh, o3d.t.geometry.RaycastingScene)
        The legacy mesh (first element) and a :class:`open3d.t.geometry.
        RaycastingScene` already containing the same geometry.
    """
    vertices, triangles = data
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh2)
    return(mesh, scene)

def o3d_mesh_to_pickable(mesh):
    r"""
    Extract NumPy *pickables* from an Open3D triangle mesh.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        Open3D legacy mesh.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        ``(vertices, triangles)`` ready for downstream NumPy routines.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertices_and_triangles =(vertices, triangles)
    return vertices_and_triangles

def gradient_and_cosine(normals):
    r"""
    Given an array of unit vectors, computes the vectors of greatest ascent on the planes whose normal vector are normals and the cosine of the normal vector with respect to the global **+Z** axis.

    For a unit vertex normal **n** the gradient of the unsigned distance
    field relative to the global **+Z** axis is

    .. math::

        \mathbf{g} = \mathbf{\hat z} -(\mathbf{\hat z}\!\cdot\!\mathbf n)\,
                     \mathbf n,

    where :math:`\mathbf{\hat z}=(0,0,1)`.  The same dot‑product
    *cos α = n·ẑ* is returned for convenience.

    Parameters
    ----------
    normals : np.ndarray, shape(n_vertices, 3)
        Unit normals at the mesh vertices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        * **vectors** –(n_vertices, 3) gradient vectors lying in the tangent
          plane at each vertex.  
        * **cosines** –(n_vertices,) cosine of the over‑hang angle α between
          the normal and **+Z**.
    """
    # the gradient is the projection of the z vector on the tangent plane
    # grad = v -(v dot n) * n
    # where v is the vector(0,0,1) and n is the normal
    dots = normals.dot(np.array([0, 0, 1]))
    vectors = np.array([0, 0, 1]) - dots[:, np.newaxis]*normals
    cosines = dots
    return vectors, cosines

def tetrahedralize(triangle_mesh):
    r"""
    Tetrahedralize a triangular mesh.

    :param triangle_mesh: Input triangular mesh to be tetrahedralized.
    :type triangle_mesh: trimesh.Trimesh
    :return: Tuple(VT, FT) where VT is an array of vertex coordinates and FT is an array of tetrahedron indices.
    :rtype: tuple of(np.ndarray, np.ndarray)
    """
    vertices = triangle_mesh.vertices
    faces = triangle_mesh.faces
    tetra = wm.Tetrahedralizer(stop_quality=1000)
    # tetra = wm.Tetrahedralizer( stop_quality=1000,     # don't stop early on energy
    #                             max_its=0,           # no optimization passes
    #                             stage=2,             # 3D envelope stage
    #                             epsilon=1e-5,        # tight envelope (relative)
    #                             edge_length_r=1/20,  # typical default; adjust if you want coarser/finer
    #                             # some builds expose:
    #                             skip_simplify=True   # same effect as --skip-simplify (if available in your build)
    #                             )
    tetra.set_mesh(vertices, faces)
    tetra.tetrahedralize()
    _ = tetra.get_tet_mesh()
    VT = _[0]
    FT = _[1]
    return VT, FT

def tetrahedral_mesh_to_pyvista(VT, FT):
    r"""
    Convert raw tetrahedral mesh data into a PyVista UnstructuredGrid.

    :param VT: Array of vertex coordinates, shape(n_vertices, 3).
    :type VT: np.ndarray
    :param FT: Array of tetrahedral connectivity, shape(n_tetrahedra, 4).
    :type FT: np.ndarray
    :return: A PyVista UnstructuredGrid representing the tetrahedral mesh.
    :rtype: pyvista.UnstructuredGrid
    """
    num_tets = FT.shape[0]
    cells = np.hstack([
        np.full((num_tets, 1), 4, dtype=np.int64),
        FT.astype(np.int64)
    ]).ravel()

    # 4. Cell types: VTK_TETRA = 10
    cell_types = np.full(num_tets, pv.CellType.TETRA, dtype=np.uint8)

    # 5. Create the UnstructuredGrid
    pyvista_tetmesh = pv.UnstructuredGrid(cells, cell_types, VT)
    return pyvista_tetmesh

def o3d_to_trimesh(o3d_mesh):
    r"""
    Convert an Open3D legacy mesh into :class:`trimesh.Trimesh`.

    Parameters
    ----------
    o3d_mesh : o3d.geometry.TriangleMesh
        Source mesh.

    Returns
    -------
    trimesh.Trimesh
        Triangle mesh with identical vertex order and (when present)
        per‑vertex normals.  Geometry is transferred *verbatim* – no
        repairs or re‑ordering are performed.
    """
    # gets an o3d object and returns the equivalent trimesh for the utilities
    vertices = np.asarray(o3d_mesh.vertices)    # shape:(N, 3)
    faces    = np.asarray(o3d_mesh.triangles)   # shape:(M, 3)

    # 3.(Optional) If you need per-vertex normals in Trimesh:
    if o3d_mesh.has_vertex_normals():
        normals = np.asarray(o3d_mesh.vertex_normals)
    else:
        normals = None

    # 4. Create the Trimesh object
    trimesh_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,    # or just leave this out
        process=False              # keep original ordering(no auto-repair)
    )
    return trimesh_mesh

def trimesh_to_o3d(trimesh_object):
    r"""
    Convert a trimesh.Trimesh into an Open3D mesh and raycasting scene.

    :param trimesh_object: Triangular mesh to be converted.
    :type trimesh_object: trimesh.Trimesh
    :return: Tuple(o3d.geometry.TriangleMesh, o3d.t.geometry.RaycastingScene).
    :rtype: tuple
    :raises ValueError: If trimesh_object is not a trimesh.Trimesh.
    """
    # Convert a trimesh object to an Open3D mesh
    if not isinstance(trimesh_object, trimesh.Trimesh):
        raise ValueError("trimesh_object must be a trimesh object.")
    vertices = trimesh_object.vertices
    faces = trimesh_object.faces
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_triangle_normals()
    mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh2)
    return(o3d_mesh, scene)

def triangulate_pointcloud(points: np.ndarray, M: int, N: int) -> np.ndarray:
    """
    Triangulate a point‐cloud arranged in N circles(rings) of M points each,
    closing the surface by joining the last ring back to the first, with outward normals.

    Parameters
    ----------
    points : np.ndarray, shape(M*N, D)
        The point‐cloud, ordered so that points[i*M + j] is the j-th point
        on the i-th ring(0 <= i < N, 0 <= j < M).
    M : int
        Number of points per ring.
    N : int
        Number of rings.

    Returns
    -------
    faces : np.ndarray, shape(F, 3), dtype=int
        Array of triangle indices into `points`, with normals flipped outward.
        There are 2*N*M triangles.

    Raises
    ------
    ValueError
        If points.shape[0] != M*N.
    """
    if points.shape[0] != M * N:
        raise ValueError(f"Expected {M}×{N}={M*N} points, got {points.shape[0]}")

    faces = []
    for ring in range(N):
        base      = ring * M
        next_base =((ring + 1) % N) * M
        for j in range(M):
            j_next =(j + 1) % M

            v0 = base       + j
            v1 = base       + j_next
            v2 = next_base  + j
            v3 = next_base  + j_next

            # swap the last two vertices on each face to flip its normal
            faces.append((v0, v3, v1))
            faces.append((v0, v2, v3))

    return np.array(faces, dtype=int)

def polyline_from_points(points):
    r"""
    Build a PyVista PolyData polyline from a sequence of points.

    :param points: Array or list of 3D points defining the vertices of the polyline, shape(n, 3).
    :type points: np.ndarray or sequence
    :returns: A PolyData object with the line defined by the input points.
    :rtype: pv.PolyData
    """
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly

def canal_surface(centers, radii, n_sides = 6):
    r"""
    Build a canal(variable‑radius tube) surface as a *trimesh* object.

    :param centers: Centre points along the spine of the canal surface.
    :type  centers: array‑like shape *(n, 3)*
    :param radii: Radius value at each centre point(same length as
      *centers*).
    :type  radii: array‑like shape *(n,)*
    :return: A watertight `trimesh.Trimesh` representing the canal surface.
    :rtype:  trimesh.Trimesh

    Implementation notes
    --------------------
    * Uses **PyVista** → **VTK `vtkTubeFilter`** to inflate a line whose
      point‑scalars carry the radii(`capping=False` leaves the tube ends open).  
    * Converts the resulting triangle‑strip `PolyData` to ordinary triangles
      with :py:meth:`pyvista.PolyData.triangulate`.  
    * Geometry only – end‑caps are *not* generated, and no boolean union of
      multiple loops is attempted.
    """
    if len(centers) != len(radii):
        raise ValueError("centers and radii must have the same length.")
    if len(centers) == 0:
        raise ValueError("centers and radii must not be empty.")

    polyline = polyline_from_points(centers)
    polyline['radius'] = radii
    tube = polyline.tube(
        radius=None,              # use per-point scalars
        scalars="radius",
        absolute=True,            # radii are in world units
        n_sides=n_sides,
        capping=False             # end‑caps are omitted(capping=False)
    )
    # VTK's TubeFilter yields (len(centers)-1) rings for open polylines when capping=False,
    # and len(centers) rings for closed polylines. Compute the correct number of rings
    # before building connectivity, otherwise the point count won't match.
    closed = np.allclose(centers[0], centers[-1])
    N_rings = len(centers) if closed else (len(centers) - 1)

    # If something odd happens, infer rings from the actual point count as a fallback
    expected = n_sides * max(N_rings, 0)
    if tube.points.shape[0] != expected and tube.points.shape[0] % n_sides == 0:
        N_rings = tube.points.shape[0] // n_sides

    faces = triangulate_pointcloud(tube.points, M=n_sides, N=N_rings)
    trimesh_object = trimesh.Trimesh(vertices=tube.points, faces=faces)
    return trimesh_object

def extract_triangles_mesh_from_tetramesh(vertices, tetras, values, level):
    r"""
    Extract an iso‑surface from a tetrahedral mesh.

    Utilises :py:meth:`open3d.geometry.TetraMesh.extract_triangle_mesh`,
    then converts the resulting Open3D mesh into *pickable*
    ``(vertices, faces)`` arrays.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Tetrahedral vertex coordinates.
    tetras : np.ndarray, shape (M, 4)
        Indices of tetrahedron vertices.
    values : np.ndarray, shape (N,)
        Scalar field defined at *vertices*.
    level : float
        Iso‑value to extract.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Triangle surface as ``(vertices, faces)``.
    """
    # construct tetramesh from vertices and tetras
    tetramesh = o3d.geometry.TetraMesh()
    tetramesh.vertices = o3d.utility.Vector3dVector(vertices)
    tetramesh.tetras   = o3d.utility.Vector4iVector(tetras)
    # Create an Open3D DoubleVector from the numpy array (or from a list):
    double_vec = o3d.utility.DoubleVector(values.tolist())

    triangle_surface = tetramesh.extract_triangle_mesh(
        values = double_vec,
        level  = level
    )
    # now triangle_surface is an o3d object but we want to return the pickable
    return o3d_mesh_to_pickable(triangle_surface)

def split_trimesh_into_connected_components(data):
    r"""
    Split a triangle mesh into its connected components.

    Parameters
    ----------
    data : tuple(np.ndarray, np.ndarray)
        ``(vertices, faces)`` describing the input surface.

    Returns
    -------
    list[tuple(np.ndarray, np.ndarray)]
        List of *pickables* – one per connected component, each formatted
        as ``(vertices, faces)``.
    """
    # data is data = [vertices, faces]
    # first, creaet the triangle surface
    triangle_surface = pickable_to_o3d_mesh(data = data)
    # Correctly unpack the three outputs:
    triangle_labels, cluster_n_triangles, cluster_area = triangle_surface.cluster_connected_triangles()

    # ‘triangle_labels’ is an array of length = number_of_triangles,
    # where each entry is the cluster ID that triangle belongs to.

    unique_clusters = np.unique(triangle_labels)
    meshes = []
    for cid in unique_clusters:
        # find all triangle‐indices belonging to cluster ‘cid’
        tri_indices = np.where(triangle_labels == cid)[0]

        # collect the vertices used by those triangles
        tris = np.asarray(triangle_surface.triangles)[tri_indices]
        verts_idx = np.unique(tris.flatten())

        # remap old→new indices
        old_to_new = {old: new for new, old in enumerate(verts_idx)}
        comp_vertices = np.asarray(triangle_surface.vertices)[verts_idx]
        comp_triangles = np.array([[old_to_new[v] for v in tri] for tri in tris])

        comp_mesh = o3d.geometry.TriangleMesh()
        comp_mesh.vertices = o3d.utility.Vector3dVector(comp_vertices)
        comp_mesh.triangles = o3d.utility.Vector3iVector(comp_triangles)
        # make all those meshes pickable
        meshes.append(o3d_mesh_to_pickable(comp_mesh))
    # return the list of pickables
    return meshes

class Mesh:
    
    def __init__(self, mesh_to_print, reference_surface, nozzle_diameter=0.4, nozzle_half_angle=np.pi/6, H_init_method='default'):
        r"""
        Construct a tetrahedral representation of *mesh_to_print* and cache
        auxiliary data required for level‑set extraction and printability
        checks.

        Parameters
        ----------
        mesh_to_print : trimesh.Trimesh
            Triangular mesh representing the part to be printed.
        reference_surface : trimesh.Trimesh
            Reference surface used to initialise the unsigned distance field
            *H* (as‐printed accuracy evaluation).
        nozzle_diameter : float, optional
            Nominal nozzle diameter in mm.  Defines the admissible layer
            thickness range  
            ``min_thickness = 0.10 × d`` and
            ``max_thickness = 0.75 × d``.
        nozzle_half_angle : float, optional
            Half‑angle θₕ (in rad) of the nozzle opening.  Controls the
            over‑hang criterion via the pre‑computed
            ``nozzle_half_angle_complimentary_cosine = cos(π − θₕ)``.

        Raises
        ------
        ValueError
            If *mesh_to_print* or *reference_surface* is not a
            :class:`trimesh.Trimesh`.
        """
        # mesh_to_print and reference_surface are both trimesh objects
        # it assumes that the scale is in mm
        if not isinstance(mesh_to_print, trimesh.Trimesh):
            raise ValueError("mesh_to_print must be a trimesh object.")
        if not isinstance(reference_surface, trimesh.Trimesh):
            raise ValueError("reference_surface must be a trimesh object.")
        self.triangle_mesh_vertice, self.triangle_mesh_faces = mesh_to_print.vertices, mesh_to_print.faces
        self.tet_vert, self.tet_faces = tetrahedralize(mesh_to_print)
        self.reference_surface_vertices, self.reference_surface_faces = reference_surface.vertices, reference_surface.faces
        self.nozzle_diameter = nozzle_diameter
        self.max_thickness = 0.75*self.nozzle_diameter
        self.min_thickness = 0.1*self.nozzle_diameter
        self._H = self.initialize_H(method=H_init_method)
        # self.max_NLEVELSETS and self.min_NLEVELSETS are computed based on the scalar field H at the very beginning. The user should not change them and they are not settable nor visible
        self.max_NLEVELSETS = self._H.max()//self.min_thickness
        self.min_NLEVELSETS = self._H.max()//self.max_thickness
        self.nozzle_half_angle = nozzle_half_angle
        self.nozzle_half_angle_sine = np.sin(nozzle_half_angle)
        self.printing_mesh_vertex_normals = mesh_to_print.vertex_normals
        self.printing_mesh_offset_verts = self.triangle_mesh_vertice + self.max_thickness*self.printing_mesh_vertex_normals
        

    def initialize_H(self, method='default'):
        r"""
        Build the unsigned distance field *H* at every tetrahedral vertex.

        The computation relies on Open3D’s ray‑casting scene created from
        *reference_surface*.  Distances smaller than ``1 e‑4`` mm are clamped
        to zero.

        Parameters
        ----------
        method : str, default ``'default'``
            Method to use for initializing *H*.  Currently supported methods
            are:
            - ``'default'`` – Compute the unsigned distance to the reference
              surface.
            - ``'flat'`` – Use the *Z* coordinate of each vertex as its
              distance.

        Returns
        -------
        np.ndarray
            1‑D array of length ``len(self.tet_vert)`` containing the distance
            from each vertex to the reference surface.
        """
        # Initialize the scalar field H on the tetrahedral mesh
        # H is a numpy array of shape(N,) where N is the number of VERTICES in the tetrahedral mesh(self.tet_vert)
        # H is initialized by the distance to the reference surface
        if method == 'default':
            points = o3d.core.Tensor(self.tet_vert.astype(np.float32), dtype=o3d.core.Dtype.Float32)
            _, scene = pickable_to_o3d_mesh(data=[self.reference_surface_vertices, self.reference_surface_faces])
            distances = scene.compute_distance(points)
            # convert Open3D Tensor to NumPy array
            distances = distances.numpy()
            # points that are within a tolerace of the reference surface are set to 0
            tolerance = 1e-4
            distances[np.abs(distances) < tolerance] = 0
            return distances
        elif method == 'flat':
            return self.tet_vert[:, 2]
        else:
            raise ValueError("Unknown method for initializing H.")
    
    @property
    def H(self):
        r"""
        Get the current scalar field H values for tetrahedral vertices.

        :return: Array of distances of shape(n_vertices,).
        :rtype: np.ndarray
        """
        return self._H

    @H.setter
    def H(self, value):
        r"""
        Set the scalar field H for tetrahedral vertices.

        :param value: Array of distances of shape(n_vertices,).
        :type value: np.ndarray
        :raises ValueError: If value is not an ndarray or has incorrect length.
        """
        if not isinstance(value, np.ndarray):
            raise ValueError("H must be a numpy array.")
        if value.shape[0] != len(self.tet_vert):
            raise ValueError("H must have the same number of elements as the number of vertices in the tetrahedral mesh.")
        self._H = value
        
    def get_one_levelset(self, levelset_value, split = False):
        r"""
        Extract a single iso‑surface of *H*.

        Parameters
        ----------
        levelset_value : float
            Iso‑value (distance) at which the surface is extracted.
        split : bool, default ``False``
            When *True* the surface is decomposed into its connected
            components; the result is then a **list** of pickables.

        Returns
        -------
        tuple(np.ndarray, np.ndarray) | list[tuple(np.ndarray, np.ndarray)]
            *Pickable* ``(vertices, faces)`` or a list thereof when
            *split=True*.
        """
        # Get a single level set from the tetrahedral mesh
        # H is a numpy array of shape(N,) where N is the number of VERTICES in the tetrahedral mesh
        # levelset_value is a single value for which to compute the isosurface
        # returns a trimesh object representing the isosurface in as many submeshes as there are connected components
        mesh = extract_triangles_mesh_from_tetramesh(self.tet_vert, self.tet_faces, self._H, levelset_value)
        # recall mesh is vertices and its connectivity, i.e., mesh = [vertices, faces]
        if split:
            return split_trimesh_into_connected_components(mesh)
        else:
            return mesh 
        
    def get_multiple_levelsets(self, levelset_values, split = False):
        r"""
        Extract several iso‑surfaces in a single call.

        Parameters
        ----------
        levelset_values : Iterable[float]
            Sequence of iso‑values (must be monotonically increasing from
            inner to outer surfaces).
        split : bool, default ``False``
            Forwarded to :meth:`get_one_levelset`.

        Returns
        -------
        list
            One element per requested iso‑value; see
            :meth:`get_one_levelset` for the exact return type.
        """
        # Get level sets from the tetrahedral mesh
        # H is a numpy array of shape(N,) where N is the number of VERTICES in the tetrahedral mesh
        # levelset_values is a list of values for which to compute the isosurfaces
        # returns a list of trimesh objects representing the isosurfaces
        # works by calling get_one_levelset  for each levelset value
        all_levelsets = [
            self.get_one_levelset(levelset_value = levelset_value, split = split)
            for levelset_value in levelset_values
        ]
        return all_levelsets

    def construct_NLEVELSETS(self, NLEVELSETS):
        r"""
        Generate *NLEVELSETS* equally‑spaced iso‑surfaces between the minimum
        and maximum of *H*.

        Parameters
        ----------
        NLEVELSETS : int
            Desired number of level‑sets.  Must satisfy  
            ``self.min_NLEVELSETS < NLEVELSETS < self.max_NLEVELSETS``.

        Returns
        -------
        list[tuple(np.ndarray, np.ndarray)]
            List of *pickables* ``(vertices, faces)``.

        Raises
        ------
        ValueError
            If *NLEVELSETS* lies outside the allowed range.
        """
        if not isinstance(NLEVELSETS, int) or NLEVELSETS>= self.max_NLEVELSETS or NLEVELSETS<=self.min_NLEVELSETS:
            raise ValueError(f"NLEVELSETS must be an integer between {self.min_NLEVELSETS} and {self.max_NLEVELSETS}.")
        # Compute the levelset values based on the current scalar field H
        levelset_values = np.linspace(self._H.min(), self._H.max(), NLEVELSETS)
        # Get the isosurfaces for these levelset values
        all_levelsets = self.get_multiple_levelsets(levelset_values)
        return all_levelsets
    
    def check_distance_validity_between_levelsets(self, list_of_levelsets ):
        r"""
        Verify that successive iso‑surfaces respect the layer‑thickness
        bounds.

        Parameters
        ----------
        list_of_levelsets : list[tuple(np.ndarray, np.ndarray)]
            Ordered list of *pickables* produced with ``split=False``.

        Returns
        -------
        tuple
            ``(bool_list, max_spacing, min_spacing)``  
            * **bool_list** – *True* for each pair whose spacing is within
              ``[min_thickness, max_thickness]``.  
            * **max_spacing** – Largest measured spacing.  
            * **min_spacing** – Smallest measured spacing.
        """
        # list_of_levelsets is a list of tuples, output of get_multiple_levelsets or construct_NLEVELSETS
        # Although the first levelset is always going to be for H=0, we cannot ensure that the vertices are well captured, so that levelset will be ignored and we will start checking from the second levelset agains the reference surface
        # Check if the distance between the levelsets is valid
        # returns True if the distance is valid, False otherwise
        is_surface_at_good_distsance = [] #list of bools, one for each levelset starting from i=1
        _, reference_surface = pickable_to_o3d_mesh(data=[self.reference_surface_vertices, self.reference_surface_faces])
        biggest_distance = -np.inf
        smallest_distance = np.inf
        for i in range(1, len(list_of_levelsets )):
            verts = list_of_levelsets [i][0]
            verts = verts.astype(np.float32)
            points = o3d.core.Tensor(verts, dtype=o3d.core.Dtype.Float32)
            distances = reference_surface.compute_distance(points)
            distances = distances.numpy()
            if distances.max() > biggest_distance:
                biggest_distance = distances.max()
            if distances.min() < smallest_distance:
                smallest_distance = distances.min()
            
            if np.all(distances >= self.min_thickness) and np.all(distances <= self.max_thickness):
                is_surface_at_good_distsance.append(True)
            else:
                is_surface_at_good_distsance.append(False)
            _, reference_surface = pickable_to_o3d_mesh(list_of_levelsets[i])
        
        return is_surface_at_good_distsance, biggest_distance, smallest_distance
    
    def construct_canal_surface(self, level_set , previous_isosurface = None, n_sides = 6):
        r"""
        Build one or more canal‑(variable‑radius tube) surfaces along the
        boundary curves of *level_set*.

        Parameters
        ----------
        level_set : tuple(np.ndarray, np.ndarray)
            *Pickable* representing a connected iso‑surface.
        previous_isosurface : tuple(np.ndarray, np.ndarray) | None, optional
            When supplied, distances are measured from this surface instead of
            *reference_surface* (used for consecutive layers).

        Returns
        -------
        list[tuple(np.ndarray, np.ndarray)]
            One *pickable* per canal segment.
        """

        boundary_vertices = get_boundary(level_set)
        #2 . create the corresponding scene
        if previous_isosurface is None:
            _, scene = pickable_to_o3d_mesh(data=[self.reference_surface_vertices, self.reference_surface_faces])
        else:
            _, scene = pickable_to_o3d_mesh(previous_isosurface)
        #3. compute the distance from the boundary vertices to the scene
        # boundary_vertices is a list of numpy arrays, but we need a single numpy array
    
        boundary_vertices_one_array = np.vstack(boundary_vertices)
        

        points = o3d.core.Tensor(boundary_vertices_one_array.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        distances = scene.compute_distance(points)
        distances = 0.5*distances.numpy()
        #4. construct the canal surface for each of the boundary curves
        len_of_curve = 0
        canal_surfaces = []
        for boundary_curve in boundary_vertices:
            len_of_curve1 = len(boundary_curve)
            # the corresponding radii are the distances up to index
            radii = distances[len_of_curve:len_of_curve1 + len_of_curve]
            centers_of_spheres = boundary_curve - radii[:, np.newaxis]*np.array([0, 0, 1]) # the centers of the spheres are the boundary curve points minus the radius in the z direction
            # print('centers of spheres are', centers_of_spheres.shape, 'radii are', radii.shape)
            canal_surface_obj = canal_surface(centers = centers_of_spheres, radii = radii, n_sides = n_sides)
            canal_surfaces.append(canal_surface_obj)
            len_of_curve += len_of_curve1
        #5. return the canal surfaces as a list of pickables
        
        return [o3d_mesh_to_pickable(_.as_open3d) for _ in canal_surfaces]
  
    def provide_printing_data(self, n_levelsets):
        # given the number of levelsets, produces the levelsets as a list of isovalues that will be returned as a list of pickables, one element per isovalue, and another arary of matching dimensions which has the thickness per point.

        threshold = 1e-3
        # Obtain the levelsets
        various_levelsets = np.linspace(self.H.min() + threshold, self.H.max() - threshold, n_levelsets)
        # Remove the first levelset, which is the support surface
        various_levelsets = various_levelsets[1:]
        # Obtain the multiple levelsets
        list_of_levelsets = self.get_multiple_levelsets(levelset_values=various_levelsets)
        # this is the levelsets as a list of pickables
        # Now, compute the thickness per point
        list_of_distances = []
        previous_isosurface = None
        for level_set in list_of_levelsets:
            if previous_isosurface is None:
                _, scene = pickable_to_o3d_mesh(data=[self.reference_surface_vertices, self.reference_surface_faces])
            else:
                _, scene = pickable_to_o3d_mesh(previous_isosurface)
            points = o3d.core.Tensor(level_set[0].astype(np.float32), dtype=o3d.core.Dtype.Float32)
            distances = scene.compute_distance(points)
            distances = distances.numpy()
            list_of_distances.append(distances)
            previous_isosurface = level_set
        return list_of_levelsets, list_of_distances

    def are_levelsets_printable(self, list_of_levelsets ):
        r"""
        Evaluate the over‑hang criterion for a list of iso‑surfaces.

        Parameters
        ----------
        list_of_levelsets : list[tuple(np.ndarray, np.ndarray)]
            Ordered from inner to outer surface.

        Returns
        -------
        list[bool]
            ``True`` if *all* vertices of the corresponding level‑set satisfy  
            ``cos(α) ≥ cos(π − θₕ)``.
        """
        # return a list of booleans, one for each levelset, indicating if the levelset is printable based on the gradient
        list_of_meshes = [pickable_to_o3d_mesh(data)[0] for data in list_of_levelsets ]
        normals = np.vstack([np.asarray(mesh.vertex_normals) for mesh in list_of_meshes])
        # compute the gradient at each vertex
        _, cosines = gradient_and_cosine(normals)
        # create a boolean mask: True where the overhang is printable
        mask = cosines > self.nozzle_half_angle_sine

        # split the mask once, based on cumulative vertex counts,
        # then reduce with np.all ‑‑ avoids Python‑level loops
        sizes = [len(data[0]) for data in list_of_levelsets ]
        splits = np.split(mask, np.cumsum(sizes)[:-1])
        boolean_by_levelset = [arr.all() for arr in splits]

        return boolean_by_levelset

    def get_various_canal_surfaces(self, list_of_levelsets , previous_isosurface = None, n_sides = 6):
        r"""
        Concatenate canal surfaces generated for multiple level‑sets.

        Parameters
        ----------
        list_of_levelsets : list[tuple(np.ndarray, np.ndarray)]
            Iso‑surfaces for which canals will be generated.
        previous_isosurface : tuple(np.ndarray, np.ndarray) | None, optional
            Passed through to :meth:`construct_canal_surface`.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Combined **vertices** and **faces** arrays of all canal segments.
        """
        canal_surfaces = []
        _previous_isosurface = previous_isosurface
        for level_set in list_of_levelsets:
            canal_surface_list = self.construct_canal_surface(level_set =level_set, previous_isosurface=_previous_isosurface, n_sides = n_sides)
        
            for _ in canal_surface_list:
                canal_surfaces.append(_)
            _previous_isosurface = level_set
        # Build a single pickable by stacking vertices/faces with offsets
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        for verts, tris in canal_surfaces:
            all_vertices.append(verts)
            all_faces.append(tris + vertex_offset)
            vertex_offset += verts.shape[0]
        if all_vertices:
            big_vertices = np.vstack(all_vertices)
            big_faces = np.vstack(all_faces)
        else:
            big_vertices = np.zeros((0, 3))
            big_faces = np.zeros((0, 3), dtype=int)
        return(big_vertices, big_faces)

    def get_cover_mesh(self, list_of_levelsets , initial_isosurface = None, n_sides = 6):
        r"""
        Assemble a *cover* mesh consisting of level‑sets, their canals, and a
        bottom surface.

        Parameters
        ----------
        list_of_levelsets : list[tuple(np.ndarray, np.ndarray)]
            Level‑sets to include.
        initial_isosurface : tuple(np.ndarray, np.ndarray) | None, optional
            If *None*, the reference surface acts as the bottom face.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Combined **vertices** and **faces** of the cover mesh.
        """
        # Determine the bottom surface as pickable
        if initial_isosurface is None:
            bottom_vertices, bottom_faces = self.reference_surface_vertices, self.reference_surface_faces
        else:
            bottom_vertices, bottom_faces = initial_isosurface
            
        # Build pickable for concatenated levelsets
        lvl_vertices = []
        lvl_faces = []
        offset = 0
        for verts, tris in list_of_levelsets :
            lvl_vertices.append(verts)
            lvl_faces.append(tris + offset)
            offset += verts.shape[0]
        if lvl_vertices:
            big_lvl_vertices = np.vstack(lvl_vertices)
            big_lvl_faces = np.vstack(lvl_faces)
        else:
            big_lvl_vertices = np.zeros((0, 3))
            big_lvl_faces = np.zeros((0, 3), dtype=int)

        # Build pickable for concatenated canal surfaces
        canal_vertices, canal_faces = self.get_various_canal_surfaces(
            list_of_levelsets =list_of_levelsets ,
            previous_isosurface=initial_isosurface,
            n_sides = n_sides)

        # Now combine all three: levelsets, canal surfaces, and bottom surface
        combined_vertices = np.vstack([big_lvl_vertices, canal_vertices, bottom_vertices])
        offset1 = big_lvl_vertices.shape[0]
        offset2 = offset1 + canal_vertices.shape[0]
        combined_faces = np.vstack([
            big_lvl_faces,
            canal_faces + offset1,
            bottom_faces + offset2
        ])
        
        return combined_vertices, combined_faces

    def surface_error_by_canal_surfaces(self, list_of_levelsets, n_sides = 6):
        r"""
        Compute the printing‑accuracy error for every vertex of the original
        triangle mesh.

        The function casts one ray per printing‑mesh vertex towards **–n̂**
        (inward normal) and measures the gap to the cover mesh built from
        *list_of_levelsets*.

        Parameters
        ----------
        list_of_levelsets : list[tuple(np.ndarray, np.ndarray)]
            Iso‑surfaces used to construct the cover mesh.

        Returns
        -------
        np.ndarray
            Error at each vertex (positive ⇒ under‑extrusion,
            negative ⇒ over‑extrusion).
        """
        # Build a single Trimesh→Open3D scene so that normals exactly match
        cover_pickable = self.get_cover_mesh(list_of_levelsets =list_of_levelsets,n_sides = n_sides )
        # get an O3D mesh + scene from that pickable; because we built the Trimesh above,
        # the normals are now identical to the non‑pickable version.
        _, scene = pickable_to_o3d_mesh(cover_pickable)
        ray_directions  = -self.printing_mesh_vertex_normals.astype(np.float32)
        ray_origins = self.printing_mesh_offset_verts.astype(np.float32)
        C = np.hstack((ray_origins, ray_directions))

        # intersect the gauging
        rays = o3d.core.Tensor(C,
                            dtype=o3d.core.Dtype.Float32)

        ans1 = scene.cast_rays(rays)
        A = ans1['t_hit'].numpy()
        A = self.max_thickness - A
        # There might be some rays that do not hit the mesh
        # For those points, the distance is infinity.
        # We want to correct that to a high value but not too high.
        distances = np.where(np.isfinite(A), A, 1000)
        return distances
    
    # ------------------------------------------------------------------
    # SERIALISATION / DESERIALISATION
    # ------------------------------------------------------------------
    def save(self, filename: str) -> None:
        r"""
        Serialise the *entire* :class:`Mesh` object to ``filename``.

        The file format is a NumPy ``.npz`` archive that stores:

        ================  =============================================
        Key name          Contents
        ================  =============================================
        ``tri_verts``     (V₀, 3) vertices of *mesh_to_print*
        ``tri_faces``     (F₀, 3) faces of *mesh_to_print*
        ``ref_verts``     (Vᵣ, 3) vertices of *reference_surface*
        ``ref_faces``     (Fᵣ, 3) faces of *reference_surface*
        ``tet_verts``     (Vₜ, 3) vertices of the tetrahedral mesh
        ``tet_tets``      (T, 4)   tetrahedron connectivity
        ``H``             (Vₜ,)    unsigned distance field (float32)
        ``nozzle_diam``   Scalar *d* (float64)
        ``nozzle_angle``  Half‑angle θₕ in radians (float64)
        ================  =============================================

        Parameters
        ----------
        filename : str
            Path ending in ``.npz``.  Parent directories are created
            automatically.

        Notes
        -----
        *The archive is self‑contained*: loading it recreates the mesh
        **without** running :func:`tetrahedralize`, so results are exactly
        reproducible even when TetWild is non‑deterministic.
        """
        if not filename.endswith(".npz"):
            raise ValueError("Filename must end with '.npz'")
        # ensure parent directories exist
        import os, pathlib, numpy as _np
        pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)

        _np.savez_compressed(
            filename,
            tri_verts=self.triangle_mesh_vertice.astype(_np.float64),
            tri_faces=self.triangle_mesh_faces.astype(_np.float64),
            ref_verts=self.reference_surface_vertices.astype(_np.float64),
            ref_faces=self.reference_surface_faces.astype(_np.float64),
            tet_verts=self.tet_vert.astype(_np.float64),
            tet_tets=self.tet_faces.astype(_np.float64),
            H=self._H.astype(_np.float64),
            nozzle_diam=_np.float64(self.nozzle_diameter),
            nozzle_angle=_np.float64(self.nozzle_half_angle),
        )

    @classmethod
    def load(cls, filename: str) -> "Mesh":
        r"""
        Recreate a :class:`Mesh` instance from a ``.npz`` file written by
        :meth:`save`.

        Parameters
        ----------
        filename : str
            Path to the ``.npz`` file.

        Returns
        -------
        Mesh
            New instance whose internal data(*tet_verts*, *tet_tets*, *H*,
            etc.) exactly match the archived values **without**
            re‑tetrahedralising the surface.

        Notes
        -----
        The constructor calls :py:meth:`Mesh.__init__` to build the basic
        scaffolding, then *overrides* the tetrahedral arrays with the stored
        ones to guarantee bit‑for‑bit reproducibility.
        """
        import numpy as _np, pathlib
        if not pathlib.Path(filename).is_file():
            raise FileNotFoundError(filename)
        data = _np.load(filename, allow_pickle=False)
        # Rebuild the surface meshes
        tri_mesh = trimesh.Trimesh(
            vertices=data["tri_verts"],
            faces=data["tri_faces"],
            process=False,
        )
        ref_mesh = trimesh.Trimesh(
            vertices=data["ref_verts"],
            faces=data["ref_faces"],
            process=False,
        )
        obj = cls(
            mesh_to_print=tri_mesh,
            reference_surface=ref_mesh,
            nozzle_diameter=float(data["nozzle_diam"]),
            nozzle_half_angle=float(data["nozzle_angle"]),
        )
        # Override non‑deterministic parts with archived arrays
        obj.tet_vert = data["tet_verts"]
        obj.tet_faces = data["tet_tets"]
        obj._H = data["H"]
        # Recompute dependent constants
        obj.max_NLEVELSETS = obj._H.max() // obj.min_thickness
        obj.min_NLEVELSETS = obj._H.max() // obj.max_thickness
        return obj

class Heatmap:

    def __init__(self, mesh, array_of_scalars, upper_bound, lower_bound,  colorscale = 'turbo_r'):
        """
        Initialize a Heatmap for coloring a mesh by scalar values.

        :param mesh: Triangular mesh whose vertices will be colored.
        :type mesh: trimesh.Trimesh
        :param array_of_scalars: Scalar values corresponding to each vertex of the mesh.
        :type array_of_scalars: np.ndarray
        :param upper_bound: Maximum scalar value for color normalization.
        :type upper_bound: float
        :param lower_bound: Minimum scalar value for color normalization.
        :type lower_bound: float
        :param colorscale: Name of the Plotly colorscale to use(default 'turbo_r').
        :type colorscale: str
        """
        self.mesh = mesh
        self.array_of_scalars = array_of_scalars
        self.colorscale = colorscale
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.rgba_colors = self.scalars_to_colors()
        self.mesh_with_colors = self._mesh_with_colors()
    
    def scalars_to_colors(self):
        """
        Convert scalar values to RGBA colors using the specified colorscale.

        :returns: An(N,4) array of RGBA colors for each scalar in `array_of_scalars`.
        :rtype: np.ndarray
        """
        normalized_values =(self.array_of_scalars - self.lower_bound) /(self.upper_bound - self.lower_bound)
        normalized_values = np.clip(normalized_values, 0, 1)  # Ensure values are within [0, 1]

        # Step 2: Retrieve the colorscale
        colorscale = get_colorscale(self.colorscale)  # Or any other Plotly colorscale

        # Step 3: Sample the colorscale
        colors = sample_colorscale(colorscale, normalized_values, colortype='rgb')

        rgba_colors = []
        for color in colors:
            rgba_colors.append(unlabel_rgb(color))

        rgb_values = np.array(rgba_colors)
        # Step 2: Create an alpha channel(fully opaque)
        alpha_channel = np.full((rgb_values.shape[0], 1), 255, dtype=np.uint8)

        # Step 3: Combine RGB and alpha to get RGBA
        rgba_values = np.hstack((rgb_values, alpha_channel))
        return rgba_values

    def _mesh_with_colors(self):
        """
        Build a new Trimesh with vertex colors applied.

        :returns: A Trimesh object with colored vertices.
        :rtype: trimesh.Trimesh
        """
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        normals = self.mesh.vertex_normals
        vertex_colors = self.rgba_colors
        
        mesh_with_colors = trimesh.Trimesh(vertices=vertices,
                                            faces=faces,
                                            vertex_normals=normals,
                                            vertex_colors=vertex_colors)
        return mesh_with_colors

    def to_Rhino(self, name):
        """
        Export the colored mesh to a PLY file for Rhino.

        :param name: Filename for the exported PLY file; must end with '.ply'.
        :type name: str
        :raises ValueError: If the filename does not end with '.ply'.
        """
        if not name.lower().endswith('.ply'):
            raise ValueError("Error: The file must have a '.ply' extension, otherwise there will be no colors!")
        mesh = self.mesh_with_colors 
        mesh.export(name)

    def plotly_figure(self, showlegend = True, name = 'Heatmap'):
        """
        Generate a Plotly 3D mesh figure with per-vertex scalar coloring.

        :param showlegend: Whether to show the legend/color scale(default True).
        :type showlegend: bool
        :param name: The name/label for this mesh in the Plotly figure.
        :type name: str
        :returns: A Plotly Figure object displaying the colored mesh.
        :rtype: plotly.graph_objects.Figure
        """

        fig = go.Figure()
        cmin, cmax =  self.lower_bound, self.upper_bound

        # make undercutting and overcutting points be the maximum value that we allow
        distances_adjusted_to_tolerances = self.array_of_scalars.copy()
        distances_adjusted_to_tolerances[distances_adjusted_to_tolerances > cmax] = cmax # set undercut
        distances_adjusted_to_tolerances[distances_adjusted_to_tolerances < cmin] = cmin # set overcut
        
        x,y,z = self.mesh.vertices.T
        i,j,k = self.mesh.faces.T
        # plot the heatmap
        fig.add_mesh3d(x = x, y = y, z = z,
                       i=i, j=j, k=k,
                       colorscale = self.colorscale,
                       cmin=cmin,
                       cmax=cmax,
                       intensity=distances_adjusted_to_tolerances,
                       name=name,
                       showscale=True,
                       showlegend=showlegend
        )

        fig.update_layout(
            showlegend = True,
            scene=dict(
                aspectmode='data'),
                width = 900,
                height = 750
            )
        return fig
