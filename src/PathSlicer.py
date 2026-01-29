import numpy as np
import shapely.geometry as sg
import shapely.affinity as sa
import shapely.ops as so
import trimesh
from MeshSlicer import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import json
import math

class LineDataUnits(Line2D):
    """FROM https://stackoverflow.com/questions/19394505/expand-the-line-with-specified-width-in-data-unit/42972469#42972469"""
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)

def importmesh():
    mesh = Mesh.load(r"/geometries/reparation/reparation_initial_mesh_objK60.npz")
    return mesh

def import_levelsets():
    path_isosurfaces = r"D:\LUCAS\COURS\POSTDOC\Curv3D\Curv3D\src\geometries\reparation\reparation_isosurfaces.npz"
    path_thikness = r"D:\LUCAS\COURS\POSTDOC\Curv3D\Curv3D\src\geometries\reparation\reparation_thickness.npz"
    dict_isosurfaces = np.load(path_isosurfaces)
    dict_thikness = np.load(path_thikness)
    return dict_isosurfaces, dict_thikness

def lvlsetmesh(lvlset):
    """Code from Juan that I stole because I dont have any idea of how his stuff work."""
    #1. extract the boundary vertices of the level set
    msh, _ = pickable_to_o3d_mesh(lvlset)
    vertices = np.asarray(msh.vertices)
    faces = np.asarray(msh.triangles)
    lvlset_mesh = trimesh.Trimesh(vertices=vertices, faces = faces)
    # the following two lines are intended to avoid weird behaviour
    lvlset_mesh.update_faces(lvlset_mesh.unique_faces())          # drop exact duplicates
    lvlset_mesh.update_faces(lvlset_mesh.nondegenerate_faces())   # drop zero-area faces
    return lvlset_mesh

def linewidth_mpl(linewidth_data_units, fig, ax):
    # Get the x-axis limits
    xmin, xmax = ax.get_xlim()
    x_range = xmax - xmin
    # Get the figure width in inches
    fig_width_inches, _ = fig.get_size_inches()
    # Get the axes position (left, bottom, width, height) in figure coordinates
    bbox = ax.get_position()
    ax_width_inches = fig_width_inches * bbox.width
    # Calculate the linewidth in points
    # Assuming 72 points per inch and a linear mapping from data to display units
    linewidth_points = (linewidth_data_units / x_range) * ax_width_inches * 72
    return linewidth_points

def show(juanmesh: Mesh):
    o3dmesh = pickable_to_o3d_mesh([juanmesh.triangle_mesh_vertice, juanmesh.triangle_mesh_faces])
    trimeshmesh = o3d_to_trimesh(o3dmesh[0])
    trimeshmesh.show()

class Thickness:
    def __init__(self, dict_isosurface, dict_thickness, i=0):
        # ... (initialisation des couches et assert)
        self.max_layer = len(dict_thickness)
        assert i < self.max_layer
        self.i = i

        base_coords = dict_isosurface["vertices_" + str(i)]
        num_points = len(base_coords)

        current_thickness_values = dict_thickness["thickness_" + str(i)]
        if len(current_thickness_values) != num_points:
            raise ValueError(
                f"Le nombre de points dans 'thickness_{i}' ({len(current_thickness_values)}) ne correspond pas aux coordonnées de base ({num_points}).")

        # Les points d'entrée (X, Y)
        input_points = base_coords[:, :2]
        # Les valeurs de sortie (Z)
        output_values = current_thickness_values.reshape(-1, 1).flatten()

        # 1. Interpolateur Linéaire (pour l'interpolation standard dans l'enveloppe convexe)
        # On utilise fill_value=np.nan pour pouvoir détecter les points extérieurs
        self.interp_linear = LinearNDInterpolator(
            input_points,
            output_values,
            fill_value=np.nan  # Renvoyer NaN si extérieur
        )

        # 2. Interpolateur Nearest (pour gérer les points extérieurs en renvoyant la valeur la plus proche)
        self.interp_nearest = NearestNDInterpolator(
            input_points,
            output_values
        )

        self.max = float(output_values.max())
        self.min = float(output_values.min())


    def __call__(self, x, y):
        # Assurer la forme attendue (N, D)
        query_point = np.array([[x, y]])

        # 1. Tenter l'interpolation linéaire
        result_linear = self.interp_linear(query_point)[0]

        # 2. Vérifier si le résultat est NaN (c'est-à-dire si le point est en dehors de l'enveloppe convexe)
        if np.isnan(result_linear):
            # Si NaN, utiliser l'interpolateur du plus proche voisin
            # Cela équivaut à "clipper l'input pour interpoler la valeur la plus proche"
            return self.interp_nearest(query_point)[0]
        else:
            # Si non-NaN, retourner le résultat linéaire
            return result_linear

class ZSlice:
    """Does not store the mesh locally to be light (not sure about that)."""
    def __init__(self, z=0.0, h=0.0, empty=False, **kwargs):
        self.empty = empty
        if empty:
            self.z = np.nan
            self.h = np.nan
        else:
            self.z = z
            self.h = h
        self.polygon = None  # Contour extérieur
        self.plane_walls = list()
        self.plane_grid = list()
        self.plane_full = list()
        self.max_offset_poly = None  # Polygon under the walls
        self.full_infill_polygon = None # Polygon of full infilled area

        self.lvlmesh = None  # trimesh of the levelset

        self.paths = None

        # Print settings
        # self.linewidth = 0.4  # mm
        self.density = 0.3  # 30%
        self.wall_count = 3
        self.linewidth = 0.4

        self.thickness = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _projected_poly(self, mesh: Mesh):
        if self.empty:
            self.polygon = sg.Polygon()
            return
        lvlset1 = mesh.get_one_levelset(self.h)
        if len(lvlset1[0]) == 0:
            self.empty = True
            self.polygon = sg.Polygon()
            return
        self.lvlmesh = lvlsetmesh(lvlset1)
        if self.lvlmesh.area == 0.0:
            self.empty = True
            self.polygon = sg.Polygon()
            return
        outlines = self.lvlmesh.outline().discrete

        contours = [loop[:, :2] for loop in outlines]  # ne garder que X, Y
        contours_sorted = sorted(contours, key=lambda c: sg.Polygon(c).area,
                                 reverse=True)  # For now, the longest is the outline --> CAN BE FALSE.

        exterior = contours_sorted[0]
        holes = contours_sorted[1:] if len(contours_sorted) > 1 else []

        self.polygon = sg.Polygon(shell=exterior, holes=holes)

    def _projected_poly_fromfiles(self, dicts, i):
        if self.empty:
            self.polygon = sg.Polygon()
            return
        faces = dicts["faces_" + str(i)]
        vertices = dicts["vertices_" + str(i)]
        self.lvlmesh = trimesh.Trimesh(vertices=vertices, faces = faces)
        if self.lvlmesh.area == 0.0:
            self.empty = True
            self.polygon = sg.Polygon()
            return
        outlines = self.lvlmesh.outline().discrete

        contours = [loop[:, :2] for loop in outlines]  # ne garder que X, Y
        contours_sorted = sorted(contours, key=lambda c: sg.Polygon(c).area,
                                 reverse=True)  # For now, the longest is the outline --> CAN BE FALSE.

        exterior = contours_sorted[0]
        holes = contours_sorted[1:] if len(contours_sorted) > 1 else []

        self.polygon = sg.Polygon(shell=exterior, holes=holes)

    def set_full_infill_area(self, polygon: sg.Polygon):
        self.full_infill_polygon = polygon

    def _walls(self):
        # Wall generation
        walls_per_level = []
        current = self.polygon

        # print("Zslice ", self.i_slice, " wall spaced:", self.linewidth)

        for i in range(self.wall_count):
            level_walls = []

            if current.geom_type == "Polygon":
                polys = [current]
            elif current.geom_type == "MultiPolygon":
                polys = list(current.geoms)
            else:
                polys = []

            for poly in polys:
                # mur extérieur
                level_walls.append(sg.Polygon(poly.exterior))
                # murs autour des trous
                for hole in poly.interiors:
                    level_walls.append(sg.Polygon(hole))

            walls_per_level.append(level_walls)

            self.max_offset_poly = current
            # current = current.buffer(-self.linewidth)
            current = current.buffer(-self.linewidth)
            if current.is_empty:
                break

        for level in walls_per_level:
                self.plane_walls.append(level)


        return walls_per_level

    def _grid(self):
        paths = []
        polygrid = self.max_offset_poly.difference(self.full_infill_polygon).buffer(-self.linewidth)

        # Normaliser en MultiPolygon
        if polygrid.geom_type == "Polygon":
            polygons = [polygrid]
        elif polygrid.geom_type == "MultiPolygon":
            polygons = list(polygrid.geoms)
        else:
            return []

        grid_spacing = (2 * self.linewidth) / self.density
        # print("Zslice ", self.i_slice, " grid spaced:", grid_spacing)

        for poly in polygons:
            minx, miny, maxx, maxy = poly.bounds

            # --- Vertical lines ---
            region_paths = []
            x = minx
            reverse = False
            while x <= maxx:
                line = sg.LineString([(x, miny), (x, maxy)])
                inter = poly.intersection(line)

                if not inter.is_empty:
                    if inter.geom_type == "MultiLineString":
                        segments = list(inter.geoms)
                    else:
                        segments = [inter]

                    # trier par y
                    segments.sort(key=lambda seg: seg.bounds[1])

                    for idx, seg in enumerate(segments):
                        coords = list(seg.coords)
                        if reverse:
                            coords = list(reversed(coords))

                        # ✅ nouveau chemin si plusieurs régions
                        if idx < len(region_paths):
                            link_seg = sg.LineString([region_paths[idx][-1], coords[0]])
                            if self.max_offset_poly.covers(link_seg):
                                # on peut continuer le chemin précédent
                                region_paths[idx].extend(coords)
                            else: # Il faut ouvrir un nouveau chemin au même index et refermer l'autre
                                region_paths.append(region_paths[idx])
                                region_paths[idx] = []
                                region_paths[idx].extend(coords)
                        else:
                            # sinon on ouvre un nouveau chemin
                            region_paths.append(coords)

                x += grid_spacing
                reverse = not reverse


            # --- Horizontal lines ---
            region_paths_horizontal = []
            y = miny
            while y < maxy:
                line = sg.LineString([(minx, y), (maxx, y)])
                inter = poly.intersection(line)
                if not inter.is_empty:
                    if inter.geom_type == "MultiLineString":
                        segments = list(inter.geoms)
                    else:
                        segments = [inter]

                    # trier par x pour ordre droite→gauche
                    segments.sort(key=lambda seg: seg.bounds[0])

                    for idx, seg in enumerate(segments):
                        coords = list(seg.coords)
                        if reverse:
                            coords = list(reversed(coords))

                        # si assez de "régions" existantes, append à la bonne
                        if idx < len(region_paths_horizontal):
                            link_seg = sg.LineString([region_paths_horizontal[idx][-1], coords[0]])
                            if self.max_offset_poly.covers(link_seg):
                                # on peut continuer le chemin précédent
                                region_paths_horizontal[idx].extend(coords)
                            else:  # Il faut ouvrir un nouveau chemin au même index et refermer l'autre
                                region_paths_horizontal.append(region_paths_horizontal[idx])
                                region_paths_horizontal[idx] = []
                                region_paths_horizontal[idx].extend(coords)
                        else:
                            # sinon on crée un nouveau chemin
                            region_paths_horizontal.append(coords)

                y += grid_spacing
                reverse = not reverse

            paths.extend(region_paths)
            paths.extend(region_paths_horizontal)

        self.plane_grid = paths
        return self.plane_grid

    def _full_layer(self, angle=0.0):
        all_paths = []
        polygrid = self.full_infill_polygon.buffer(-self.linewidth)


        # Normaliser en liste de polygones simples
        if polygrid.geom_type == "Polygon":
            polygons = [polygrid]
        elif polygrid.geom_type == "MultiPolygon":
            polygons = list(polygrid.geoms)
        else:
            raise ValueError("full_infill_polygon must be Polygon or MultiPolygon")

        for poly in polygons:
            rotated = sa.rotate(poly, angle, origin='centroid', use_radians=False)

            spacing = self.linewidth
            # print("Zslice ", self.i_slice, " full spaced:", spacing)
            minx, miny, maxx, maxy = rotated.bounds

            # Chaque "région" aura son propre chemin accumulé
            region_paths = []

            x = minx
            reverse = False
            while x <= maxx:
                line = sg.LineString([(x, miny), (x, maxy)])
                inter = rotated.intersection(line)

                if not inter.is_empty:
                    if inter.geom_type == "MultiLineString":
                        segments = list(inter.geoms)
                    else:
                        segments = [inter]

                    # trier par y pour ordre haut→bas
                    segments.sort(key=lambda seg: seg.bounds[1])

                    for idx, seg in enumerate(segments):
                        coords = list(seg.coords)
                        if reverse:
                            coords = list(reversed(coords))

                        # si assez de "régions" existantes, append à la bonne
                        if idx < len(region_paths):
                            region_paths[idx].extend(coords)
                        else:
                            # sinon on crée un nouveau chemin
                            region_paths.append(coords)

                x += spacing
                reverse = not reverse

            # Revenir à l’orientation initiale
            if angle != 0.0:
                new_region_paths = []
                for path in region_paths:
                    line = sg.LineString(path)
                    line = sa.rotate(line, -angle, origin=rotated.centroid, use_radians=False)
                    new_region_paths.append(list(line.coords))
                region_paths = new_region_paths

            all_paths.extend(region_paths)

        self.plane_full = all_paths
        return self.plane_full

    def layer_render(self):
        plt.plot(*self.polygon.exterior.xy, color='black')
        plt.fill(*self.polygon.exterior.xy, color='lightblue', alpha=0.5)
        plt.show()

    def render(self):
        fig, ax = plt.subplots()
        lines = []
        for p in self.paths:
            x = [pt[0] for pt in p]
            y = [pt[1] for pt in p]
            line, = ax.plot(x, y)
            lines.append(line)
        # Set line width
        for line in lines:
            line.set_linewidth(linewidth_mpl(self.linewidth, fig, ax))
        plt.show()

    def get_paths(self):
        """Return a list of p2d points in the order og the gcode"""
        all_paths = []
        # for polygon_paths in self.plane_walls:
        #     for polygon in polygon_paths:
        #         if polygon.area > 0.0:
        #             contourpath = []
        #             for x, y in polygon.exterior.coords:
        #                 contourpath.append((x, y))
        #             all_paths.append(contourpath)
        # ---- MURS regroupés
        # self.plane_walls = liste de niveaux -> chaque niveau est une liste de polygons
        n_levels = len(self.plane_walls)
        if n_levels > 0:
            # On suppose que la structure est parallèle : chaque "région" a ses offsets à chaque niveau
            n_regions = len(self.plane_walls[0])

            for region_idx in range(n_regions):
                chained = []
                last_point = None

                for level in range(n_levels):
                    if region_idx >= len(self.plane_walls[level]):
                        continue

                    poly = self.plane_walls[level][region_idx]
                    coords = list(poly.exterior.coords)

                    # Si on avait déjà un chemin en cours, choisir le point le plus proche
                    if last_point is not None:
                        start_idx = min(
                            range(len(coords)),
                            key=lambda i: (coords[i][0] - last_point[0]) ** 2 + (coords[i][1] - last_point[1]) ** 2
                        )
                        coords = coords[start_idx:] + coords[:start_idx]

                    chained.extend(coords)
                    last_point = coords[-1]

                if chained:
                    all_paths.append(chained)
        for l in self.plane_full:
            all_paths.append(l)
        for l in self.plane_grid:
            all_paths.append(l)

        self.paths = all_paths
        return all_paths

    def backprojection(self, max_step=2.0):
        if self.empty:
            return None
        # 4. Projection des points sur la surface 3D via ray tracing
        points = self.lvlmesh.vertices[:, :2]
        values = self.lvlmesh.vertices[:, 2]
        interp = LinearNDInterpolator(points, values)

        # --- Définir les bornes du maillage ---
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)

        def clip(x, y):
            """Ramène x,y dans les bornes de l'interpolateur"""
            x_clipped = min(max(x, xmin), xmax)
            y_clipped = min(max(y, ymin), ymax)
            return x_clipped, y_clipped

        self.get_paths()
        paths3d = []
        for path in self.paths:
            path3d = []
            for i, (x, y) in enumerate(path):
                # clip dans la zone de définition
                x_c, y_c = clip(x, y)
                z = float(interp(x_c, y_c))

                if i > 0:
                    # Vérifie la distance au point précédent, and add a point at the middle to split it and get new z
                    x_prev, y_prev = path3d[-1][:2]
                    dx, dy = x_c - x_prev, y_c - y_prev
                    dist = math.sqrt(dx ** 2 + dy ** 2)

                    if dist > max_step:
                        # Nombre de sous-segments nécessaires
                        n_steps = int(dist // max_step)
                        for k in range(1, n_steps + 1):
                            t = k / (n_steps + 1)
                            xi = x_prev + t * dx
                            yi = y_prev + t * dy
                            xic, yic = clip(xi, yi)
                            thickness = float(self.thickness(xic, yic))
                            zi = float(interp(xic, yic))
                            path3d.append((xic, yic, zi, thickness))

                # fallback si encore NaN (au cas où point est exactement hors convexe)
                if math.isnan(z):
                    # Cherche le plus proche point du maillage
                    idx = np.argmin(np.sum((points - [x_c, y_c]) ** 2, axis=1))
                    z = float(values[idx])
                    print(f"⚠️ interp hors domaine, fallback au plus proche voisin ({x},{y}) -> z={z}")

                t = float(self.thickness(x, y))
                path3d.append([x, y, z, t])  # Matematically not true BUT BUT BUT should be ok as the angle is not really big...
            paths3d.append(path3d)

        return paths3d


class FullSlicer:
    def __init__(self, mesh, dict_thickness=None):
        if type(mesh) == Mesh:
            self.mesh = mesh
            self.dicts = None
            self.zmin, self.zmax = self.mesh.triangle_mesh_vertice[:, 2].min(), self.mesh.triangle_mesh_vertice[:, 2].max()
            self.hmin, self.hmax = self.mesh.H.min(), self.mesh.H.max()
            self.h = self.z_to_h()

        elif type(mesh) == np.lib.npyio.NpzFile:
            self.mesh = None
            self.dicts = mesh
            keys = [k for k in self.dicts.keys()]
            self.zmin, self.zmax = self.dicts[keys[0]].min(), self.dicts[keys[0]].max()

            if dict_thickness is not None:
                self.thickness = list()
                for i in range(len(dict_thickness)):
                    self.thickness.append(Thickness(self.dicts, dict_thickness, i=i))


        self.slices = None
        self.all_3d_paths = None

        # SETTINGS
        self.n_solid_layers = 2  # Number of 100% surface infill layer before closing the shape
        self.maxlinewidth = max([t.max for t in self.thickness])  # mm
        self.linewidth = 0.4
        self.density = 0.3
        self.wall_count = 2

    def set_n_slices(self, n_slices):
        if self.mesh is not None and n_slices < self.mesh.min_NLEVELSETS:
            self.n_slices = self.mesh.min_NLEVELSETS
            print(n_slices, " is not enough slices, set to the minimum: ", self.n_slices)
        elif self.mesh is not None and n_slices > self.mesh.max_NLEVELSETS:
            self.n_slices = self.mesh.max_NLEVELSETS
            print(n_slices, " is too many slices, set to the maximum: ", self.n_slices)
        else:
            self.n_slices = n_slices

    def z_to_h(self):
        A = (self.hmax - self.hmin) / (self.zmax - self.zmin)
        B = self.hmin - A * self.zmin
        return lambda z: A * z + B

    def get_slice(self, i):
        if self.slices is None:
            self._init_slices()
        if i < 0 or i >= self.n_slices:
            zsl = ZSlice(empty=True)
            if self.mesh is not None:
                zsl._projected_poly(self.mesh)
            else:
                zsl._projected_poly_fromfiles(self.dicts, i)
            return zsl
        return self.slices[i]

    def _init_slices(self):
        self.slices = list()
        if self.mesh is not None:
            i_slice = 0
            for z in np.linspace(self.zmin, self.zmax, self.n_slices, endpoint=False):
                h = self.h(z)
                zslice = ZSlice(z, h,
                                linewidth=self.linewidth,
                                density=self.density,
                                wall_count=self.wall_count,
                                i_slice=i_slice,
                                thickness=self.thickness[i_slice])
                i_slice += 1
                if zslice.empty:
                    continue
                zslice._projected_poly(self.mesh)  # Mesh must be a Juanmesh
                self.slices.append(zslice)
        else:
            for i_slice in range(len(self.dicts)//2):
                zslice = ZSlice(empty=False,
                                i_slice=i_slice,
                                thickness=self.thickness[i_slice])
                zslice._projected_poly_fromfiles(self.dicts, i_slice)
                self.slices.append(zslice)

    def compute(self):
        self._init_slices()
        max_diff = self.linewidth * self.wall_count * self.n_solid_layers
        self.all_3d_paths = []
        for i, sl in enumerate(self.slices):
            polyzero = sl.polygon
            tiny_poly_zero = polyzero.buffer(-max_diff)

            poly_bottom = self.get_slice(i - self.n_solid_layers).polygon
            area_to_fill_bottom = tiny_poly_zero.difference(poly_bottom)   # Area to fill due to bottom lack

            poly_top = self.get_slice(i + self.n_solid_layers).polygon
            area_to_fill_top = tiny_poly_zero.difference(poly_top)  # Area to fill due to top lack

            polygon_to_fill = area_to_fill_bottom.union(area_to_fill_top)
            sl.set_full_infill_area(polygon_to_fill.buffer(max_diff - self.linewidth * (self.wall_count)))

            sl._walls()
            sl._full_layer()
            sl._grid()
            paths3D = sl.backprojection()
            print(i, sl.thickness.min, sl.thickness.max)
            if paths3D is not None:
                self.all_3d_paths.append(paths3D)

        print(len(self.all_3d_paths))

    def save(self):
        with open(r"D:\LUCAS\COURS\POSTDOC\Curv3D\Curv3D\src\geometries\reparation\paths.json", "w") as f:
            json.dump(self.all_3d_paths, f)




if __name__ == "__main__":
    # mesh = importmesh()
    dict_isosurfaces, dict_thikness = import_levelsets()
    n_slices = len(dict_isosurfaces) // 2
    slicer = FullSlicer(dict_isosurfaces, dict_thickness=dict_thikness)
    slicer.set_n_slices(n_slices)
    slicer.compute()
    slicer.save()
    print()