import fullcontrol as fc
import numpy as np
import extragcode
import json
from collections import deque
import mmap
import math
import os

class GcodeGenerator:
    """
    Classe qui génère des fichiers G-code à partir d'une liste de points 3D,
    sans utiliser la librairie fullcontrol. Le calcul de l'extrusion (E)
    est effectué manuellement pour chaque segment.
    """

    def __init__(self):
        # Paramètres d'extrusion et de filament
        self.filament_diameter = 1.75
        # Aire de la section transversale du filament pour le calcul de E
        self.filament_area = math.pi * (self.filament_diameter / 2) ** 2

        # Vitesses (converties en mm/min pour le G-code F)
        # Basé sur les défauts de l'ancienne VariableExtrusionPrinter (3000/6000)
        self.speed = 3000  # Vitesse d'extrusion (F value in G1, mm/min)
        self.travel_speed = 6000  # Vitesse de déplacement (F value in G0/G1, mm/min)
        self.linewidth = 0.4

        # Dimensions et offsets de la plateforme (Ender 5 exemple)
        self.avg_bedx = 220.0 / 2
        self.avg_bedy = 220.0 / 2

        # Centre de la boîte englobante de l'objet de référence (FreeCAD/Slicer data)
        # (FreeCAD.ActiveDocument.reference_volume.Mesh.BoundBox.Center)
        # self.avg_bbx = 35.638626646250486
        # self.avg_bby = 25.35800564289093
        self.avg_bbx = 13.920511364936829
        self.avg_bby = -0.30611419677734375
        # self.avg_bbx = 0
        # self.avg_bby = 0

        # Offsets pour Cura slicer (ajustement X/Y/Z)
        self.xoffset = self.avg_bedx - self.avg_bbx
        self.yoffset = self.avg_bedy - self.avg_bby
        # self.zoffset = - 6.80933
        self.zoffset = 0

        # Hauteur de sécurité Z pour les déplacements rapides (Z max de l'objet + offset + marge)
        # (FreeCAD.ActiveDocument.reference_volume.Mesh.BoundBox.ZMax)
        self.zsafe = 20.850784301757812 + self.zoffset + 1.0

        # G-code de base (début et fin de tranche du slicer)
        self.basis_gcode = str()

        # self.steps est maintenant une liste de chaînes G-code
        self.steps = list()

        self.current_extruder = 'T0'

    def _transition_path(self, p1: np.ndarray, p2: np.ndarray) -> list[str]:
        """
        Génère les lignes G-code pour un mouvement de transition (voyage)
        entre la fin d'un chemin (p1) et le début d'un autre (p2).

        :param p1: Point de départ [x, y, z, h]
        :param p2: Point d'arrivée [x, y, z, h]
        :return: Liste de chaînes G-code
        """
        gcode_lines = []

        gcode_lines.append(
            f"; TRANSITION START (From {p1[0]:.3f}, {p1[1]:.3f}, {p1[2]:.3f} to {p2[0]:.3f}, {p2[1]:.3f}, {p2[2]:.3f})")

        # Rétraction (simulée par FullControl's Extruder(on=False) - pas d'E dans le G0)
        # On utilise une rétraction explicite si nécessaire, mais ici on se contente du mouvement G0

        # 1. Déplacement sécurisé en Z (montée à z_safe)
        z_safe = self.zsafe

        # Déplacement en X/Y du point de départ à Z_safe
        gcode_lines.append(
            f"G0 X{p1[0] + self.xoffset:.3f} Y{p1[1] + self.yoffset:.3f} Z{z_safe:.3f} F{self.travel_speed:.0f}")

        # Déplacement en X/Y vers le point d'arrivée à Z_safe
        gcode_lines.append(
            f"G0 X{p2[0] + self.xoffset:.3f} Y{p2[1] + self.yoffset:.3f} Z{z_safe:.3f} F{self.travel_speed:.0f}")

        # 2. Retour à la hauteur de la couche suivante
        # La hauteur Z cible est Z_point + Z_offset + Hauteur_locale (p2[3])
        target_z = p2[2] + self.zoffset + p2[3]
        gcode_lines.append(
            f"G0 X{p2[0] + self.xoffset:.3f} Y{p2[1] + self.yoffset:.3f} Z{target_z:.3f} F{self.travel_speed:.0f}")

        # Dérétraction (simulée par Extruder(on=True) - pas d'E dans le G0)

        gcode_lines.append(f"; TRANSITION END")
        return gcode_lines

    def _change_layer(self) -> str:
        """Génère la ligne de commentaire pour un changement de couche."""
        return ";LAYER_CHANGE"

    def _change_extruder(self) -> str:
        """Génère les lignes G-code pour changer d'extrudeur (T0 <-> T1)."""
        if self.current_extruder == 'T0':
            new_extruder = "T1"
        elif self.current_extruder == 'T1':
            new_extruder = "T0"
        else:
            raise ValueError('Extruder must be T0 or T1.')

        # Note: Les commandes M104/M109 sont spécifiques à votre machine (K1)
        gcode = ("; CHANGE EXTRUDER from " + self.current_extruder + " to " + new_extruder + "\n"
                                                                                             "G92 E0\n"  # Réinitialiser l'extrusion pour le nouvel extrudeur
                 + new_extruder + "\n"
                                  "G92 E0\n"  # Réinitialiser à nouveau l'extrusion (précautions)
                                  "M109 S200\n"  # Attendre que la nouvelle buse atteigne 200°C (exemple)
                                  "M104 " + self.current_extruder + " S170\n"  # Refroidir l'ancienne buse
                                                                    "M104 S210\n")  # Chauffer la nouvelle buse à 210°C (exemple)

        self.current_extruder = new_extruder
        return gcode

    # Renommage de paths_into_steps2 en generate_paths_gcode pour une meilleure clarté
    def generate_paths_gcode(self, paths: np.ndarray):
        """
        Génère les lignes G-code à partir des chemins 3D, calculant l'extrusion E.

        :param paths: Liste de couches (list) contenant des chemins (list) de points [x, y, z, h]
        :return: void (ajoute le G-code à self.steps)
        """
        self.steps.append("\nM83; relative extrusion")

        self.steps.append(self._change_layer())
        self.steps.append(
            f"G0 X{self.xoffset:.3f} Y{self.yoffset:.3f} Z{self.zsafe:.3f} F{self.travel_speed:.0f}")


        for l, layer in enumerate(paths):
            for i, path in enumerate(layer):
                if len(path) < 2: continue

                # --- 1. Premier Point (Voyage) ---
                first_point = path[0]
                curr_h = first_point[3]  # Hauteur locale (Line Width/Height)

                # Coordonnées du point de départ ajustées
                start_x = first_point[0] + self.xoffset
                start_y = first_point[1] + self.yoffset
                start_z = first_point[2] + self.zoffset + curr_h

                # Le premier point est un mouvement de voyage (G0)
                self.steps.append(
                    f"G0 X{start_x:.3f} Y{start_y:.3f} Z{start_z:.3f} F{self.travel_speed:.0f}"
                )

                # Initialiser les coordonnées précédentes pour le calcul de distance du segment
                prev_x, prev_y, prev_z = start_x, start_y, start_z

                # --- 2. Points Suivants (Extrusion G1) ---
                for point in path[1:]:
                    # Coordonnées actuelles
                    curr_h = point[3]  # Hauteur locale
                    curr_x = point[0] + self.xoffset
                    curr_y = point[1] + self.yoffset
                    curr_z = point[2] + self.zoffset + curr_h

                    # CALCUL DE LA DISTANCE (Longueur du segment 3D)
                    dist = math.sqrt(
                        (curr_x - prev_x) ** 2 +
                        (curr_y - prev_y) ** 2 +
                        (curr_z - prev_z) ** 2
                    )

                    # CALCUL DU VOLUME EXTRUDÉ (Volume = Longueur * Hauteur * Largeur)
                    # On suppose Largeur = Hauteur = curr_h (comme dans votre logique fullcontrol)
                    segment_vol = dist * curr_h * self.linewidth

                    # CALCUL DE L'EXTRUSION E (Longueur de filament)
                    e_step = segment_vol / self.filament_area

                    # Créer la ligne G-code d'extrusion (G1)
                    self.steps.append(
                        f"G1 X{curr_x:.3f} Y{curr_y:.3f} Z{curr_z:.3f} E{e_step:.5f} F{self.speed:.0f}"
                    )

                    # Mettre à jour "prev" pour la prochaine itération
                    prev_x, prev_y, prev_z = curr_x, curr_y, curr_z

                # Transition entre chemins dans la même couche
                if i < len(layer) - 1:
                    start, end = layer[i][-1], layer[i + 1][0]
                    self.steps.extend(self._transition_path(start, end))

            # Transition entre couches (sauf la dernière)
            if l < len(paths) - 1:
                start, end = paths[l][-1][-1], paths[l + 1][0][0]
                self.steps.extend(self._transition_path(start, end))

            self.steps.append(self._change_layer())

    def add_existing_gcode3(self, gcode_file, slicer="cura"):
        # La méthode mmap est conservée (la plus rapide et recommandée)
        if slicer == "cura":
            triggers = [b";TYPE:SKIRT", b"M140 S0"]
        elif slicer == "prusa":
            triggers = [b";LAYER_CHANGE", b";TYPE:Custom"]
        else:
            raise AssertionError("Slicer must be 'cura' or 'prusa'")

        try:
            with open(gcode_file, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

                start_pos = mm.find(triggers[0])
                if start_pos == -1:
                    raise ValueError("Trigger de début non trouvé")

                # rfind cherche de la fin vers le début, ce qui est correct pour le trigger de fin
                end_pos_start_of_line = mm.rfind(triggers[1])
                if end_pos_start_of_line == -1:
                    raise ValueError("Trigger de fin non trouvé")

                # Trouver la fin de la ligne du trigger de fin pour l'exclure
                temp_end_pos = end_pos_start_of_line + len(triggers[1])
                line_end_pos = mm.find(b'\n', temp_end_pos)  # Trouver le prochain saut de ligne

                if line_end_pos == -1:  # Si c'est la fin du fichier
                    line_end_pos = len(mm)

                # Extraire le segment du fichier
                mm.seek(start_pos)
                gcode_bytes = mm.read(line_end_pos - start_pos)

                # Convertir en str (UTF-8 typique des GCode)
                self.basis_gcode = gcode_bytes.decode("utf-8", errors="ignore")

                mm.close()
        except FileNotFoundError:
            print(f"ERROR: Le fichier G-code de base '{gcode_file}' est introuvable.")
        except Exception as e:
            print(f"ERROR reading G-code: {e}")

    def save_gcode(self, filename: str = r"D:\LUCAS\COURS\POSTDOC\Curv3D\Curv3D\src\geometries\reparation\output.gcode"):
        """Sauvegarde le G-code complet dans un fichier."""
        # Le G-code est maintenant une simple concaténation des chaînes
        gcode = extragcode.initgcode_K1
        gcode += self.basis_gcode
        # Utiliser join pour combiner toutes les lignes générées dans self.steps
        gcode += "\n".join(self.steps) + "\n"
        gcode += extragcode.endgcode_K1

        # Assurer que le répertoire existe
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Sauvegarder
        with open(filename, "w") as f:
            f.write(gcode)

        print("✅ G-code généré dans ", filename)


if __name__ == "__main__":
    # --- Exemple d'utilisation (simulé) ---

    # Créer un chemin simulé si le fichier JSON n'existe pas
    paths_file = r"D:\LUCAS\COURS\POSTDOC\Curv3D\Curv3D\src\geometries\reparation\paths.json"
    with open(paths_file, "r") as f:
        # Paths doit être une liste de listes de listes de points [x, y, z, h]
        # Ex: [[[x1, y1, z1, h1], [x2, y2, z2, h2], ...], [[...], ...]]
        paths = json.load(f)
        print(f"Paths chargés depuis {paths_file}")


    # Filename of basis shape gcode from Prusa
    basis_gcode_file = r"./geometries/reparation/reference_volume.gcode"
    generator = GcodeGenerator()

    # Utilisation de la méthode mmap pour la lecture du G-code de base
    generator.add_existing_gcode3(basis_gcode_file, slicer="prusa")

    # Génération du G-code des chemins (anciennement paths_into_steps2)
    generator.generate_paths_gcode(paths)

    # Sauvegarde du fichier G-code final
    output_file = r"D:\LUCAS\COURS\POSTDOC\Curv3D\Curv3D\src\geometries\reparation\complete_volume.gcode"
    generator.save_gcode(output_file)
    print(f"Vérifiez le G-code généré à: {output_file}")
