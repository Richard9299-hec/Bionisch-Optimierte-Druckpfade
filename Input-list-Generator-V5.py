import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
from scipy.spatial import Delaunay

###############################################################################
# Alpha Shape berechnen
###############################################################################
def alpha_shape(points, alpha):
    if len(points) < 4:
        return Polygon(points).convex_hull  # Notfalls konvexe HÃ¼lle

    tri = Delaunay(points)
    triangles = points[tri.simplices]

    def triangle_area(a, b, c):
        # Explizites 2D Kreuzprodukt (NumPy 2.0 kompatibel!)
        return 0.5 * np.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def circum_radius(a, b, c, area):
        edges = [np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a)]
        return np.prod(edges) / (4 * area) if area > 0 else np.inf

    filtered = []
    for a, b, c in triangles:
        area = triangle_area(a, b, c)
        R = circum_radius(a, b, c, area)
        if R < 1 / alpha:
            filtered.append(Polygon([a, b, c]))

    return unary_union(filtered)

###############################################################################
# Gitterpunkte innerhalb der FlÃ¤che generieren
###############################################################################
def generate_grid_within_shape(shape, n_points, min_x, max_x, min_y, max_y):
    area = shape.area
    spacing = np.sqrt(area / n_points)
    x_vals = np.arange(min_x, max_x, spacing)
    y_vals = np.arange(min_y, max_y, spacing)
    xx, yy = np.meshgrid(x_vals, y_vals)
    candidates = np.column_stack([xx.ravel(), yy.ravel()])

    inside = []
    for pt in MultiPoint(candidates).geoms:
        if shape.contains(pt):
            inside.append(pt.coords[0])
    return np.array(inside)

###############################################################################
# GleichmÃ¤ÃŸig n Punkte auswÃ¤hlen
###############################################################################
def select_n_points(points, n):
    if len(points) <= n:
        return points
    indices = np.linspace(0, len(points) - 1, n, dtype=int)
    return points[indices]

###############################################################################
# Hauptfunktion
###############################################################################
def main():
    # Parameter
    alpha = 0.8   # Alpha Shape Parameter
    s = 0.8       # Abstand zur Grenze
    n = 2000      # Zielanzahl an Punkten

    # Einlesen der Rohdaten (NLIST.txt)
    raw_points = []
    with open("NLIST.txt", "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            try:
                x = float(parts[1])
                y = float(parts[2])
                raw_points.append([x, y])
            except ValueError:
                continue

    points = np.array(raw_points)
    if len(points) < 3:
        raise ValueError("Zu wenige gÃ¼ltige Punkte in 'NLIST.txt' gefunden!")

    # Alpha Shape und innerer Bereich
    shape = alpha_shape(points, alpha)
    inner_shape = shape.buffer(-s)
    if inner_shape.is_empty:
        raise ValueError("Der geschrumpfte Bereich ist leer. ÃœberprÃ¼fen Sie den Wert von s.")

    # Gitter generieren und Punkte auswÃ¤hlen
    min_x, min_y, max_x, max_y = inner_shape.bounds
    candidates = generate_grid_within_shape(inner_shape, n * 3, min_x, max_x, min_y, max_y)
    selected = select_n_points(candidates, n)

    # Speichern der selektierten Punkte
    concave_file = "concave_hull_points.txt"
    np.savetxt(concave_file, selected, header="X Y", fmt="%.4f")
    print(f"{len(selected)} Punkte in '{concave_file}' gespeichert.")

    # Generiere output.txt im gewÃ¼nschten Format
    output_file = "output.txt"
    with open(output_file, "w") as outf:
        for x, y in selected:
            line1 = f"XS0_={x:.4f}     $ YS0_={y:.4f}   $ Ls_=.1   $ NS_=100     ! NS_ < 1000"
            line2 = "/INPUT,SKRIPT_Spannungslinien,txt,C:\\Users\\rrrsc\\Documents\\Wichtig\\Hector\\Koop\\Spannugsfluss\\"
            outf.write(line1 + "\n")
            outf.write(line2 + "\n\n")

    # Ausgabe des Inhalts von output.txt
    with open(output_file, "r") as outf:
        content = outf.read()
    print(content)

    # Visualisierung
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=10, label='Originale Punkte')
    if shape.geom_type == 'Polygon':
        x_sh, y_sh = shape.exterior.xy
        plt.plot(x_sh, y_sh, 'k--', label='Alpha Shape')
    else:
        for poly in shape.geoms:
            x_sh, y_sh = poly.exterior.xy
            plt.plot(x_sh, y_sh, 'k--')

    plt.scatter(selected[:, 0], selected[:, 1], color='red', s=30, label='Neu verteilte Punkte')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('GleichmÃ¤ÃŸig verteilte Punkte (Abstand s zur Grenze)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
