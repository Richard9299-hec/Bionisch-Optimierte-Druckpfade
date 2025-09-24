# --------------------------------------------------------------------------
# Imports & Konfigurierbare Variablen
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import CubicSpline, PchipInterpolator
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from numba import njit
from multiprocessing import Pool
from tkinter import TclError
import math

# === Plot-Einstellungen ===
PLOT_TANGENTS                      = 0     # Tangenten im Final-Plot anzeigen
AUTO_CONNECT_DEBUG                 = 0     # Debug-Plot fÃ¼r jede Verbindung ein/aus

# === Globaler Punktfilter ===
START_POINT_TOLERANCE              = 1e-100 # Toleranz, um Startpunkte wiederzuerkennen
PROXIMITY_RADIUS                   = 0.08   # minimaler Abstand im globalen Filter

# === Segmentierung & Dichte ===
MIN_POINTS_PER_SEGMENT             = 5      # minimale Anzahl Punkte pro Segment
MIN_POINT_DENSITY                  = 0.95   # minimale (Punkte/LÃ¤nge)-Dichte
OUTPUT_POINT_DENSITY               = 2      # AuflÃ¶sung fÃ¼r Splines und funktionalen Filter
X_SHIFT                            = -1.4   # Verschiebung aller X-Koordinaten

# === Konflikt-Split ===
CONFLICT_SPLIT_RADIUS              = 0.35   # Radius fÃ¼r Konflikt-Erkennung
MAX_EXCLUDE_OVERLAP_RATIO          = 1.0    # maximaler Anteil gemeinsamer X-Range beim funktionalen Filter

# === Funktionaler Filter ===
MIN_FUNCTION_DISTANCE              = 0.1    # max. vertikaler Abstand im funktionalen Filter
MIN_FUNCTION_VERTICAL_X_RANGE      = 2.5    # minimaler Î”X, unterhalb dessen Segmente von funktionaler Filterung ausgenommen werden

# === Alpha-Shapeâ€“Filter ===
ALPHA_SHAPE_ALPHA                  = 0.8    # Î± fÃ¼r Alpha-Shape
ALPHA_BUFFER_MARGIN                = 0      # Puffer innerhalb der Alpha-Shape
EXEMPT_X_MIN, EXEMPT_X_MAX         = 0.0, 0.0
EXEMPT_Y_MIN, EXEMPT_Y_MAX         = 0.0, 0.0  # Bereich, der immer exempt bleibt

# === Monotonie-Split in X ===
SPLIT_NON_MONOTONIC_X              = True   # Monotonie-Split aktivieren
MONOTONIC_SPLIT_TOL                = 0.1    # erhÃ¶hte Toleranz fÃ¼r near-vertikale Segmente

# === Auto-Connectâ€“Einstellungen ===
TAN_LENGTH                         = 3.5    # LÃ¤nge der gezeichneten Tangenten
CONNECT_AUTO                       = True   # Automatische Verbindung aktivieren
SLOPE_DIFF_TOLERANCE               = 15.0   # max. Unterschied der Tangenten-Steigungen
AUTO_CONNECT_CHECK_TANGENT_INTERSECT = True  # Kriterium 1: Tangenten schneiden sich
AUTO_CONNECT_REQUIRE_TAN_BOUNDED   = True   # Kriterium 2: Schnittpunkt x zwischen Tangenten-Startpunkten
AUTO_CONNECT_CHECK_SLOPE           = True   # Kriterium 3: Steigungsdifferenz innerhalb Toleranz
AUTO_CONNECT_CHECK_CROSSING        = False  # Kriterium 4: keine Kreuzung mit anderen Segmenten
AUTO_CONNECT_CHECK_MIN_DISTANCE    = True   # Kriterium 5: minimale Verbindungsdistanz wÃ¤hlen
AUTO_CONNECT_CHECK_REVERSAL        = True   # Kriterium 6: Reihenfolge-Anpassung (Umdrehen)

# Neuer Parameter fÃ¼r FÃ¼hler an den Tangenten
TANGENT_FUELLER_ANGLE              = 5.0    # Winkel in Grad, um den Tangenten zur SchnittprÃ¼fung â€žgefÃ¤chertâ€œ werden

# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------
def merge_intervals(intervals):
    if not intervals:
        return []
    ints = sorted(intervals, key=lambda x: x[0])
    merged = [list(ints[0])]
    for s, e in ints[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged

def complement_intervals(start, end, intervals):
    if not intervals:
        return [(start, end)]
    comp, prev = [], start
    for s, e in sorted(intervals, key=lambda x: x[0]):
        if prev < s:
            comp.append((prev, s))
        prev = max(prev, e)
    if prev < end:
        comp.append((prev, end))
    return comp

def read_nlist(path):
    coords = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    coords.append((float(parts[1]), float(parts[2])))
                except:
                    pass
    return np.array(coords)

def split_segments_by_monotonicity(segs, tol=1e-8):
    new = []
    for xs, ys in segs:
        x = np.array(xs)
        if np.ptp(x) < tol:
            new.append((xs, ys)); continue
        y = np.array(ys)
        if len(x) < MIN_POINTS_PER_SEGMENT:
            continue
        dir = None
        for i in range(1, len(x)):
            d = x[i] - x[i-1]
            if abs(d) > tol:
                dir = 1 if d > 0 else -1
                break
        if dir is None:
            new.append((xs, ys)); continue
        start = 0
        for i in range(1, len(x)):
            diff = x[i] - x[i-1]
            mono = (dir==1 and diff>tol) or (dir==-1 and diff<-tol)
            if not mono:
                partx, party = x[start:i].tolist(), y[start:i].tolist()
                if len(partx)>=MIN_POINTS_PER_SEGMENT:
                    new.append((partx, party))
                start = i-1
                dir = None
                for k in range(i, len(x)):
                    d2 = x[k]-x[k-1]
                    if abs(d2)>tol:
                        dir = 1 if d2>0 else -1
                        break
                if dir is None:
                    break
        if start < len(x)-1 and dir is not None:
            partx, party = x[start:].tolist(), y[start:].tolist()
            if len(partx)>=MIN_POINTS_PER_SEGMENT:
                new.append((partx, party))
    return new

def close_all():
    try: plt.close('all')
    except: pass

def plot_splines(segs, dens, title='Splines'):
    close_all()
    fig, ax = plt.subplots()
    for xs, ys in segs:
        x, y = np.array(xs), np.array(ys)
        d = np.hypot(np.diff(x), np.diff(y))
        t = np.insert(np.cumsum(d), 0, 0.0)
        if t[-1]==0:
            t_norm, num = t, len(t)
        else:
            t_norm = t/t[-1]
            num    = max(int(t[-1]*dens), len(t))
        tnew = np.linspace(0,1,num)
        spx  = CubicSpline(t_norm, x)
        spy  = CubicSpline(t_norm, y)
        ax.plot(spx(tnew), spy(tnew), linewidth=1)
    ax.set(xlabel='X', ylabel='Y', title=title)
    ax.grid(True)
    try: plt.show()
    except TclError: pass

def filter_segments_by_polygon(segs, poly):
    out = []
    for xs, ys in segs:
        cx, cy = [], []
        for x, y in zip(xs, ys):
            if poly.contains(Point(x, y)):
                cx.append(x); cy.append(y)
            else:
                if len(cx)>=MIN_POINTS_PER_SEGMENT:
                    out.append((cx, cy))
                cx, cy = [], []
        if len(cx)>=MIN_POINTS_PER_SEGMENT:
            out.append((cx, cy))
    return out

def filter_segments_by_function_distance(segs, thr, dens):
    # Teile in nahezu-vertikale (ausgenommen) und normale Segmente
    vertical_exempt = []
    normal_segs = []
    for xs, ys in segs:
        x_arr = np.array(xs)
        if abs(x_arr[-1] - x_arr[0]) < MIN_FUNCTION_VERTICAL_X_RANGE:
            vertical_exempt.append((xs, ys))
        else:
            normal_segs.append((xs, ys))
    # Wenn keine zu filtern, gib komplette Liste zurÃ¼ck
    if not normal_segs:
        return segs

    # Baue Interpolatoren fÃ¼r normale Segmente
    spl, inter = [], []
    for xs, ys in normal_segs:
        idx  = np.argsort(xs)
        x, y = np.array(xs)[idx], np.array(ys)[idx]
        spl.append(PchipInterpolator(x, y))
        inter.append((x[0], x[-1]))

    # Finde Ã¼berlappende Intervalle mit zu kleinem Abstand
    excl = {i: [] for i in range(len(spl))}
    for i in range(len(spl)):
        for j in range(i+1, len(spl)):
            a, b = max(inter[i][0], inter[j][0]), min(inter[i][1], inter[j][1])
            if b <= a:
                continue
            num   = max(200, int((b - a) * dens * 5))
            xsamp = np.linspace(a, b, num)
            dist  = np.abs(spl[i](xsamp) - spl[j](xsamp))
            idxs  = np.where(dist < thr)[0]
            if idxs.size == 0:
                continue
            groups = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
            li, lj = inter[i][1] - inter[i][0], inter[j][1] - inter[j][0]
            for g in groups:
                sx, ex = xsamp[g[0]], xsamp[g[-1]]
                if (ex - sx) / (b - a + 1e-16) > MAX_EXCLUDE_OVERLAP_RATIO:
                    continue
                excl_idx = i if li < lj else j
                excl[excl_idx].append((sx, ex))

    # Erstelle neue Segmente aus erlaubten Intervallen
    new = []
    for k, interp in enumerate(spl):
        s0, s1 = inter[k]
        allowed = complement_intervals(s0, s1, merge_intervals(excl[k]))
        if not allowed:
            # kein Intervall Ã¼brig â†’ Originalsegment beibehalten
            xs_o, ys_o = normal_segs[k]
            new.append((xs_o, ys_o))
            continue
        for a, b in allowed:
            xs_new = np.arange(a, b + 1.0 / dens, 1.0 / dens)
            ys_new = interp(xs_new)
            if len(xs_new) >= MIN_POINTS_PER_SEGMENT:
                new.append((xs_new.tolist(), ys_new.tolist()))

    # FÃ¼ge die ausgenommenen fast-vertikalen Segmente wieder hinzu
    new.extend(vertical_exempt)
    return new

def alpha_shape(points, alpha):
    if len(points)<4:
        return Polygon(points).convex_hull
    tri  = Delaunay(points)
    tris = points[tri.simplices]
    a,b,c= tris[:,0],tris[:,1],tris[:,2]
    area = 0.5*np.abs((b[:,0]-a[:,0])*(c[:,1]-a[:,1])
                    - (b[:,1]-a[:,1])*(c[:,0]-a[:,0]))
    dab = np.linalg.norm(a-b,axis=1)
    dbc = np.linalg.norm(b-c,axis=1)
    dca = np.linalg.norm(c-a,axis=1)
    radii = np.where(area>0,(dab*dbc*dca)/(4*area),np.inf)
    return unary_union([Polygon(tr) for tr,r in zip(tris,radii) if r<1/alpha])

def _filter_one_segment_alpha(xs, ys, inner):
    cx, cy, out = [], [], []
    for x,y in zip(xs, ys):
        exempt = (EXEMPT_X_MIN <= x <= EXEMPT_X_MAX
               and EXEMPT_Y_MIN <= y <= EXEMPT_Y_MAX)
        if exempt or inner.contains(Point(x,y)):
            cx.append(x); cy.append(y)
        else:
            if len(cx)>=MIN_POINTS_PER_SEGMENT:
                out.append((cx,cy))
            cx, cy = [], []
    if len(cx)>=MIN_POINTS_PER_SEGMENT:
        out.append((cx,cy))
    return out

def filter_segments_by_alpha_parallel(segs, inner):
    with Pool() as pool:
        res = pool.starmap(_filter_one_segment_alpha,
                           [(xs,ys,inner) for xs,ys in segs])
    return [seg for sub in res for seg in sub]

@njit
def global_filter_points_numba(xc, yc, sp_array, min_dist):
    fx, fy = [], []
    for i in range(xc.shape[0]):
        x,y = xc[i], yc[i]
        is_start=False
        for sx,sy in sp_array:
            if abs(x-sx)<START_POINT_TOLERANCE and abs(y-sy)<START_POINT_TOLERANCE:
                is_start=True; break
        if is_start:
            fx.append(x); fy.append(y)
        else:
            keep=True
            for k in range(len(fx)):
                if (fx[k]-x)**2+(fy[k]-y)**2<min_dist**2:
                    keep=False; break
            if keep:
                fx.append(x); fy.append(y)
    return fx, fy

def read_coordinates(path):
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts)>=2:
                try:
                    x,y = map(float, parts[:2])
                    xs.append(x); ys.append(y)
                except: pass
    return xs, ys

def read_start_points(path):
    pts=[]
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                try:
                    pts.append(tuple(map(float,line.split()[:2])))
                except: pass
    return pts

def is_start_point(x,y,starts):
    return any(abs(x-sx)<START_POINT_TOLERANCE
           and abs(y-sy)<START_POINT_TOLERANCE
           for sx,sy in starts)

def segment_filtered_points(fx, fy, starts, min_pts):
    segs=[]
    inds=[i for i,(x,y) in enumerate(zip(fx,fy)) if is_start_point(x,y,starts)]
    if not inds:
        if len(fx)>=min_pts:
            segs.append((fx,fy))
    else:
        for k,s in enumerate(inds):
            e = inds[k+1] if k+1<len(inds) else len(fx)
            sx,sy = fx[s:e], fy[s:e]
            if len(sx)>=min_pts:
                segs.append((sx,sy))
    return segs

def filter_segments_by_density(segs, min_den):
    out=[]
    for xs,ys in segs:
        length = np.sum(np.hypot(np.diff(xs),np.diff(ys)))
        if length>0 and len(xs)/length>=min_den:
            out.append((xs,ys))
    return out

def split_segments_by_conflict(segs, radius):
    lengths=[len(xs) for xs,ys in segs]
    coords, sids, idxs = [],[],[]
    for sid,(xs,ys) in enumerate(segs):
        for i,(x,y) in enumerate(zip(xs,ys)):
            coords.append((x,y)); sids.append(sid); idxs.append(i)
    if not coords: return []
    coords = np.array(coords); sids = np.array(sids); idxs = np.array(idxs)
    tree = cKDTree(coords)
    nbrs = tree.query_ball_tree(tree, r=radius)
    conflict = np.zeros(len(coords),dtype=bool)
    for i,neigh in enumerate(nbrs):
        for j in neigh:
            if j<=i: continue
            si,sj = sids[i], sids[j]
            if si==sj: continue
            if lengths[si]<lengths[sj]:
                conflict[i]=True
            elif lengths[sj]<lengths[si]:
                conflict[j]=True
    new=[]
    for sid in np.unique(sids):
        mask   = (sids==sid)
        ids_sid= np.where(mask)[0]
        ordered= ids_sid[np.argsort(idxs[mask])]
        tx,ty  = [],[]
        for ix in ordered:
            x,y = coords[ix]
            if conflict[ix]:
                if len(tx)>=MIN_POINTS_PER_SEGMENT:
                    new.append((tx,ty))
                tx,ty = [],[]
            else:
                tx.append(x); ty.append(y)
        if len(tx)>=MIN_POINTS_PER_SEGMENT:
            new.append((tx,ty))
    return new

# --------------------------------------------------------------------------
# plot_segments & write_segments
# --------------------------------------------------------------------------
def write_segments(segs, path):
    with open(path, 'w') as f:
        for i,(xs,ys) in enumerate(segs,1):
            f.write(f'# Segment {i}\n')
            for x,y in zip(xs,ys):
                f.write(f'{x:.1f} {y:.1f}\n')
            f.write('\n')

def plot_segments(segs, title='Segmente'):
    close_all()
    fig, ax = plt.subplots()
    for xs, ys in segs:
        ax.plot(xs, ys, marker='o', markersize=1)
        if PLOT_TANGENTS and len(xs)>=2:
            # zeichne Tangenten am Anfang/Ende
            start_pt, next_pt = (xs[0], ys[0]), (xs[1], ys[1])
            vx,vy = start_pt[0]-next_pt[0], start_pt[1]-next_pt[1]
            norm  = math.hypot(vx,vy)+1e-16
            ux,uy = vx/norm, vy/norm
            tan_s = (start_pt[0]+ux*TAN_LENGTH, start_pt[1]+uy*TAN_LENGTH)
            ax.plot([start_pt[0],tan_s[0]], [start_pt[1],tan_s[1]], 'k-', linewidth=1)
            end_pt, prev_pt   = (xs[-1],ys[-1]), (xs[-2],ys[-2])
            vx,vy = end_pt[0]-prev_pt[0], end_pt[1]-prev_pt[1]
            norm  = math.hypot(vx,vy)+1e-16
            ux,uy = vx/norm, vy/norm
            tan_e = (end_pt[0]+ux*TAN_LENGTH, end_pt[1]+uy*TAN_LENGTH)
            ax.plot([end_pt[0],tan_e[0]], [end_pt[1],tan_e[1]], 'k-', linewidth=1)
    ax.set(xlabel='X', ylabel='Y', title=title)
    ax.grid(True)
    try: plt.show()
    except: pass

# --------------------------------------------------------------------------
# Angepasste Auto-Connect mit Tangenten-FÃ¼hler
# --------------------------------------------------------------------------
def auto_connect_segments(segs, slope_tol=SLOPE_DIFF_TOLERANCE, tan_len=TAN_LENGTH):
    def tangent(pt1, pt2, length):
        vx, vy = pt2[0]-pt1[0], pt2[1]-pt1[1]
        norm    = math.hypot(vx, vy)+1e-16
        ux, uy  = vx/norm, vy/norm
        return (pt2[0]+ux*length, pt2[1]+uy*length), (ux, uy)

    def rotate(u, angle_rad):
        ux, uy = u
        ca, sa = math.cos(angle_rad), math.sin(angle_rad)
        return (ux*ca - uy*sa, ux*sa + uy*ca)

    def point_segment_distance(P, A, B):
        ap = (P[0]-A[0], P[1]-A[1])
        ab = (B[0]-A[0], B[1]-A[1])
        ab2 = ab[0]**2 + ab[1]**2 + 1e-16
        t   = max(0.0, min(1.0, (ap[0]*ab[0] + ap[1]*ab[1]) / ab2))
        closest = (A[0] + ab[0]*t, A[1] + ab[1]*t)
        return math.hypot(P[0]-closest[0], P[1]-closest[1])

    used = set()
    result = []
    all_lines = [LineString(zip(xs, ys)) for xs, ys in segs]

    # Endpunkte + Tangenten vorbereiten
    ends = []
    for xs, ys in segs:
        start_pt  = (xs[0], ys[0])
        second_pt = (xs[1], ys[1])
        end_pt    = (xs[-1], ys[-1])
        prev_pt   = (xs[-2], ys[-2])
        (tan_start, dir_start) = tangent(start_pt, second_pt, -tan_len)
        (tan_end,   dir_end)   = tangent(prev_pt, end_pt, tan_len)
        ends.append({
            'seg': (xs, ys),
            'line': LineString(zip(xs, ys)),
            'start':{'pt': start_pt, 'tan': tan_start, 'dir': dir_start},
            'end':  {'pt': end_pt,   'tan': tan_end,   'dir': dir_end}
        })

    # Winkel fÃ¤cher fÃ¼r FÃ¼hler
    fuzz_rad = math.radians(TANGENT_FUELLER_ANGLE)

    # Paare verbinden
    for i, ei in enumerate(ends):
        if i in used: continue
        best = None
        for j, ej in enumerate(ends):
            if j == i or j in used: continue
            for side_i in ('start','end'):
                for side_j in ('start','end'):
                    Pi = ei[side_i]['pt']
                    Pj = ej[side_j]['pt']
                    # Haupt-Tangenten
                    Ti_main = ei[side_i]['tan']
                    Tj_main = ej[side_j]['tan']
                    # Richtungsvektoren
                    ui = ei[side_i]['dir']
                    uj = ej[side_j]['dir']
                    # FÃ¼hler-Richtungen
                    ui_max = rotate(ui, +fuzz_rad)
                    ui_min = rotate(ui, -fuzz_rad)
                    uj_max = rotate(uj, +fuzz_rad)
                    uj_min = rotate(uj, -fuzz_rad)
                    # LineStrings
                    lines_i = [
                        LineString([Pi, Ti_main]),
                        LineString([Pi, (Pi[0]+ui_max[0]*tan_len, Pi[1]+ui_max[1]*tan_len)]),
                        LineString([Pi, (Pi[0]+ui_min[0]*tan_len, Pi[1]+ui_min[1]*tan_len)])
                    ]
                    lines_j = [
                        LineString([Pj, Tj_main]),
                        LineString([Pj, (Pj[0]+uj_max[0]*tan_len, Pj[1]+uj_max[1]*tan_len)]),
                        LineString([Pj, (Pj[0]+uj_min[0]*tan_len, Pj[1]+uj_min[1]*tan_len)])
                    ]

                    # Schnitt prÃ¼fen: wenn mindestens eine Kombination schneidet
                    if AUTO_CONNECT_CHECK_TANGENT_INTERSECT:
                        intersects_flag = False
                        for li in lines_i:
                            for lj in lines_j:
                                if li.intersects(lj):
                                    intersects_flag = True
                                    break
                            if intersects_flag:
                                break
                        if not intersects_flag:
                            continue

                    if AUTO_CONNECT_REQUIRE_TAN_BOUNDED:
                        # prÃ¼fe Bounding nur fÃ¼r Haupttangenten als NÃ¤herung
                        ip = LineString([Pi, Ti_main]).intersection(LineString([Pj, Tj_main]))
                        if not isinstance(ip, Point) or not (min(Pi[0],Pj[0])<=ip.x<=max(Pi[0],Pj[0])):
                            continue

                    slope_i = (Ti_main[1]-Pi[1])/(Ti_main[0]-Pi[0]+1e-16)
                    xs2, ys2 = ej['seg']
                    slope_j = ((ys2[1]-ys2[0])/(xs2[1]-xs2[0]+1e-16)
                               if side_j=='start'
                               else (ys2[-1]-ys2[-2])/(xs2[-1]-xs2[-2]+1e-16))
                    if AUTO_CONNECT_CHECK_SLOPE and abs(slope_i-slope_j)>=slope_tol:
                        continue

                    dist = point_segment_distance(Pj, Pi, Ti_main)

                    seg1_x, seg1_y = ei['seg']
                    seg2_x, seg2_y = ej['seg']
                    if AUTO_CONNECT_CHECK_REVERSAL:
                        other_i = (seg1_x[-1], seg1_y[-1]) if side_i=='start' else (seg1_x[0],seg1_y[0])
                        interior_i = (other_i[0]-Pi[0], other_i[1]-Pi[1])
                        tang_i     = (Ti_main[0]-Pi[0], Ti_main[1]-Pi[1])
                        if interior_i[0]*tang_i[0]+interior_i[1]*tang_i[1]<0:
                            seg1_x, seg1_y = seg1_x[::-1], seg1_y[::-1]
                        other_j = (seg2_x[-1], seg2_y[-1]) if side_j=='start' else (seg2_x[0],seg2_y[0])
                        interior_j = (other_j[0]-Pj[0], other_j[1]-Pj[1])
                        tang_j     = (Tj_main[0]-Pj[0], Tj_main[1]-Pj[1])
                        if interior_j[0]*tang_j[0]+interior_j[1]*tang_j[1]<0:
                            seg2_x, seg2_y = seg2_x[::-1], seg2_y[::-1]

                    new_x = seg1_x + seg2_x
                    new_y = seg1_y + seg2_y
                    if AUTO_CONNECT_CHECK_CROSSING and any(LineString(zip(new_x,new_y)).crosses(ek['line']) for k,ek in enumerate(ends) if k not in (i,j)):
                        continue

                    # speichere bestes Paar nach Distanz
                    if best is None or dist<best[0]:
                        best = (dist, j, side_i, side_j, Pi, Ti_main, Pj, Tj_main, new_x, new_y)

        if best:
            _, j, side_i, side_j, Pi_dbg, Ti_dbg, Pj_dbg, Tj_dbg, new_x, new_y = best
            if AUTO_CONNECT_DEBUG:
                close_all()
                fig, ax = plt.subplots()
                len_i = len(ends[i]['seg'][0])
                ax.plot(new_x[:len_i], new_y[:len_i], 'o-', label=f'Segment {i}')
                ax.plot(new_x[len_i:], new_y[len_i:], 'o-', label=f'Segment {j}')
                ax.plot([Pi_dbg[0],Ti_dbg[0]],[Pi_dbg[1],Ti_dbg[1]],'--',label='Tangent i')
                ax.plot([Pj_dbg[0],Tj_dbg[0]],[Pj_dbg[1],Tj_dbg[1]],'--',label='Tangent j')
                ax.legend(); ax.set_title(f'Debug Connect {i}â†”{j}')
                try: plt.show()
                except: pass
            result.append((new_x, new_y))
            used.update({i, j})
        else:
            result.append(ei['seg'])
            used.add(i)

    for k, ek in enumerate(ends):
        if k not in used:
            result.append(ek['seg'])
    return result

# --------------------------------------------------------------------------
# Hauptprogramm
# --------------------------------------------------------------------------
if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))

    # 1) Alpha-Shape und Polygon-Interieur
    nlist_pts = read_nlist(os.path.join(base, 'NLIST.txt'))
    shape      = alpha_shape(nlist_pts, ALPHA_SHAPE_ALPHA)
    inner      = shape.buffer(-ALPHA_BUFFER_MARGIN)
    print('Alpha-Shape berechnet' if not inner.is_empty else 'Warnung: leer')

    # 2) Daten einlesen
    ox, oy = read_coordinates(os.path.join(base, 'HSLinie.txt'))
    sp      = read_start_points(os.path.join(base, 'concave_hull_points.txt'))

    # 3) Globalfilter
    fx, fy = global_filter_points_numba(np.array(ox), np.array(oy),
                                        np.array(sp), PROXIMITY_RADIUS)

    # 4) Segmentierung & Verschiebung
    segs = segment_filtered_points(fx, fy, sp, MIN_POINTS_PER_SEGMENT)
    segs = [([x+X_SHIFT for x in xs], ys) for xs, ys in segs]
    plot_segments(segs, 'ungefilterte Segmente')

    # 5) Dichte-Filter
    segs = filter_segments_by_density(segs, MIN_POINT_DENSITY)

    # 6) Konflikt-Split
    segs = split_segments_by_conflict(segs, CONFLICT_SPLIT_RADIUS)

    # 7) Alpha-Filter
    if not inner.is_empty:
        segs = filter_segments_by_alpha_parallel(segs, inner)

    # 8) Monotonie-Split in X
    if SPLIT_NON_MONOTONIC_X:
        segs = split_segments_by_monotonicity(segs, tol=MONOTONIC_SPLIT_TOL)

    # 9) Splines vor funktionalem Filter
    plot_splines(segs, OUTPUT_POINT_DENSITY,
                 'Splines vor funktionalem Filter')

    # 10) Funktionaler Filter
    segs = filter_segments_by_function_distance(segs, MIN_FUNCTION_DISTANCE, OUTPUT_POINT_DENSITY)

    # 11) Segmente verbinden
    if CONNECT_AUTO:
        segs = auto_connect_segments(segs)
        segs = auto_connect_segments(segs)

    # 12) Polygon-Clip
    segs = filter_segments_by_polygon(segs, shape)

    # 13) Ausgabe und finaler Plot
    out = os.path.join(os.path.expanduser('~'),
                       'Downloads', 'Segments.txt')
    write_segments(segs, out)
    print(f'Segment-Datei: {out}')
    plot_segments(segs, 'Final mit Tangenten')
