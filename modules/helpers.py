import numpy as np
import math

def rot_pnt_around_pnt(p, c, rad):
    q = p-c
    rot = np.array([[ math.cos(rad), -math.sin(rad) ], 
                    [ math.sin(rad),  math.cos(rad)]])

    q_rot = rot.dot(q)
    return c + q_rot

def coord_is_in_rot_rect_domain(x, y, cx, cy, w, h, r):
    if r == 0:
        p_rot = np.array([x, y])
    else:
        p = np.array([x, y])
        c = np.array([cx, cy])
        p_rot = rot_pnt_around_pnt(p, c, r)

    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0

    if (p_rot[0] > x1 and p_rot[0] < x2 and p_rot[1] > y1 and p_rot[1] < y2):
        return True
    else:
        return False