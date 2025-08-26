#!/usr/bin/env python

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from skimage.draw import polygon2mask

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from utils import *

timer = Timer()

# Functions defined in the original notebook cells
def fit_kde_contour(points, bandwidth=None, contour_level=0.01, grid_res=100):
    kde = gaussian_kde(points.T, bw_method=bandwidth)

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    x_grid = np.linspace(x_min - (x_max - x_min)*0.1, x_max + (x_max - x_min)*0.1, grid_res)
    y_grid = np.linspace(y_min - (y_max - y_min)*0.1, y_max + (y_max - y_min)*0.1, grid_res)

    X, Y = np.meshgrid(x_grid, y_grid)
    xy_coords = np.vstack([X.ravel(), Y.ravel()])

    Z = kde(xy_coords).reshape(X.shape)

    return X, Y, Z

def get_convex_hull_corners(pts):
    hull = ConvexHull(pts)
    hull_corners = pts[hull.vertices]
    return hull_corners

def angle_between_points(a, b, c):
    vector_ab = b - a
    vector_ac = c - a

    dot_product = np.dot(vector_ab, vector_ac)

    magnitude_ab = np.linalg.norm(vector_ab)
    magnitude_ac = np.linalg.norm(vector_ac)

    cosine_angle = dot_product / (magnitude_ab * magnitude_ac + 1e-8)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)

def preprocess_data(pts_np, foot_pts_np, max_height=5, gd_buffer=0.05, maxdist=5):
    centroid = foot_pts_np.mean(axis=0).reshape(1, 3)
    print("CENTROID", np.round(centroid, 3))
    centroid[0, 2] += gd_buffer

    distances = np.linalg.norm(pts_np - centroid, axis=1)
    pts_np = pts_np[distances <= maxdist]

    z = centroid[0, 2]
    zs = pts_np[:, 2]
    pts_np = pts_np[((z - zs) < max_height) & (zs > z)]

    pts_np[:, :2] = pts_np[:, :2] - centroid[0, :2]
    foot_pts_np[:, :2] = foot_pts_np[:, :2] - centroid[0, :2]

    return pts_np

def perform_2d_analysis_and_plot(pts_np, shadow_length=10):
    global timer

    eps = 0.1
    min_samples = 5

    model = DBSCAN(eps=eps, min_samples=min_samples)

    timer.start_time("cluster")
    labels = model.fit_predict(pts_np)
    timer.end_time("cluster")
    unique_labels = set(labels)

    shadows = []
    hulls = []
    
    timer.start_time("hull loop")
    for k in unique_labels:
        points = pts_np[:, :2][labels == k]
        if k == -1 or points.shape[0] < min_samples: continue

        timer.start_time("hull " + str(points.shape[0]))
        try:
            hull = ConvexHull(points)
        except Exception as e:
            print(e)
            print("NUM PTS FAILED:", points.shape)

        timer.end_time("hull " + str(points.shape[0]))
        hull_corners = points[hull.vertices]

        timer.start_time("shadow " + str(len(hull_corners)))
        max_angle = -1
        edges = []
        for i in range(len(hull_corners)):
            for kk in range(len(hull_corners)):
                if i == kk: continue
                a = angle_between_points(np.array([0, 0]), hull_corners[i], hull_corners[kk])
                if abs(a) > max_angle:
                    max_angle = abs(a)
                    edges = np.array([hull_corners[i], hull_corners[kk]])

        shadow = []
        outers = []
        for p in edges:
            outer = p + shadow_length * (p) / np.linalg.norm((p))
            outers.append(outer)

        shadow = np.array([edges[0], outers[0], outers[1], edges[1]])
        timer.end_time("shadow " + str(len(hull_corners)))

        shadows.append(shadow)
        hulls.append(hull_corners)

    timer.end_time("hull loop")
    return shadows, hulls # Return necessary values for next stage

def generate_and_save_masks(shadows, hulls, dist2robo=10, res=128):
    global timer

    extent = [-dist2robo, dist2robo,
              -dist2robo, dist2robo]

    shadow_mask = np.zeros((res, res), dtype=bool)

    def to_image_coords(poly, extent, res):
        x_min, x_max, y_min, y_max = extent
        x_scaled = (poly[:, 0] - x_min) / (x_max - x_min) * res
        y_scaled = (poly[:, 1] - y_min) / (y_max - y_min) * res
        return np.stack([y_scaled, x_scaled], axis=-1)

    timer.start_time("masks")
    for h_poly in hulls:
        img_coords = to_image_coords(h_poly, extent, res)
        mask = polygon2mask((res, res), img_coords)
        shadow_mask |= mask

    for s_poly in shadows:
        img_coords = to_image_coords(s_poly, extent, res)
        mask = polygon2mask((res, res), img_coords)
        shadow_mask |= mask
    timer.end_time("masks")

    return shadow_mask