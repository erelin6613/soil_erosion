import cv2
import geopandas as gpd
import numpy as np
from tqdm import tqdm

smooth = 1e-5


def calculate_iou(pred_poly, gt_gdf):
    """Get the best intersection over union for a predicted polygon.

    Arguments
    ---------
    pred_poly : :py:class:`shapely.Polygon`
        Prediction polygon to test.
    gt_gdf : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame of ground truth polygons to test ``pred_poly`` against.

    Returns
    -------
    iou_GDF : :py:class:`geopandas.GeoDataFrame`
        A subset of ``gt_gdf`` that overlaps ``pred_poly`` with an added
        column ``iou_score`` which indicates the intersection over union value.

    """

    # Fix bowties and self-intersections
    if not pred_poly.is_valid:
        pred_poly = pred_poly.buffer(0.0)

    precise_matches = gt_gdf[gt_gdf.intersects(pred_poly)]

    iou_row_list = []
    for _, row in precise_matches.iterrows():
        # Load ground truth polygon and check exact iou
        test_poly = row.geometry
        # Ignore invalid polygons for now
        if pred_poly.is_valid and test_poly.is_valid:
            intersection = pred_poly.intersection(test_poly).area
            union = pred_poly.union(test_poly).area
            # Calculate iou
            iou_score = intersection / float(union)
        else:
            iou_score = 0
        row['iou_score'] = iou_score
        iou_row_list.append(row)

    iou_GDF = gpd.GeoDataFrame(iou_row_list)
    return iou_GDF


def process_iou(pred_poly, gt_gdf, iou_threshold=0.8):
    """Get the maximum IoU score for a predicted polygon.
    If the value is higher than threshold, returns IoU score as well as test polygon with which it was intersected

    Arguments
    ---------
    pred_poly : :py:class:`shapely.geometry.Polygon`
        Prediction polygon to test.
    gt_gdf : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame of ground truth polygons to test ``pred_poly`` against.

    Returns
    -------
        Row with maximum IoU score and IoU score

    """

    iou_GDF = calculate_iou(pred_poly, gt_gdf)
    if len(iou_GDF) == 0:
        return
    max_iou = iou_GDF['iou_score'].max()
    if max_iou > iou_threshold:
        max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax(axis=0, skipna=True)]
        return max_iou_row, max_iou


def evaluate_polys(pred_gdf, gt_gdf, iou_threshold=0.8, eps=1e-5):
    """Gets prediction and ground truth GeoDataFrames and returns metrics based on the IoU of polygons and IoU threshold that was set

    Arguments
    ---------
    pred_poly : :py:class:`shapely.geometry.Polygon`
        Prediction polygon to test.
    test_data_GDF : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame of ground truth polygons to test ``pred_poly`` against.
    iou_threshold :
        IoU threshold to decide which prediction is a good prediction

    Returns
    -------
        List of matches between predicted and ground truth polygons as well as evaluation metrics
    """

    tp, fp, fn = 0, 0, 0
    gt_gdf_copy = gt_gdf.copy()
    matches = []

    for idx, pred_series in tqdm(pred_gdf.iterrows()):
        pred_poly = pred_series["geometry"]
        result = process_iou(pred_poly, gt_gdf_copy, iou_threshold=iou_threshold)
        if result is not None:
            gt_row, iou = result
            gt_idx = int(gt_row.name)
            gt_gdf_copy.drop(gt_idx, inplace=True)
            matches.append((idx, gt_idx, iou))
            tp += 1
        else:
            fp += 1
            matches.append((idx, None, 0.0))
    fn = len(gt_gdf_copy)  # all of the polygons that weren't deleted from test gdf
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    try:
        f_score = (2 * precision * recall) / (precision + recall)
    except Exception:
        f_score = 0
    metrics = {
        "precision": precision,
        "recall": recall,
        "f_score": f_score
    }
    return matches, metrics


def filter_pixels(image, filter_image, dilate=False):
    if dilate:
        kernel = np.ones((7, 7))
        filter_image = cv2.dilate(filter_image, kernel)
    image[filter_image == 1.0] = 0
    return image, filter_image


def thresh(array, val=127):
    return np.where(array > val, 1, 0)


def iou(y_true, y_pred, threshold=127, smooth=1.0):
    y_true = thresh(y_true, val=threshold)
    y_pred = thresh(y_pred, val=threshold)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (1. * intersection + smooth) / (
            np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)
