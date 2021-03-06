import cv2
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask as riomask
from rasterio.warp import (
    aligned_target,
    calculate_default_transform,
    reproject,
    Resampling,
)


def min_max_scale(img_array):
    """Scale the raster pixel values according to min/max principle"""
    mask = np.where(img_array == 0, True, False)
    min_val = max(img_array.min(), img_array.mean() - 2 * img_array.std())
    max_val = min(img_array.max(), img_array.mean() + 2 * img_array.std())
    img = (img_array - min_val) * 255 / (max_val - min_val)
    img[mask] = 0
    return img


def max_scale(img_array):
    """Scale the raster pixel values with division by maximum value"""
    max_val = img_array.max()
    img_array = (img_array / max_val) * 255
    return img_array.astype(np.uint8)


def transform_resolution(data_path, save_path, resolution=(10, 10)):
    """Transform raster spatial resolution"""
    with rasterio.open(data_path) as src:

        transform, width, height = aligned_target(
            transform=src.meta["transform"],
            width=src.width,
            height=src.height,
            resolution=resolution,
        )

        kwargs = src.meta.copy()
        kwargs.update(
            {"transform": transform, "width": width, "height": height, "nodata": 0}
        )

        if ".jp2" in data_path:
            save_path = save_path.replace(".jp2", ".tif")
            kwargs["driver"] = "GTiff"
        with rasterio.open(save_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    resampling=Resampling.nearest,
                )

    return save_path


def transform_crs(data_path, save_path, dst_crs="EPSG:4326", resolution=(10, 10)):
    """Transform coordinate reference system of the raster"""
    with rasterio.open(data_path) as src:
        if resolution is None:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
        else:
            transform, width, height = calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=resolution,
            )
        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )
        with rasterio.open(save_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )

    return save_path


def crop_raster(raster_path, aoi_path, out_raster_name=None):
    """Crop a patch of the raster where it is within
    the bounds of specified Area of Interest"""
    aoi = gpd.read_file(aoi_path)
    with rasterio.open(raster_path) as tile:
        meta = tile.meta
        region, region_tfs = riomask.mask(
            tile, aoi.to_crs(tile.crs).geometry, all_touched=False, crop=True
        )

    if out_raster_name is None:
        out_raster_name = raster_path.replace(".tif", "_cropped.tif")
        out_raster_name = out_raster_name.replace(".jp2", "_cropped.tif")
    out_raster_name = out_raster_name.replace(".jp2", ".tif")

    assert out_raster_name != raster_path

    meta["width"] = region.shape[-1]
    meta["height"] = region.shape[-2]
    meta["transform"] = region_tfs
    meta["nodata"] = 0
    meta["driver"] = "GTiff"

    with rasterio.open(out_raster_name, "w", **meta) as dst:
        for band in range(meta["count"]):
            dst.write(region[band], band + 1)

    return out_raster_name


def calculate_ndwi(b03_path, b08_path, out_path=None):
    with rasterio.open(b08_path) as src:
        b08 = src.read(1).astype(rasterio.float32)
        crs = src.crs
        meta = src.meta

    with rasterio.open(b03_path) as src:
        b03 = src.read(1).astype(rasterio.float32)

    ndvi = np.where((b08 + b03) == 0, 0, (b08 - b03) / (b08 + b03))
    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is None:
        out_path = b08_path.replace("_B08", "_NDWI").replace(".jp2", ".tif")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.meta["nodata"] = 0
        dst.meta["max"] = 1
        dst.meta["min"] = 0
        dst.write(ndvi.astype(rasterio.float32), 1)

    return out_path


def calculate_ndvi(b04_path, b08_path, out_path=None, nodata=0):

    with rasterio.open(b08_path) as src:
        b08 = src.read(1).astype(rasterio.float32)
        crs = src.crs
        meta = src.meta

    with rasterio.open(b04_path) as src:
        b04 = src.read(1).astype(rasterio.float32)

    ndvi = np.where((b08 + b04) == 0, nodata, (b08 - b04) / (b08 + b04))
    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is None:
        out_path = b08_path.replace("_B08", "_NDVI").replace(".jp2", ".tif")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.meta["nodata"] = 0
        dst.meta["max"] = 1
        dst.meta["min"] = 0
        dst.write(ndvi.astype(rasterio.float32), 1)

    return out_path


def calculate_ndmi(b08_path, b12_path, out_path=None, nodata=0):

    with rasterio.open(b08_path) as src:
        b08 = src.read(1).astype(rasterio.float32)
        crs = src.crs
        meta = src.meta

    with rasterio.open(b12_path) as src:
        b12 = src.read(1).astype(rasterio.float32)

    b12 = cv2.resize(b12, (b08.shape[-1], b08.shape[-2]), interpolation=cv2.INTER_AREA)
    ndvi = np.where((b08 + b12) == 0, nodata, (b08 - b12) / (b08 + b12))

    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is None:
        out_path = b08_path.replace("_B08", "_NDMI").replace(".jp2", ".tif")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.meta["nodata"] = 0
        dst.meta["max"] = 1
        dst.meta["min"] = 0
        dst.write(ndvi.astype(rasterio.float32), 1)

    return out_path


def calculate_ndre(b07_path, b05_path, out_path=None, nodata=0):
    with rasterio.open(b07_path) as src:
        b07 = src.read(1).astype(rasterio.float32)
        crs = src.crs
        meta = src.meta

    with rasterio.open(b05_path) as src:
        b05 = src.read(1).astype(rasterio.float32)

    if b05.shape != b07.shape:
        b05 = cv2.resize(
            b05, (b07.shape[-2], b07.shape[-1]), interpolation=cv2.INTER_AREA
        )

    ndvi = np.where((b07 + b05) == 0, nodata, (b07 - b05) / (b07 + b05))
    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is None:
        out_path = b07_path.replace("_B07", "_NDRE").replace(".jp2", ".tif")

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.meta["nodata"] = 0
        dst.meta["max"] = 1
        dst.meta["min"] = 0
        dst.write(ndvi.astype(rasterio.float32), 1)

    return out_path


def stack_layers(layers, out_path=None):
    arrays = []
    for layer in layers:

        with rasterio.open(layer) as src:
            img = src.read().astype(rasterio.float32)
            crs = src.crs
            meta = src.meta

        for channel in img:
            arrays.append(channel)

    img_size = max([x.shape for x in arrays])
    for i in range(len(arrays)):
        if arrays[i].shape != img_size:
            arrays[i] = cv2.resize(
                arrays[i], (img_size[-2], img_size[-1]), interpolation=cv2.INTER_AREA
            )

    stacked_img = np.stack(arrays, axis=0)

    meta.update(count=len(arrays))
    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is not None:
        with rasterio.open(out_path, "w", **meta) as dst:
            for i, band in enumerate(stacked_img):
                dst.write(band, i + 1)

    return stacked_img


def combine_bands(band1, band2, op="+", out_path=None, out_shape=None):

    with rasterio.open(band1) as src:
        b1 = src.read(1).astype(rasterio.float32)
        crs = src.crs
        meta = src.meta

    with rasterio.open(band2) as src:
        b2 = src.read(1).astype(rasterio.float32)

    if out_shape is None:
        if sum(b1.shape) > sum(b2.shape):
            b2 = cv2.resize(
                b2, (b1.shape[-2], b1.shape[-1]), interpolation=cv2.INTER_AREA
            )
        elif sum(b1.shape) < sum(b2.shape):
            b1 = cv2.resize(
                b1, (b2.shape[-2], b2.shape[-1]), interpolation=cv2.INTER_AREA
            )
    else:
        b2 = cv2.resize(b2, out_shape, interpolation=cv2.INTER_AREA)
        b1 = cv2.resize(b1, out_shape, interpolation=cv2.INTER_AREA)
        meta["width"] = out_shape[-1]
        meta["height"] = out_shape[-2]

    if op == "+":
        calculated_img = b1 + b2
    elif op == "-":
        calculated_img = b1 - b2
    elif op == "/":
        calculated_img = b1 / b2

    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is not None:
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.meta["nodata"] = 0
            dst.meta["max"] = calculated_img.max()
            dst.meta["min"] = calculated_img.min()
            dst.write(calculated_img.astype(rasterio.float32), 1)

    return calculated_img
