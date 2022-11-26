import os
import numpy as np
import geopandas as gpd
from PIL import Image
import cv2
import imageio
import rasterio
import rasterio.mask as riomask
from rasterio.plot import reshape_as_image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.utils.data import Dataset
from shapely.ops import unary_union
from shapely.geometry import Polygon, LinearRing, box
from shapely.geometry.collection import GeometryCollection
import skimage.morphology as morph

from .utils import exclude_classes, scale_ndvi, max_scale


train_add_targets = {'image': 'image',
                     'gt_mask': 'mask',
                     'filter_mask': 'mask'}

train_tfs = A.Compose([A.Transpose(p=0.5),
                       A.HorizontalFlip(p=0.5),
                       A.VerticalFlip(p=0.5),
                       A.ShiftScaleRotate(p=0.5),
                       A.Rotate(p=0.5),
                       ToTensorV2(p=1.0)],
                      additional_targets=train_add_targets)

val_tfs = A.Compose([ToTensorV2(p=1.0)],
                    additional_targets=train_add_targets)

drivers = {
    'tif': 'GTiff',
    'jp2': 'JP2OpenJPEG'
}


class IndexDataset(Dataset):
    """The IndexDataset is a generic object helping to
    keep a structure of files needed for datasets
    (creation of NDVI rasters, indexing paths to geojson
    files and tiles depending on bands, etc.)

    @param: df - pd.DataFrame object with necessary columns:
                "tci_tile_path" - paths to TCI images,
                "aoi_fp" - paths to Area of Interest geojsons,
                "geom_fp" - paths to corresponding geojson files.
                df can be None if and only if train_val_test_dir
                is specified

    @param: src_format - format of an input images

    @param: subset - the directory which contains folders with mini
                    chips organized in folders "masks", "tiles",
                    "filters" and if applicable "ndvi" and "nir".
                    Will be created if does not exist.

    @param: train_val_test_dir - directory with subsets "train",
                                "test", "valid" if created before

    @param: label_col - if geojson files should be filtered out based
                        on label the column name by which to classify
                        geometries should be specified. Requires column
                        "txt_ignore_file" with corresponding .txt files
                        which contain labels to be excluded.

    @param: bands - list of image bands to be processed;
                    possible elements: 'TCI', 'B01'-'B12', 'NDVI'
    """

    def __init__(self, df=None,
                 src_format='jp2',
                 dst_format='png',
                 dst_crs='EPSG:32636',
                 subset='train',
                 train_val_test_dir=None,
                 label_col=None,
                 bands=['TCI', 'B04', 'NDVI']):

        super(Dataset, self).__init__()

        self.src_format = src_format
        self.dst_crs = dst_crs
        self.dst_format = dst_format
        self.subset = subset
        self.label_col = label_col
        self.bands = bands

        if train_val_test_dir is None:
            self.tci_files = sorted(df.tci_tile_path.unique())
            if 'b04_tile_path' in df.columns:
                self.b04_files = sorted(df.b04_tile_path.unique())
            if 'b08_tile_path' in df.columns:
                self.b08_files = sorted(df.b08_tile_path.unique())

            self.polygons_list = df.geom_fp.tolist()
            self.aois_list = df.aoi_fp.tolist()

            if 'txt_ignore_file' in df.columns:
                self.txt_ignore_files = df.txt_ignore_file.tolist()
            else:
                self.txt_ignore_files = [None for _ in range(len(self.aois_list))]

            if 'filters_fp' in df.columns:
                self.excluded_polys = df.filters_fp.tolist()
            else:
                self.excluded_polys = [None for _ in range(len(self.aois_list))]

            if 'label_col' in df.columns:
                self.label_cols = df.label_col.tolist()
            else:
                self.label_cols = [None for _ in range(len(self.aois_list))]

            ndvi_checkfile = self.tci_files[0].replace(
                '_TCI_', '_NDVI_').replace('.jp2', '.tif')

            if not os.path.exists(ndvi_checkfile) and ('NDVI' in self.bands):
                print('making ndvis')
                self.ndvi_files = self._make_ndvis()
            else:
                ndvi_files = [x.replace('_TCI_', '_NDVI_') for x in self.tci_files]
                self.ndvi_files = [x.replace('.jp2', '.tif') for x in ndvi_files]

    def _index_splits(self, mask_dir):
        print(self.tci_files, self.aois_list, self.b08_files, self.polygons_list)
        self.masks = sorted([os.path.join(mask_dir, x) for x in os.listdir(
            mask_dir) if x.endswith('.'+self.dst_format)])
        self.masks_ignore = [file.replace(
            'mask', 'filters') for file in self.masks]
        self.tci_chips = [file.replace(
            'mask', 'tiles') for file in self.masks]
        self.nir_chips = [file.replace(
            'mask', 'nir') for file in self.masks]
        self.ndvi_chips = [file.replace(
            'mask', 'nir') for file in self.masks]

    def _make_ndvis(self):
        tiles = set(self.tci_files)
        ndvi_files = []
        np.seterr(divide='ignore', invalid='ignore')
        for tile in tiles:
            with rasterio.open(tile.replace('_TCI_', '_B08_')) as src:
                meta = src.meta
                nir = src.read(1).astype(rasterio.float32)
            with rasterio.open(tile.replace('_TCI_', '_B04_')) as src:
                red = src.read(1).astype(rasterio.float32)

            outname = tile.replace('_TCI_', '_NDVI_')
            outname = outname.replace('.jp2', '.tif')
            self._write_ndvi(nir, red, meta, outname)

            ndvi_files.append(outname)
        return ndvi_files


class PolyDataset(IndexDataset):
    """Dataset object is aimed to prepare the
    binary masks and images for future training.
    Creates two folder with '_mask' and '_tiles'
    suffixes named by corresponding width and height
    @param: tiles_dir - path to the directory where loaded
                        tile images are stored
    @param: polygons_path - path to the file containing
                            polygons (as of now was tested
                            with geojson where all polygons
                            are combined)
    @param: mini_tile_size - size of images to cut images into,
                                by default images will be cut
                                into 256x256 tiles
    @param: src_format - format of an input image
    @param: dst_format - format of output tiles
    @param: dst_crs - CRS of desired format to convert shapes to
    @param: skip_empty_masks - controls the behavior of saving
                                tiles, i.e. if True algorythm will
                                not write to files images and masks
                                where there is no mask pixels
    @param: transforms - transforms applied to images
    """

    def __init__(self, df=None,
                 mask_type='bounds',
                 mini_tile_size=256,
                 src_format='jp2',
                 dst_format='png',
                 dst_crs='EPSG:32636',
                 skip_empty_masks=True,
                 transforms=train_tfs,
                 subset='train',
                 train_val_test_dir=None,
                 label_col=None,
                 bands=['TCI', 'B04', 'NDVI']):

        super().__init__(df=df, src_format=src_format,
                         dst_format=dst_format,
                         dst_crs=dst_crs,
                         subset=subset,
                         train_val_test_dir=train_val_test_dir,
                         label_col=label_col,
                         bands=bands)

        self.mini_tile_size = mini_tile_size
        self.transforms = transforms
        self.skip_empty_masks = skip_empty_masks

        mask_dir = f"dataset/{self.subset}/mask"
        tiles_dir = f"dataset/{self.subset}/tiles"
        nir_dir = f"dataset/{self.subset}/nir"
        ndvi_dir = f"dataset/{self.subset}/ndvi"

        if train_val_test_dir is None:

            self._prepare_mini_tiles(mask_dir=mask_dir,
                                     tiles_dir=tiles_dir,
                                     nir_dir=nir_dir,
                                     ndvi_dir=ndvi_dir,
                                     filters_dir=None)

        self._index_splits(mask_dir)

    def _write_ndvi(self, nir, red, meta, outname):

        ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))
        ndvi = ndvi.astype(rasterio.float32)

        meta.update(driver='GTiff')
        meta.update(dtype=rasterio.float32)

        with rasterio.open(outname, 'w', **meta) as dst:
            dst.meta['nodata'] = 0
            dst.meta['max'] = 1
            dst.meta['min'] = 0
            dst.write(ndvi.astype(rasterio.float32), 1)

    def _make_aoi_filter(self, polygons, aoi, crs, save_path=None):

        if isinstance(polygons, list):
            polygons = gpd.GeoDataFrame({'geometry': polygons})
            polygons.crs = crs

        aoi = aoi.to_crs(polygons.crs)
        try:
            chunk = polygons[polygons.within(aoi.geometry.values[0].buffer(50))]
            filters = aoi.difference(chunk.buffer(20).unary_union)
        except Exception as e:
            print(e)

        if save_path is not None:
            filters.to_file(save_path, driver='GeoJSON')

        return filters

    def _make_aoi_boxes(self, polygons_paths):

        aois = []
        for geom in polygons_paths:
            geoms = gpd.read_file(geom)
            pseudo_aoi = box(*geoms.total_bounds)
            filename = geom.replace('.geojson', '_aoi.geojson')
            gdf = gpd.GeoDataFrame({'geometry': [pseudo_aoi]})
            gdf.set_crs(geoms.crs, inplace=True)
            gdf.to_file(filename, driver='GeoJSON')

            aois.append(filename)

        return aois

    def _load_polygons(self, poly_path, dst_crs,
                       cascade=True, preprocess='boundry',
                       ignore_classes=False, txt_ignore_file=np.nan,
                       label_col=None, borders_width=8):
        """Load shapes to create masks from"""

        polygons = gpd.read_file(poly_path)
        if ignore_classes and str(txt_ignore_file) != str(np.nan):
            polygons = exclude_classes(polygons,
                                       column=label_col,
                                       txt_file=txt_ignore_file,
                                       verbose=False)

        try:
            if polygons.crs != dst_crs:
                polygons = polygons.to_crs(dst_crs)
        except Exception:
            print(f'Invalid geometry captured, trying to handle: {poly_path}')
            polygons['geometry'] = polygons.geometry.apply(
                lambda x: x if x else GeometryCollection())

        polygons = polygons.buffer(0)

        if preprocess == 'boundry':
            bounds = polygons.boundary.buffer(borders_width)
        else:
            bounds = polygons.geometry.buffer(borders_width*3)
            polygons = polygons.geometry.buffer(-borders_width)

        bounds = bounds.geometry.tolist()
        polygons = polygons.geometry.tolist()

        if len(bounds) < 1:
            print(f'Warning: {poly_path} as no valid geometries, crs: {polygons.crs}')
        if cascade:
            return [unary_union(bounds)], [unary_union(polygons)]
        return bounds, polygons

    def _prepare_mini_tiles(self, mask_dir, tiles_dir,
                            nir_dir=None,
                            ndvi_dir=None, filters_dir=None,
                            external_buffer=5):
        """Prepare folders with mini-tiles if folders are abesnt"""

        if not os.path.exists(mask_dir) or len(
            list(os.listdir(mask_dir)))==0:

            os.makedirs(mask_dir, exist_ok=True)
            os.makedirs(tiles_dir, exist_ok=True)

            if nir_dir is not None:
                os.makedirs(nir_dir, exist_ok=True)
            if ndvi_dir is not None:
                os.makedirs(ndvi_dir, exist_ok=True)
            if filters_dir is not None:
                os.makedirs(filters_dir, exist_ok=True)

            print('Creating folder structure...')
            self.count = 0

            pbar = tqdm(self.aois_list)

            for j, aoi_name in enumerate(pbar):

                pbar.set_description(
                    f'Processing AOI: {os.path.split(aoi_name)[-1]}')
                aoi = gpd.read_file(aoi_name)

                for i in range(len(self.tci_files)):
                    tile_name = self.tci_files[i]
                    pbar.set_postfix(
                        {'raster': os.path.split(tile_name)[-1],
                         'total_images': self.count})
                    tile = rasterio.open(tile_name)

                    aoi = aoi.to_crs(tile.crs)

                    try:
                        region, region_tfs = riomask.mask(
                            tile, aoi.geometry, all_touched=False, crop=True)
                    except Exception:
                        continue

                    nir = rasterio.open(tile_name.replace(
                        '_TCI_', '_B08_'))
                    ndvi = rasterio.open(
                        tile_name.replace('_TCI_', '_NDVI_').replace(
                            '.jp2', '.tif'))

                    nir_arr, _ = riomask.mask(
                        nir, aoi.geometry, all_touched=False, crop=True)
                    ndvi_arr, _ = riomask.mask(
                        ndvi, aoi.geometry, all_touched=False, crop=True)

                    bounds, polys = self._load_polygons(
                        aoi_name.replace('_aoi.geojson', '.geojson'),
                        tile.crs,
                        cascade=False,
                        ignore_classes=True,
                        label_col=self.label_cols[j],
                        txt_ignore_file=self.txt_ignore_files[j])

                    fill_bounds = aoi.boundary.buffer(external_buffer)
                    aoi_mask, mask_bounds = self._preprare_mask(
                        fill_bounds, bounds, region_tfs, region.shape)
                    tile_name = os.path.split(tile_name)[-1]

                    if filters_dir is not None:

                        try:
                            # tile.crs: list object has no attribute crs, rasterio error
                            _, filters = self._load_polygons(
                                aoi_name.replace(
                                    '_aoi.geojson', '_filters.geojson'),
                                tile.crs,
                                cascade=False)

                        except Exception as e:
                            print(e, tile)
                            continue

                    else:
                        filters = self._make_aoi_filter(
                            polys, aoi, tile.crs,
                            aoi_name.replace('_aoi.geojson', '_filters.geojson'))

                    mask_filters = rasterio.features.rasterize(
                        shapes=filters,
                        out_shape=(region.shape[-2], region.shape[-1]),
                        transform=region_tfs,
                        default_value=255)

                    self._crop_arrays(
                        tile_arr=region,
                        nir_arr=nir_arr[0],
                        ndvi_arr=ndvi_arr[0],
                        mask=mask_bounds,
                        filters=mask_filters,
                        tiles_dir=tiles_dir,
                        mask_dir=mask_dir,
                        filters_dir=filters_dir,
                        ndvi_dir=tiles_dir.replace('tiles', 'ndvi'),
                        nir_dir=tiles_dir.replace('tiles', 'nir'),
                        tile_name=tile_name.split('.')[0],
                        aoi_name=os.path.split(aoi_name)[-1].split('.')[0])

                tile.close()
                nir.close()
                ndvi.close()
        else:
            print('Dataset has been created beforehand, skipping')

    def _preprare_mask(self, fill_bounds, bounds, region_tfs, shape):

        aoi_mask = rasterio.features.rasterize(
            shapes=fill_bounds.geometry.tolist(),
            out_shape=(shape[-2], shape[-1]),
            transform=region_tfs,
            default_value=255)

        mask_bounds = rasterio.features.rasterize(
            shapes=bounds,
            out_shape=(shape[-2], shape[-1]),
            transform=region_tfs,
            default_value=255)

        mask_bounds[aoi_mask != 0] = 255

        return aoi_mask, mask_bounds

    def _crop_arrays(self, tile_arr,
                     mask, filters=None,
                     nir_arr=None, ndvi_arr=None,
                     tiles_dir='tiles',
                     mask_dir='mask',
                     nir_dir=None,
                     ndvi_dir=None,
                     filters_dir=None,
                     tile_name='img',
                     aoi_name='aoi',
                     mask_erosions=0):

        if nir_arr is not None:
            if (nir_arr.max()) < 1:
                print('Empty NIR array aquired, skipping')
                # catch if this is not the right raster,
                # usually occuring on the overlap of the tiles
                return

            nir_arr = max_scale(nir_arr)

        if ndvi_arr is not None:
            ndvi_arr = scale_ndvi(ndvi_arr)

        for i in range(tile_arr.shape[-1]//self.mini_tile_size):
            for j in range(tile_arr.shape[-2]//self.mini_tile_size):

                width, heigth = int(i*self.mini_tile_size), int(
                    j*self.mini_tile_size)

                mask_arr = mask[heigth:heigth+self.mini_tile_size,
                                width:width+self.mini_tile_size]
                tile_arr_ = tile_arr[:, heigth:heigth+self.mini_tile_size,
                                     width:width+self.mini_tile_size]

                if mask_arr.sum() < 8e4 or tile_arr_.sum() < 8e3:
                    continue

                if mask_erosions != 0:
                    kernel = np.ones((5, 5), np.uint8)
                    mask_arr = cv2.erode(mask_arr, kernel, int(mask_erosions))

                if os.path.exists(
                    os.path.join(
                        mask_dir, f'{tile_name}_{aoi_name}_{i}_{j}.png')):
                    continue

                if nir_arr is not None:
                    nir_img = nir_arr[heigth:heigth+self.mini_tile_size,
                                      width:width+self.mini_tile_size]
                    if nir_img.sum() < 1e3:
                        continue

                    imageio.imwrite(
                        os.path.join(
                            nir_dir,
                            f'{tile_name}_{aoi_name}_{i}_{j}.png'),
                        nir_img)

                if ndvi_arr is not None:
                    ndvi_img = ndvi_arr[heigth:heigth+self.mini_tile_size,
                                        width:width+self.mini_tile_size]
                    imageio.imwrite(
                        os.path.join(
                            ndvi_dir, f'{tile_name}_{aoi_name}_{i}_{j}.png'),
                        ndvi_img)

                imageio.imwrite(
                    os.path.join(
                        mask_dir, f'{tile_name}_{aoi_name}_{i}_{j}.png'),
                    np.uint8(mask_arr))

                imageio.imwrite(
                    os.path.join(
                        tiles_dir, f'{tile_name}_{aoi_name}_{i}_{j}.png'),
                    np.uint8(reshape_as_image(tile_arr_)))

                if filters is not None:
                    if filters_dir is None:
                        filters_dir = f"dataset/{self.subset}/filters"
                        os.makedirs(filters_dir, exist_ok=True)

                    filters_arr = filters[heigth:heigth+self.mini_tile_size,
                                          width:width+self.mini_tile_size]
                    imageio.imwrite(
                        os.path.join(
                            filters_dir, f'{tile_name}_{aoi_name}_{i}_{j}.png'),
                        np.uint8(filters_arr))

                self.count += 1

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):

        tci_chip = Image.open(self.tci_chips[idx])
        nir_chip = Image.open(self.nir_chips[idx])
        ndvi_chip = Image.open(self.ndvi_chips[idx])
        mask = Image.open(self.masks[idx])
        mask_ignore = Image.open(self.masks_ignore[idx])

        if tci_chip.mode != 'RGB':
            tci_chip = tci_chip.convert('RGB')

        tci_chip = np.asarray(tci_chip) / 255
        nir_chip = np.asarray(nir_chip) / 255
        ndvi_chip = np.asarray(ndvi_chip) / 255
        mask = np.asarray(mask) / 255
        mask_ignore = np.asarray(mask_ignore) / 255

        if self.bands == ['TCI']:
            image = tci_chip
        else:
            image = np.stack([tci_chip[:, :, 0],
                              tci_chip[:, :, 1],
                              tci_chip[:, :, 2],
                              nir_chip,
                              ndvi_chip],
                             axis=-1)

        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                gt_mask=mask,
                filter_mask=mask_ignore)

            img, mask, mask_ignore = (transformed['image'],
                                      transformed['gt_mask'],
                                      transformed['filter_mask'])

            mask_ignore = np.where(mask_ignore == 1.0, True, False)

        else:
            img = torch.tensor(image)
            mask, mask_ignore = torch.tensor(mask), torch.tensor(mask_ignore)

        if len(mask.shape) == 2:
            mask, mask_ignore = (mask.unsqueeze(0),
                                 torch.tensor(mask_ignore).unsqueeze(0))

        return img.float(), mask.float(), mask_ignore


class BoundaryDetector(IndexDataset):

    def __init__(self, model,
                 df,
                 tiles_dir='test',
                 mini_tile_size=256,
                 dst_crs='EPSG:4326',
                 src_format='jp2',
                 dst_format='png',
                 train_val_test_dir=None,
                 bands=['TCI', 'NIR'],
                 device=None
                 ):

        super().__init__(df=df, src_format=src_format,
                         dst_format=dst_format,
                         dst_crs=dst_crs,
                         train_val_test_dir=train_val_test_dir,
                         bands=bands)

        self.mini_tile_size = mini_tile_size
        self.dst_crs = dst_crs
        if device is None:
            self.device = self._get_device()
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device)

    def _get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

    def _load_tile(self, tile_path):
        with rasterio.open(tile_path) as src:
            image = src.read()
            crs = src.crs
            meta = src.meta
        return image, crs, meta

    def _write_raster(self, image, img_path, meta):

        if len(image.shape) == 2:
            bands = 1
        else:
            bands = image.shape[0]

        with rasterio.open(img_path, 'w', **meta) as dst:
            if bands == 1:
                dst.write(image.astype(meta['dtype']), 1)
                return

            for band in range(bands):
                dst.write(image[band].astype(meta['dtype']), band+1)

    def _predict_chip(self, chip, device, conf_thresh=0.5, transforms=val_tfs):

        chip = chip/255
        chip = transforms(image=chip)['image']
        pred = chip.float().to(device).unsqueeze(0)

        with torch.no_grad():
            try:
                pred = self.model.predict(pred)
            except Exception:
                pred = self.model(pred)

        pred = pred[0][0].cpu().detach().numpy()

        return pred

    def _get_aoi(self, aoi_path, src_image, meta, dst_crs):
        if aoi_path is not None:
            aoi = gpd.read_file(aoi_path).geometry
            aoi = aoi.to_crs(dst_crs)

            tile_arr, region_tfs = riomask.mask(
                src_image, aoi, all_touched=False, crop=True)
            meta['transform'] = region_tfs
            meta['height'] = tile_arr.shape[-2]
            meta['width'] = tile_arr.shape[-1]

            aoi_bounds = rasterio.features.rasterize(
                        shapes=aoi.boundary.geometry.tolist(),
                        out_shape=(tile_arr.shape[-2], tile_arr.shape[-1]),
                        transform=region_tfs,
                        default_value=255)

        else:
            tile_arr = src_image.read()
            try:
                aoi_bounds = cv2.copyMakeBorder(
                    np.zeros(tile_arr.shape),
                    10, 10, 10, 10,
                    cv2.BORDER_CONSTANT,
                    value=255)
            except Exception:
                aoi_bounds = np.zeros(tile_arr[0].shape)

        return aoi_bounds, tile_arr, meta

    def _fix_duplicates(self, gdf):

        gdf['area_meters'] = gdf.area
        gdf = gdf.sort_values('area_meters', ascending=False)
        df = gdf.copy()

        for idx, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
            poly = row['geometry']
            dublicates = gdf.within(poly)
            if dublicates.sum() > 1:
                df = df[~df.geometry.geom_equals(poly)]

        return df

    def _filter_polygons(self, polygons_list,
                         aoi=None, src_crs='EPSG:4326',
                         dst_crs='EPSG:4326',
                         min_poly_area=1e3,
                         max_poly_area=10e6):

        ring_ext = 20

        if aoi is not None:
            aoi = gpd.read_file(aoi)
            aoi = aoi.to_crs(src_crs)
            aoi_ext = aoi.geometry.values[0].buffer(ring_ext)
            aoi_ext = LinearRing(aoi_ext.exterior.coords)

        df = gpd.GeoDataFrame({"geometry": polygons_list},
                              crs=src_crs)

        if 'UTM zone' not in df.crs.name:
            print(f'Warning: {df.crs.name} coordinates reference\
            system might introduce inaccturate geometries estimation')

        df['length_ratio'] = df.length / df.area

        # drop only first large area element, it is usually the full aoi
        max_id = -1
        for i, item in df.iterrows():
            if item.geometry.area == df.area.max():
                max_id = i
                break

        df = df.drop(max_id)

        df.geometry = df.geometry.buffer(0)

        if aoi is not None:
            df = df[~df.geometry.intersects(aoi_ext)]

        df = df[df.area > min_poly_area]
        df = df[df.area < max_poly_area]
        df = df.to_crs(dst_crs)

        return df

    def raster_prediction(self,
                          raster_dir,
                          out_raster_path=None,
                          aoi_path=None,
                          transforms=val_tfs,
                          pred_window=256,
                          step=224,
                          raster_format='tif',
                          conf_thresh=0.5,
                          bands=['NDVI', 'NIR', 'TCI']):

        aoi = gpd.read_file(aoi_path)

        tci_path = [os.path.join(raster_dir, x) for x in os.listdir(
            raster_dir) if '_TCI_' in x][0]
        tci_tile = rasterio.open(tci_path)

        if 'NIR' in bands:
            nir_tile = rasterio.open(tci_path.replace('_TCI_', '_B08_'))
        if 'NDVI' in bands:
            ndvi_name = tci_path.replace('_TCI_', '_NDVI_').replace('.jp2', '.tif')
            ndvi_tile = rasterio.open(ndvi_name)

        meta = tci_tile.meta
        aoi = aoi.to_crs(tci_tile.crs)

        try:

            image, region_tfs = riomask.mask(
                tci_tile, aoi.geometry, all_touched=True, crop=True)
            image_nir, _ = riomask.mask(
                nir_tile, aoi.geometry, all_touched=True, crop=True)
            if 'NDVI' in bands:
                image_ndvi, _ = riomask.mask(
                    ndvi_tile, aoi.geometry, all_touched=True, crop=True)

        except Exception as e:
            print(e, tci_path, aoi_path)

        image_nir = max_scale(image_nir)
        if 'NDVI' in bands:
            image_ndvi = scale_ndvi(image_ndvi)
        else:
            image_ndvi = None

        self.model.eval()

        src_format = tci_path.split('.')[-1]
        if out_raster_path is None:
            out_raster_path = tci_path.replace('.', '_prediction.')

        out_raster_path = out_raster_path.replace(src_format, raster_format)
        if tci_path.split('.')[-1] != raster_format:
            meta['driver'] = drivers[raster_format]

        with rasterio.open(tci_path) as src:
            aoi_bounds, image, meta = self._get_aoi(
                aoi_path, src, meta, src.crs)

        mask = self._predict_array(image=image,
                                   image_nir=image_nir,
                                   image_ndvi=image_ndvi,
                                   bands=bands,
                                   aoi_bounds=aoi_bounds,
                                   pred_window=pred_window,
                                   step=step,
                                   conf_thresh=conf_thresh)
        meta['count'] = 1
        self._write_raster(mask, out_raster_path, meta)
        print(f'Writing raster: {out_raster_path}')

        return out_raster_path

    def _predict_array(self, image, image_nir,
                       image_ndvi, bands, aoi_bounds,
                       pred_window=256, step=224,
                       conf_thresh=0.5):

        side_padding = (pred_window - step)//2
        mask = np.zeros((image[0].shape[0]+2*side_padding+pred_window,
                         image[0].shape[1]+2*side_padding+pred_window))

        for i in tqdm(range(mask.shape[-1]//step)):
            for j in range(mask.shape[-2]//step):

                width, heigth = int(i*step), int(j*step)

                img_chip = image[:, heigth:heigth+pred_window,
                                 width:width+pred_window]
                nir_chip = image_nir[:, heigth:heigth+pred_window,
                                     width:width+pred_window]
                if 'NDVI' in bands:
                    ndvi_chip = image_ndvi[:, heigth:heigth+pred_window,
                                         width:width+pred_window]

                if img_chip.sum() < 1:
                    continue

                if (img_chip.shape[-1] != pred_window) or (
                    img_chip.shape[-2] != pred_window):

                    img = np.zeros((image.shape[0], pred_window, pred_window))
                    img[:, :img_chip.shape[-2], :img_chip.shape[-1]] = img_chip

                    nir = np.zeros((image_nir.shape[0], pred_window, pred_window))
                    nir[:, :nir_chip.shape[-2], :nir_chip.shape[-1]] = nir_chip
                    if 'NDVI' in bands:
                        ndvi = np.zeros((image_ndvi.shape[0], pred_window, pred_window))
                        ndvi[:, :nir_chip.shape[-2], :nir_chip.shape[-1]] = ndvi_chip

                else:
                    img = img_chip
                    nir = nir_chip
                    if 'NDVI' in bands:
                        ndvi = ndvi_chip
                try:
                    if ('NIR' in bands) and ('NDVI' in bands):
                        chip = np.stack(
                            [nir[0], ndvi[0], img[0], img[1], img[2]], axis=-1)
                    elif 'NIR' in bands:
                        chip = np.stack([nir[0], img[0], img[1], img[2]], axis=-1)
                    elif 'NDVI' in bands:
                        chip = np.stack([nir[0], img[0], img[1], img[2]], axis=-1)
                    else:
                        chip = img
                except Exception:
                    continue

                pred = self._predict_chip(
                    chip, self.device, conf_thresh, val_tfs)
                pred = pred[
                    side_padding:pred_window-side_padding,
                    side_padding:pred_window-side_padding]*255

                mask[heigth+side_padding:heigth+step+side_padding,
                     width+side_padding:width+step+side_padding] = pred
        mask = mask[:image[0].shape[0], :image[0].shape[1]]
        mask[aoi_bounds == 255] = 255
        return mask

    def process_raster_predictions(self,
                                   raster_path,
                                   shapes_path=None,
                                   aoi_path=None,
                                   conf_thresh=0.5,
                                   dst_crs='EPSG:4326',
                                   min_poly_area=1e3,
                                   max_poly_area=10e6):

        if shapes_path is None:
            shapes_path = raster_path.split('.')[0]+'_prediction.geojson'
        mask, crs, meta = self._load_tile(raster_path)
        if mask.shape[0] == 1:
            mask = mask[0]

        _, contours = self.get_contours(mask, int(conf_thresh*255))
        polygons = self.polygonize(contours, meta)

        df = self._filter_polygons(polygons, aoi_path,
                                   src_crs=crs, dst_crs=crs,
                                   min_poly_area=min_poly_area,
                                   max_poly_area=max_poly_area)

        return df

    def get_contours(self, mask, threshold=128):

        skeleton = morph.skeletonize(
            np.where(mask > threshold, 1, 0), method='lee')

        contours, _ = cv2.findContours(
            skeleton,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE)

        new_mask = np.zeros(mask.shape)
        new_mask = cv2.drawContours(new_mask, contours, -1, 255, 1)
        return new_mask, contours

    def polygonize(self, contours, meta, transform=True):
        """Credit for base setup: Michael Yushchuk. Thank you!"""
        polygons = []
        for i in range(len(contours)):
            c = contours[i]
            n_s = (c.shape[0], c.shape[2])
            if n_s[0] > 2:
                if transform:
                    polys = [tuple(i) * meta['transform'] for i in c.reshape(n_s)]
                else:
                    polys = [tuple(i) for i in c.reshape(n_s)]
                polygons.append(Polygon(polys))
        return polygons


class ChipsDataset(Dataset):
    """The ChipsDataset is a simple torch.utils.data.Dataset
    instance helping to index and supply image mini chips with
    corresponding masks and/or filters

    @param: index_df - pd.DataFrame object with necessary columns:
                        "filter_chip", "mask_chip", "tci_chip",
                        these contain a paths to mask to ignore,
                        label mask and tci images.

    @param: bands - bands to be stacked together as an input image
    """

    def __init__(self, root_dir="imagery/2018_dataset",
                bands=['tci', 'nir', 'ndvi'],
                transforms=train_tfs):

        images = os.listdir(os.path.join(root_dir, "tiles"))
        self.images = [os.path.join(root_dir, "tiles", x) for x in images]
        masks = [x.replace("tiles", "masks") for x in self.images]
        self.masks = [x.replace(".jpg", ".png") for x in masks]
        self.bands = bands
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        mask_ignore = np.where(np.array(mask) != 255, 255, 0)

        if self.transforms:
            transformed = self.transforms(
                image=np.array(img),
                gt_mask=np.array(mask),
                filter_mask=mask_ignore)

            image, mask, mask_ignore = (transformed['image'],
                                        transformed['gt_mask'],
                                        transformed['filter_mask'])
            try:
                mask = torch.from_numpy(mask)
            except:
                pass

            mask = mask.unsqueeze(0)/255
            image = image / 255
            mask_ignore = np.where(mask_ignore >= 250, True, False)

        return image.float(), mask.float(), torch.from_numpy(mask_ignore)
