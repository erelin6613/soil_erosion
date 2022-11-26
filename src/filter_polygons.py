def filter_sindex(shapes, filters, area_frac=0.25, dst_crs=None):
    """Filter geometries which intersect filtering geometry blob

    @param: shapes - geopandas.GeoDataFrame object with
                    geometries to be checked
    @param: filters - geopandas.GeoDataFrame object with
                    1 row only (f.e. unary_union of several
                    filtering geometires)
    @param: area_frac - a fraction below which intersection
                        is kept as a vlid geometry
    @param: dst_crs - destination CRS to which convert
                        resulting geojson, None will do no
                        conversion

    @return: geopandas.GeoDataFrame object with polygons
            which do not intersect filters blob

    """
    filters['geometry'] = filters.geometry.buffer(0)

    shapes['geometry'] = shapes.to_crs(
        filters.crs).buffer(0).geometry

    src_index = shapes.sindex
    candidates = []

    for geom in filters.itertuples():
        bounds = geom.geometry.bounds
        c = list(src_index.intersection(bounds))
        candidates += c

    candidates_idx = set(candidates)
    candidates = shapes.iloc[list(candidates_idx)]

    areas = candidates.area
    overlaps = candidates.intersection(filters.geometry.values[0]).area

    shapes['overlap_frac'] = overlaps/areas
    shapes = shapes[(
        shapes.overlap_frac < area_frac) | (shapes.overlap_frac.isna())]

    if dst_crs is not None:
        shapes = shapes.to_crs(dst_crs)

    return shapes


def filter_polygons(shapes, filters, area_frac=0.25, dst_crs=None):
    """Filter geometries which intersect filtering geometry blob

    @param: shapes - geopandas.GeoDataFrame object with
                    geometries to be checked
    @param: filters - geopandas.GeoDataFrame object with
                    1 row only (f.e. unary_union of several
                    filtering geometires)
    @param: area_frac - a fraction below which intersection
                        is kept as a vlid geometry
    @param: dst_crs - destination CRS to which convert
                        resulting geojson, None will do no
                        conversion

    @return: geopandas.GeoDataFrame object with polygons
            which do not intersect filters blob

    """
    filters['geometry'] = filters.geometry.buffer(0)

    shapes['geometry'] = shapes.to_crs(filters.crs).buffer(0).geometry

    areas = shapes.area
    overlaps = shapes.intersection(filters.geometry.values[0]).area

    shapes['overlap_frac'] = overlaps/areas
    shapes = shapes[shapes.overlap_frac < area_frac]

    if dst_crs is not None:
        shapes = shapes.to_crs(dst_crs)

    return shapes

def fix_invalid_polys(polygons):
    valid = polygons[polygons.geometry.is_valid]

    return valid
