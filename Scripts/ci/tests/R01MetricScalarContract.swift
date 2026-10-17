import VectorCore

#if REJECT_DOUBLE
func nearest<M: DistanceMetric>(metric: M, query: DynamicVector) where M.Scalar == Double {
    _ = TopKSelection.nearest(k: 1, query: query, candidates: [query], metric: metric)
}
func batch<M: DistanceMetric>(metric: M, query: DynamicVector) where M.Scalar == Double {
    _ = TopKSelection.batchNearest(k: 1, queries: [query], candidates: [query], metric: metric)
}
func collection<M: DistanceMetric, C: VectorCollection>(metric: M, value: C, query: C.Vector) where M.Scalar == Double {
    _ = value.search(query: query, k: 1, metric: metric)
}
#else
func nearest<M: DistanceMetric>(metric: M, query: DynamicVector) where M.Scalar == Float {
    _ = TopKSelection.nearest(k: 1, query: query, candidates: [query], metric: metric)
}
func batch<M: DistanceMetric>(metric: M, query: DynamicVector) where M.Scalar == Float {
    _ = TopKSelection.batchNearest(k: 1, queries: [query], candidates: [query], metric: metric)
}
func collection<M: DistanceMetric, C: VectorCollection>(metric: M, value: C, query: C.Vector) where M.Scalar == Float {
    _ = value.search(query: query, k: 1, metric: metric)
}
#endif
