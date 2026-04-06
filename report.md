**TEAM NAME:** The uchihas
**Team Members and Roll No.:** 
- Sabarishvar  MM24B046
- Sanjai       NA24B018
- Dhipak Kumar NA24B007
---
# **SUMMARY:**
# **PROBLEM UNDERSTANDING:**
# **DATASET OVERVIEW:**
# **METHODOLOGY:**
# **SECONDARY OUTPUTS:**
## Route level fuel benchmarking:
- The core idea behind this is to: **“separate what the route costs from what the dumper/operator costs”**. If we manage to find the fuel consumed for a route independent of the vehicle traversing we could give a better estimate as to how much fuel was consumed as the vehicle took that route.
- The methodology for our codebase is as follows:
    1. Get the 3d points of the haul roads from the gpkg file 
    ```python
    try:
        haul_gdf = gpd.read_file(path, layer="haul_road")
        haul_xyz = layer_union_xyz(haul_gdf)           # the layer_union_xyz function calls the unary_union on all the haul road linestring geometries and then merges them all into one geometry
        haul_xy  = haul_xyz[:, :2] if len(haul_xyz) else np.zeros((0,2))
    ```
    2. the next process is to get every gps points near the haul road points which happens in the attach\_projected\_and_spatial() function. for an overview of the function it does the following:
       - first it converts the raw telemetry lat and long data into the projected coords using the **transformer** from the **pyproj** lib.
       - then it creates a dictionary called _spatial_ which looks like `spatial = {"mine_001": sp_obj, "mine_002": sp_obj}` where each of the sp_obj contains k-dimensional trees (for locating the haul roads nearby the points at that ts), mine boundaries and 3d points.
       - new cols required for final df are also added such as dist\_haul\_m, dist\_dump\_m, on\_haul\_ and a lot (refer the code !!!mention the line from the script!!!)
    3. now back to the `featurize_file_()` function, where we next compute the grade as per the gps points (grade here means steepness of the climb or the descent). also the grade is capped to 30% cause above that is gonna be unrealistic if you think about it. and then creates a col in df for the `haul_climb` and `haul_descent`.
    4. after creating the new cols for the dataframe we aggregate that to the main `agg_kw` dict like:
    ```python
    haul_cum_climb_m   = ("haul_climb",       "sum")
    haul_cum_descent_m = ("haul_descent",      "sum")
    haul_grade_abs_mean= ("haul_grade_pct",   lambda x: np.nanmean(np.abs(x)))
    haul_snap_z_mean   = ("haul_snap_z",      lambda x: np.nanmean(x))
    haul_snap_z_std    = ("haul_snap_z",      lambda x: np.nanstd(x))
    ```
    5. now one of the issues is that we dont have a col route but we do have the necessary data for building this.
    ```python
        ROUTE_FEATS = [
        "dump_dist_mean", "stock_dist_mean", "haul_dist_mean",
        "alt_mean", "cum_climb_m", "dist_km", "run_hrs",
        # New 3D terrain features — much richer route signal
        "haul_snap_z_mean",     # average elevation on haul road
        "haul_snap_z_std",      # elevation variability (hilly vs flat route)
        "haul_grade_abs_mean",  # average absolute grade (terrain difficulty)
        "haul_cum_climb_m",     # total climb on haul road (energy cost)
        "haul_cum_descent_m",   # total descent (regeneration potential)
        "bench_z_delta_mean",   # terrain adherence to bench contours
    ]
    ```
    this is for the cluster formation for the routes using the previously found data.
    refer this line for more info!!!!!
    and eventually after this clustering of the route leads to a df having every shift row in both train and test a `route_enc` col which ranges from 0 to 19 because of how KMeans works !!!! explain more about this too 
    
    6. physics benchmarking: instead of the approach of simple averaging we use ols (ordinary least squares). this answers the question of "If a perfectly efficient dumper ran this exact shift — no idling, no aggressive driving, no mechanical inefficiency then how much fuel would the terrain itself demand?" this is the purely physics problem and we use the following data:
        - dist_km which would be a function of the rolling resistance and also relates the fuel consumed.
        - haul\_cum\_climb_m: work done against gravity (computed from the 3d gpkg elevation data for each ts to each gps point.)
        - haul\_net\_lift_pos: this is irrecoverable energy ie. the energy spent climbing 80m is all burned as fuel. but the net positive lift assuming 20m represents the route's permanent elevation gain across the shift which is energy that had to be spent and could never come back even in theory.
    
    7. route benchmark table: `build_route_benchmarks()` function represents combining all of the data from the above steps.
## Dumper Efficiency Component: 
## Cycle segmentation methodology:
## Daily fuel consistency:
# **KEY FINDINGS & INSIGHTS:**
# **REFERENCES & TOOLS USED:**
# **CODEBASE:**
