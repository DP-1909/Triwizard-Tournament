# Socket Centroids

**Saved:** 2026-11-12 00:10:19 (iteration 1, rewritten every iteration)  
**Frames averaged:** 30 (median), ArUco marker seen in 0  
**Point:** centre of the socket opening, at the plate's top surface. Units: metres.

## Frames

| Frame | Origin | Axes |
| :--- | :--- | :--- |
| Camera | RealSense colour lens | x right, y down, z forward |
| Marker | ArUco marker centre | x along marker top edge, y up the marker, z up out of the table |
| Robot | robot base | base at (0.0400, 0.0000, 0.0000) m, yaw 0.0 deg in the marker frame (**placeholder: measure before use**) |

## Socket centres

| Socket | Seen | Pixel (u, v) | Robot (X, Y, Z) | Marker (X, Y, Z) | Camera (X, Y, Z) |
| :--- | ---: | :--- | :--- | :--- | :--- |
| **triangle** | 26/30 | (347.8, 54.2) | `n/a` | `n/a` | `(0.0232, -0.1765, 0.5290)` |
| **square** | 21/30 | (352.0, 135.3) | `n/a` | `n/a` | `(0.0262, -0.1028, 0.5170)` |
| **hexagon** | 30/30 | (356.8, 220.4) | `n/a` | `n/a` | `(0.0294, -0.0291, 0.5010)` |
| **circle** | 15/30 | (361.6, 310.2) | `n/a` | `n/a` | `(0.0325, 0.0443, 0.4870)` |

**Layout (camera frame):** triangle -> square -> hexagon -> circle; spacing 7.5, 7.6, 7.5 cm; max off-line 1.4 mm

## CSV (for the trajectory planner)

```csv
socket,robot_x_m,robot_y_m,robot_z_m,marker_x_m,marker_y_m,marker_z_m,cam_x_m,cam_y_m,cam_z_m
triangle,,,,,,,0.0232,-0.1765,0.5290
square,,,,,,,0.0262,-0.1028,0.5170
hexagon,,,,,,,0.0294,-0.0291,0.5010
circle,,,,,,,0.0325,0.0443,0.4870
```
