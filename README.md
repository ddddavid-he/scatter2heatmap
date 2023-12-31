# Scatter to HeatMap
Python tool to transfer data points distributed in coordinations to matrix-formed heatmap.

Input Data: `numpy.ndarray` of coordinations `[[x,y], [x,y], .., [x,y]]`

Output Data: 2D `numpy.ndarray` of add up counts, which is in a size of `[H, W]`

If, in coordinations of samples, `(xmax-xmin)>(ymax-ymin)`, `H=grid_size` and `W=grid_size * round((xmax-xmin)/(ymax-ymin))`. When `(xmax-xmin)<=(ymax-ymin)`, `H=grid_size * round((xmax-xmin)/(ymax-ymin))` and `W=grid_size`



Demo:

| Scatter | HeatMap |
| :----: | :----: |
|<img src="https://github.com/ddddavid-he/scatter2heatmap/blob/main/demo/ra-scatter.png" width="90%"> | <img src="https://github.com/ddddavid-he/scatter2heatmap/blob/main/demo/ra-heatmap.png" width="90%">|
|<img src="https://github.com/ddddavid-he/scatter2heatmap/blob/main/demo/sin-scatter.png" width="90%"> | <img src="https://github.com/ddddavid-he/scatter2heatmap/blob/main/demo/sin-heatmap.png" width="90%">|





