# PAGI_25m07s_red_phase — Traffic-Light Annotation Notes

## Ground-truth phase ranges

| Raw frame range | Clip-frame range | Ground-truth state |
|---|---:|---|
| 44250–44270 | 0–20 | Yellow |
| 44271–44828 | 21–578 | Red |
| 44829–44918 | 579–668 | Green |
| 44919–45249 | 669–999 | Yellow |

## Observed HSV limitation

During portions of the manually verified yellow phase, the HSV traffic-light estimator may predict green when yellow pixels are weak or not detected. The manual ground-truth annotation remains yellow whenever the yellow lamp is visibly active.