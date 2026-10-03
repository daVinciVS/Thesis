# PAGI_25m07s_red_phase-Review Notes

## Detection/tracking limitation

One visible car was temporarily or completely missed during an occlusion event.

- Approximate raw frame: TBD
- Approximate clip frame: TBD
- Direction: TBD
- Traffic-light state: TBD
- Violation status: non-violation
- Observed cause: partial/full occlusion by surrounding traffic
- Effect on final violation detection: none directly, because the vehicle did not cross R_violation during red
- Effect on component evaluation: contributes to car-detection/tracking error analysis if this frame/sequence is included in ground truth.