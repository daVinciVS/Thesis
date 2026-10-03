# SIANG_multi_08m15s_red_occlusion - Review Notes

## Traffic-light-state failure

The pipeline predicted `green` for all 23 passage candidates, including the manual red phase from raw frames 15581 to 16135.

- Effect: No candidate was armed for red-light violation detection.
- Result: The pipeline reported zero violations despite three manually confirmed red-light violations: C07, C08, and C09.
- Error category: Traffic-light-state classification failure causing downstream violation false negatives.

## Detection and tracking limitations

- Pipeline track ID 11 later changed to ID 17.
- A white van stopped legally during red but was not continuously detected or tracked; it was intermittently associated with IDs 27 and 45.
- Pipeline track ID 21 changed to ID 33.
- Manual C12 changed from pipeline track ID 44 to 55 after occlusion by the white van.
- Pipeline track ID 57 was assigned to two overlapping cars during an occlusion event; the front car partially hid the rear car and both appeared similar in colour.
- Pipeline track ID 54 was reused for a different physical car. Manual C16 was later reassigned to pipeline track ID 73, while C19 received the reused track ID 54.