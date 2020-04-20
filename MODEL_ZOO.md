# Model Zoo

## Settings

- For all datasets, the results are reported on the test set.

- The inference time is averaged over 100 runs, with batch size 1 on a single NVIDIA V100 GPU.

  For Scene Flow dataset, the inference resolution is 576x960. For KITTI dataset, the resolution is 384x1248.

## Scene Flow

| Method       | EPE  | > 1px | Params | Time(ms) |                           Download                           |
| ------------ | :--: | :---: | :----: | :------: | :----------------------------------------------------------: |
| StereoNet-AA | 1.08 | 12.9  | 0.53M  |    17    | [model](https://drive.google.com/open?id=1CimLtdQvh1dl60rCyVZZCojF4XNmFkj3) |
| GC-Net-AA    | 0.98 | 10.8  | 2.15M  |    91    | [model](https://drive.google.com/open?id=18NisQ6vp4oMrv-cVARLmlRe8prIfdGIL) |
| PSMNet-AA    | 0.97 | 10.2  | 4.15M  |    77    | [model](https://drive.google.com/open?id=1m5swiWdJ7PK9Oei0NK8osTFWwq2n14Ri) |
| GA-Net-AA    | 0.87 |  9.2  | 3.68M  |    57    | [model](https://drive.google.com/open?id=1i7PJ8YVbviJe7Xc_RhBYxfvLD9QTjsBM) |
| AANet        | 0.87 |  9.3  | 3.93M  |    68    | [model](https://drive.google.com/open?id=1_OuMEE5v5DcSUuKUZeCBZ17TZgxi4--h) |
| AANet+       | 0.72 |  7.4  | 8.44M  |    64    | [model](https://drive.google.com/open?id=1m3Wo2k7w3OlVaDYdIm-x9K8_frQ4Vq9u) |

## KITTI 2012

| Method | Out-Noc | Out-All | Params | Time(ms) |                           Download                           |
| ------ | :-----: | :-----: | :----: | :------: | :----------------------------------------------------------: |
| AANet  |  1.91   |  2.42   | 3.93M  |    62    | [model](https://drive.google.com/open?id=1G2M6w0RIe6kZU_iHpOyrYg_cVd1Jc34Q) |
| AANet+ |  1.55   |  2.04   | 8.44M  |    60    | [model](https://drive.google.com/open?id=1a2P41zmaNVVcU-Yux1AMXtgMS1fPlsGj) |

## KITTI 2015

| Method | D1-bg | D1-all | Params | Time(ms) |                           Download                           |
| ------ | :---: | :----: | :----: | :------: | :----------------------------------------------------------: |
| AANet  | 1.99  |  2.55  | 3.93M  |    62    | [model](https://drive.google.com/open?id=1Qa1rSWQDBcW4D6LvvjDztfTLNGA_rgGn) |
| AANet+ | 1.65  |  2.03  | 8.44M  |    60    | [model](https://drive.google.com/open?id=1m0z8e2Ndau_eFR3BETffzjkNXhJNHAiS) |



