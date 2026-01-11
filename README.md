# DiffAVSE

Audio-Visual Speech Enhancement using Diffusion Models.

University of Bristol EEME30003 Major Project, 2025-26.

## Changelog

`2026-01-13`: Project direction confirmed: diffusion-based AVSE

`2026-01-10`: Repository initialized

## Overview

This project explores using lip visual information to guide diffusion-based speech enhancement. The idea is that lip movements are immune to acoustic noise, so they can help recover cleaner speech in noisy conditions.

This repository builds on two existing codebases:
- [SGMSE](https://github.com/sp-uhh/sgmse) - Diffusion-based speech enhancement
- [Auto-AVSR](https://github.com/mpc001/auto_avsr) - Visual speech recognition (we use the visual encoder)

Our contribution is adding visual conditioning to the diffusion score network.

```
Input:  Noisy Audio + Lip Video
Output: Enhanced Audio
```

## Status

Current focus: reproduce SGMSE audio-only baseline and apply for LRS3 dataset access.

## Documentation

- Team & timeline: see [`docs/PLAN.md`](docs/PLAN.md)
- Weekly log: see [`docs/LOG.md`](docs/LOG.md)
- Results: see [`docs/RESULTS.md`](docs/RESULTS.md)

## Reproducibility

All experiments must have:
1. A config file under `configs/`
2. A log under `experiments/`

## References

### Core Papers

| Paper | Venue | Year | Usage |
|-------|-------|------|-------|
| [SGMSE+](https://arxiv.org/abs/2303.15299) | TASLP | 2023 | Diffusion SE framework |
| [Auto-AVSR](https://arxiv.org/abs/2303.14307) | ICASSP | 2023 | Visual encoder |
| [RT-LA-VocE](https://arxiv.org/abs/2407.07825) | Interspeech | 2024 | Real-time AVSE reference |

### Additional Papers

| Paper | Venue | Year |
|-------|-------|------|
| [VisualVoice](https://arxiv.org/abs/2101.03149) | CVPR | 2021 |
| [Looking to Listen](https://arxiv.org/abs/1804.03619) | SIGGRAPH | 2018 |
| [DAVSE](https://www.isca-archive.org/avsec_2024/chen24b_avsec.pdf) | AVSEC | 2024 |

### Code

| Repository | Description |
|------------|-------------|
| [SGMSE](https://github.com/sp-uhh/sgmse) | Diffusion-based speech enhancement |
| [Auto-AVSR](https://github.com/mpc001/auto_avsr) | Audio-visual speech recognition |
| [AVSE Challenge](https://github.com/cogmhear/avse_challenge) | COG-MHEAR baseline & evaluation |

## Acknowledgements

- [SGMSE](https://github.com/sp-uhh/sgmse) by [Welker et al.](https://arxiv.org/abs/2303.15299)
- [Auto-AVSR](https://github.com/mpc001/auto_avsr) by [Ma et al.](https://arxiv.org/abs/2303.14307)
- [COG-MHEAR AVSE Challenge](https://challenge.cogmhear.org/) organizers

## License

Code will be released under an open-source license (TBD). Upstream components retain their original licenses.

## Contact

**Supervisor:** Dr. Fadi Karameh

For questions, please open an issue.
