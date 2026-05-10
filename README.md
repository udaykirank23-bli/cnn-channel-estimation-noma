# CNN-Based Channel Estimation for LIS-Aided mmWave Massive MIMO with NOMA

## Overview
This project implements a deep learning framework for channel estimation 
in Large Intelligent Surface (LIS) assisted mmWave Massive MIMO systems, 
extended with NOMA (Non-Orthogonal Multiple Access) sum-rate analysis.

## Features
- CNN-based direct channel estimation (ChannelNet)
- Wideband mmWave channel generation
- NOMA two-user sum-rate computation
- Dataset generation for training and testing
- Early stopping and training utilities

## Tech Stack
- MATLAB (Deep Learning Toolbox)
- Convolutional Neural Networks (CNN)
- mmWave Massive MIMO
- LIS / RIS (Reconfigurable Intelligent Surface)
- NOMA

## File Structure
| File | Description |
|------|-------------|
| `NOMAf_2U.m` | NOMA two-user sum-rate analysis |
| `train_ChannelNet.m` | CNN training script |
| `generate_channel_H_LIS.m` | LIS channel generation |
| `generate_dataset_DC_V3000.m` | Test dataset generation |
| `generate_dataset_DC_V3000_1.m` | Training dataset generation |
| `array_respones.m` | Array response vector computation |
| `calculateRate_WB_SU.m` | Wideband rate calculation |
| `non_overlapbeam.m` | Non-overlapping beam selection |
| `stopIfAccuracyNotImproving.m` | Early stopping callback |

## Requirements
- MATLAB R2020a or later
- Deep Learning Toolbox
- Communications Toolbox

## Credits
Base ChannelNet architecture adapted from:
> A. M. Elbir et al., "Deep Channel Learning for Large Intelligent Surfaces 
> Aided mm-Wave Massive MIMO Systems," IEEE Wireless Communications Letters, 2020.
> GitHub: https://github.com/meuseabe/deepChannelLearning4RIS

## Original Contributions
- NOMA two-user sum-rate analysis (`NOMAf_2U.m`)
- Extended dataset generation pipeline (`generate_dataset_DC_V3000.m`)

## Institution
Indian Institute of Information Technology, Sri City (IIITS)

## License
MIT License — see LICENSE file for details.
