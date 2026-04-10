# 🔥 High-Precision Hot Rolling Force Prediction

**Complete PyTorch Implementation** of the paper:  
**"High-precision rolling force prediction for hot rolling based on multimodal physical information and reinforcement learning"**  
*(Journal of Manufacturing Processes 151 (2025) 655–678)*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Overview

This repository provides a **production-ready, fully reproducible** implementation of the hybrid modeling framework proposed in the paper. It integrates:

- **Differentiable Johnson-Cook constitutive equations** coupled with Zener-Hollomon parameter (Section 2)
- **Multi-channel CNN-RNN-Attention network** (4 parallel channels: process timing, composition-performance, equipment state, physical coupling) with Inception + Bi-LSTM + hierarchical attention (Section 4.2)
- **DDPG reinforcement learning** for adaptive hyper-parameter tuning (learning rate, physics loss weight λ, dropout, attention temperature) (Section 4.3)
- **Residual-Physics dual-driven anomaly detection** with Monte Carlo Dropout uncertainty estimation + LightGBM classifier (Section 5)
- **Incremental correction mechanisms**: Kalman filter (sensor drift), online JC parameter update (material offset), MPC-based process adjustment (Section 5.3)

The model achieves **≤3% relative error** on F1/F7 stands and **<1%** on intermediate stands (as reported in Table 5 of the paper), with real-time anomaly correction reducing MAE by ~28–30% under disturbances.

**All code is written in English with detailed docstrings, type hints, logging, TensorBoard support, and unit tests.**

## ✨ Key Features

- ✅ **Physics-informed layer** – fully differentiable Johnson-Cook (learnable θ parameters)
- ✅ **Multi-channel architecture** – exactly matches Fig. 7 and Section 4.2
- ✅ **DDPG agent** – dynamic hyper-parameter optimization every 5 epochs
- ✅ **Anomaly detection & correction** – residual + physical consistency + LightGBM (4 anomaly types)
- ✅ **Synthetic data generator** – matches Table 1 variable ranges and realistic physics correlations
- ✅ **Online inference** – real-time prediction with anomaly correction (<0.05 s per stand)
- ✅ **Training pipeline** – with early stopping, gradient clipping, and TensorBoard logging
- ✅ **Reproducibility** – fixed seeds, config-driven, full test suite

## 📁 Project Structure
