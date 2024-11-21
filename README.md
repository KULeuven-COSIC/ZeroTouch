# ZeroTouch: Reinforcing RSS for Secure Geofencing

> **Abstract**. Geofencing, the virtual demarcation of physical spaces, is widely used for managing the localisation of Internet of Things (IoT) devices. However, traditional techniques face significant security limitations indoors due to signal interference, variability, and susceptibility to spoofing, making accurate and secure boundary enforcement challenging. These methods frequently overlook critical security concerns, leaving them vulnerable to adversarial attacks. Moreover, existing methods often require time-intensive calibrations or additional hardware, reducing their scalability. In this work, we propose *ZeroTouch*, a machine learning-based system that leverages Received Signal Strength (RSS) measurements from multiple receivers to address these challenges. While RSS-based localisation is inherently insecure, we demonstrate that by combining RSS readings and detecting anomalies using an autoencoder model, our approach can effectively verify whether a device is inside or outside a defined boundary. *ZeroTouch* eliminates manual calibration, removes the *human-in-the-loop* element, and simplifies deployment. We evaluate our solution in a realistic simulated environment and achieve very good accuracy in distinguishing between in-room and out-of-room devices. Furthermore, we introduce a framework to customise accuracy levels, making *ZeroTouch* adaptable to varying security requirements, thus making our solution both flexible and scalable.

We are submitting the artefact of our work, `ZeroTouch`, for evaluation. Submitted at `WiSec 2025 Cycle 1` with submission number `#18`.

## Directory Structure Overview

This repository is organised into two primary parts:

1. **MATLAB**: This folder houses all the necessary files related to the localisation procedures and security checks.

2. **Wireless InSite**: This folder contains all the files pertinent to the RSS simulations.

## Further Information

Each folder contains a `README.md` file that provides detailed instructions and information specific to the contents and procedures within that folder. Please refer to these files for comprehensive guidance on how to run the simulations.
