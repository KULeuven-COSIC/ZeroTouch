# ZeroTouch: Reinforcing RSS for Secure Geofencing

> **Abstract**. Geofencing, the virtual demarcation of physical spaces, is widely used for managing the localisation of Internet of Things (IoT) devices. However, traditional localisation techniques face security challenges indoors due to signal interference and susceptibility to spoofing, often requiring extensive calibration or extra hardware, limiting scalability. In this work, we propose *ZeroTouch*, a machine learning-based system that leverages Received Signal Strength (RSS) measurements from multiple receivers to improve the security of geofencing without introducing additional deployment overhead. While RSS-based localisation is known to have inherent security limitations, we show that by aggregating RSS readings from multiple anchor points and detecting anomalies using an autoencoder model, *ZeroTouch* provides a practical and automated mechanism for verifying whether a device is inside or outside a defined boundary. Rather than serving as a standalone security mechanism, *ZeroTouch* enhances existing authentication frameworks by adding an additional *zero-touch* security layer that operates passively in the background. *ZeroTouch* eliminates manual calibration, removes the *human-in-the-loop* element, and simplifies deployment. We evaluate our solution in a realistic simulated environment and demonstrate that it achieves high accuracy in distinguishing between in-room and out-of-room devices, even in strong adversarial settings.

We are submitting the artefact of our work, `ZeroTouch`, for evaluation. Submitted at `SACMAT'25` with submission number `#34`.

## Directory Structure Overview

This repository is organised into two primary parts:

1. **MATLAB**: This folder houses all the necessary files related to the localisation procedures and security checks.

2. **Wireless InSite**: This folder contains all the files pertinent to the RSS simulations.

## Further Information

Each folder contains a `README.md` file that provides detailed instructions and information specific to the contents and procedures within that folder. Please refer to these files for comprehensive guidance on how to run the simulations.

## Additional Visualisations and Resources

Additional visualisations and results from the simulations, due to size constraints, have been uploaded and can be accessed [here](https://mega.nz/folder/xxclXYiR#Do4h264nC4XnJwbCjHicMA). 

Furthermore, a video showcasing the apartment setup used in the simulations is also available at the same link. These resources provide further insights into the experimental environment and results, offering a comprehensive view of the `ZeroTouch` system and its evaluation.
