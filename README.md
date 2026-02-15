# Embedded RGB-D Segmentation for Indoor Spaces

> Work focused on recognizing walkable paths, walls, and doors by leveraging the Jetson Nano GPU.

## .git

[Git repository (Lottie)](https://lottie.host/a6ba3ec7-3044-4141-b3f3-a356e35e1824/QlR0IclYTr.lottie) 

## Key requirements

1. Use an Intel RealSense D435i camera to capture RGB-D data.
2. Implement the system on the NVIDIA Jetson Nano 4 GB Developer Kit.
3. Optimize system performance through GPU acceleration using CUDA 10.2 available on the Jetson Nano.
4. Develop the system in Python 3.8.
5. Implement the graphical user interface (GUI) using the Tkinter library on Tcl/Tk 8.6.

## Activities and status

- [x] ~~1. Conduct a literature review to define functional and non-functional requirements.~~
  - [x] ~~1.1 Prepare a comparative analysis of technical background to identify functional requirements relevant to the proposed system.~~
  - [x] ~~1.2 Analyze the common limitations found in the consulted technical background.~~
  - [x] ~~1.3 Document the functional and non-functional requirements of the system based on the identified capabilities and limitations.~~
- [x] 2. Build a test dataset with annotations for recognizing walls, doors, and walkable paths in structured indoor environments, using captured point clouds or existing databases.
  - [x] ~~2.1 Collect data using RGB-D cameras or from existing databases in structured indoor environments.~~
  - [x] ~~2.2 Process point clouds to remove noise and optimize spatial structure.~~
  - [x] ~~2.3 Provide semantic annotations for walls, doors, walkable paths, and other objects in the collected data.~~
  - [x] 2.4 Validate the dataset in a structured format for training, with documentation of the results obtained.
- [ ] 3. Design a recognition model for walls, doors, walkable paths, and other objects using the test dataset.
  - [x] ~~3.1 Create the conceptual diagram of the recognition model according to the SCRUM methodology.~~
  - [x] ~~3.2 Define user stories in a project planning tool according to the SCRUM methodology.~~
  - [x] ~~3.3 Prioritize the backlog with the technical and operational items needed for training the model, following the SCRUM methodology.~~
  - [x] 3.4 Design the recognition model according to the SCRUM methodology.
  - [ ] 3.5 Prepare the technical report of the recognition model along with the design sketches of the graphical user interface (GUI).
- [ ] 4. Implement on an embedded system the recognition model for walls, doors, walkable paths, and other objects.
  - [x] ~~4.1 Configure the embedded system environment to ensure compatibility and deployment of the trained model.~~
  - [x] ~~4.2 Implement the perception module by integrating the recognition model into the embedded system.~~
  - [x] 4.3 Implement the processing module to apply the integrated model to input data, generating classified outputs under controlled operating conditions.
  - [x] 4.4 Implement the feedback module with performance metric logging.
  - [ ] 4.5 Document the integration of the recognition model into the embedded system.
- [ ] 5. Validate the application's functionality and accuracy in recognizing the selected classes in real environments through a test protocol.
  - [x] 5.1 Define the validation protocol for recognition accuracy and functionality, including metrics, scenarios, and acceptance criteria.
  - [ ] 5.2 Execute the test protocol in real environments to validate recognition accuracy and system functionality.
  - [ ] 5.3 Record the performance results, errors, and technical observations.

**Summary:** 18 of 25 activities completed (7 pending).

Current progress: 72%  
`[####################################--------------]` 18/25  

Current status:
- Core dataset and annotation workflow completed.
- Embedded integration, processing, and metric logging completed.
- GUI execution flow and model-loading feedback completed.
- Pending closure focused on technical documentation and real-environment validation.

<p align="center">
  <img alt="Progress 72%" src="https://img.shields.io/badge/Progress-72%25-00b86b?labelColor=111&color=00b86b" />
  <br/>
  <img alt="Progress chart 72%" src="https://quickchart.io/chart?c=%7B%0A%20%20type%3A%20%27doughnut%27%2C%0A%20%20data%3A%20%7B%0A%20%20%20%20datasets%3A%20%5B%7B%0A%20%20%20%20%20%20data%3A%20%5B72%2C%2028%5D%2C%0A%20%20%20%20%20%20backgroundColor%3A%20%5B%27%2300b86b%27%2C%20%27%23e5e7eb%27%5D%2C%0A%20%20%20%20%7D%5D%0A%20%20%7D%2C%0A%20%20options%3A%20%7B%0A%20%20%20%20plugins%3A%20%7Blegend%3A%20false%7D%2C%0A%20%20%20%20cutout%3A%20%2770%25%27%0A%20%20%7D%0A%7D" />
</p>
