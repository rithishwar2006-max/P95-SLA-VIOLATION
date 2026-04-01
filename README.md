Overview
Developed a binary classification machine learning model to predict P95 tail latency SLA violations in complex microservice architectures.
By analysing the Alibaba Cluster Trace dataset, the system proactively identifies latency spikes to enable better resource allocation and system reliability.

Tech Stack & Libraries
  Language:Python
  Data Processing: Pandas, NumPy
  Machine Learning: Scikit-Learn 
  Data Visualization: Matplotlib, Seaborn

 Dataset
This project utilizes the Alibaba Microservice Cluster Trace Dataset.
Target Variable:Binary classification (1 = SLA Violation Predicted, 0 = Normal Operation)

System Architecture & Workflow
1.Data Preprocessing:Cleaning missing values and normalizing cluster metrics.
2.Feature Engineering:Selecting the most impactful features contributing to tail latency.
3.Model Training:Training the binary classifier using various Classification models
4.Evaluation:Testing the model against unseen data to measure predictive accuracy.
