# Real-Time-Object-Detection-in-Adverse-Weather-Conditions-with-Selective-Video-Frame-Restoration
Adverse weather conditions such as rain, fog, and snow significantly degrade the
performance of real-time 



object detection systems by reducing image clarity, introducing
occlusions, and distorting semantic features. Traditional approaches to handle such scenarios
often rely on uniform restoration of all video frames using deep learning models, which, while
improving visibility, incur considerable computational costs and latency, making them
unsuitable for real-time deployment in resource-constrained environments. To address these
limitations, this project proposes a modular and adaptive framework that selectively restores
only those frames which are both severely degraded and contain semantically relevant content.
The pipeline begins with a frame filtering module that utilizes YOLOv8n to identify
object-containing frames, while heuristic metrics such as Laplacian variance and contrast
estimation assess the extent of visual degradation. Frames classified as severely degraded and
object-relevant are passed through the WeatherFormer model, a transformer-based restoration
architecture capable of handling multi-weather distortions in a single pass. The restored and
unaffected frames are then processed by YOLOv11, an advanced object detection network
incorporating efficient spatial attention mechanisms and multi-scale feature fusion for
improved detection accuracy.
The system concludes with a merging module that reassembles the processed frames
into a coherent, temporally consistent output video. This selective and intelligent processing
strategy significantly reduces computational load while preserving detection fidelity.
Experimental evaluation using both the curated All-Weather dataset and real-world traffic
surveillance footage demonstrates that the proposed framework effectively balances restoration
quality, detection accuracy, and system efficiency, making it highly suitable for deployment in
autonomous navigation, smart surveillance, and edge-based monitoring systems

This is final video after passing through the pipeline from real time scenarioes, More development will be included in the future.

https://github.com/user-attachments/assets/a638cfda-908d-4425-8936-3cc12405f97c
