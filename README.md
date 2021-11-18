The SmartErasor Project
In SmartErasor, you are required to implement a program to erase several characters auto-
matically and gradually in a given video, while keeping the background texture unchanged 3
perceptively.
The input is a video segment 1, in which there is a foreground ”Can you erase me?”,
and the background of the video is a newspaper texture, as shown in Figure 1. You will also notice that on top of the noisy characters there is a pen tip that keeps moving from right side to the left as the video runs. For example, in Figure 1 (a) that is the 100th frame extracted from the video, the current pen tip’s location is on top of noisy word “me?”, so your program should use this location as an indicator, and erase the noisy words between this location and the right side of the noisy sentence. Several output frames are shown in Figure 2
The location of the pen tip can be effectively obtained by using the visual tracking technique covered in class, and the erasing work can be done by the texture impainting techniques.
