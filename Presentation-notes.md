topic: running posture analysis

Day 1 (07/09/26)

We had 2 models in discussion for perception: YOLO and MediaPipe
YOLO is the go to one.
With MediaPipe, there were more poses available including additional points for front foot and back foot. It's not there in YOLO. So we considered this.
Tried both YOLO and MediaPipe examples - both were doable.
Issue with mediapipe: not compatible with latest python version which denotes deprecation/non-maintenance. It also required us to download a .task file manually for it to work. 'uv add mediapipe' was not enough for using the package. 
We looked into running posture issues to understand the relevant landmarks. Only 10-20% issues required these additional poses from mediapipe. Decided to omit those and go with YOLO model as it'll include the relevant issues.
