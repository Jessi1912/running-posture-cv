topic: running posture analysis

Day 1 (07/09/26)

We had 2 models in discussion for perception: YOLO and MediaPipe
YOLO is the go to one.
With MediaPipe, there were more poses available including additional points for front foot and back foot. It's not there in YOLO. So we considered this.
Tried both YOLO and MediaPipe examples - both were doable.
Issue with mediapipe: not compatible with latest python version which denotes deprecation/non-maintenance. It also required us to download a .task file manually for it to work. 'uv add mediapipe' was not enough for using the package. 
We looked into running posture issues to understand the relevant landmarks. Only 10-20% issues required these additional poses from mediapipe. Decided to omit those and go with YOLO model as it'll include the relevant issues.


## Running Posture Mispostures (Side View)

1. **Overstriding**
   Foot lands far ahead of the hip at foot-strike.
   *Signal:* horizontal distance between ankle and hip x-position at foot-strike.

2. **Knee Flexion at Foot-Strike**
   How bent the knee is when the foot lands; a near-straight leg increases impact and often co-occurs with overstriding.
   *Signal:* hip-knee-ankle angle at foot-strike frames.

3. **Vertical Oscillation**
   Excessive up-and-down bounce per stride, wastes energy and increases impact.
   *Signal:* range of hip y-position over a stride cycle.

4. **Trunk Lean**
   Too much forward or backward lean; forward strains the lower back, backward often pairs with overstriding.
   *Signal:* angle of shoulder-hip line relative to vertical.

5. **Cadence (Steps per Minute)**
   Low cadence correlates with longer strides and higher impact forces.
   *Signal:* count of foot-strikes over time.

6. **Heel Whip / Kick-Back Height**
   During swing phase, whether the heel lifts close to the glutes (efficient) or barely lifts ("shuffling", inefficient).
   *Signal:* max vertical distance between ankle and hip during the swing phase.

7. **Arm signal**
   Arm swing forward/backward - swing amplitude & angle
