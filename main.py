import cv2
from detection import Detector
from tracking import Tracker

video_path = '15sec_input_720p.mp4' # Change your input path here
cap = cv2.VideoCapture(video_path)

detector = Detector('best.pt') #Insert Model Here
tracker = Tracker()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('tracked_output.mp4', fourcc, fps,   #Add Output path here
                      (int(cap.get(3)), int(cap.get(4))))  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections)

    for track in tracks:
        x1, y1, x2, y2 = track['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID: {track['id']}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
