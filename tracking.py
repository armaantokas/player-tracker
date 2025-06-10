class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.id = track_id
        self.lost_frames = 0

class Tracker:
    def __init__(self, iou_threshold=0.3, max_lost=10):
        self.next_id = 0
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            matched = False
            for track in self.tracks:
                if self._iou(track.bbox, det[:4]) > self.iou_threshold:
                    track.bbox = det[:4]
                    track.lost_frames = 0
                    updated_tracks.append(track)
                    matched = True
                    break
            if not matched:
                new_track = Track(det[:4], self.next_id)
                self.next_id += 1
                updated_tracks.append(new_track)

        for track in self.tracks:
            if track not in updated_tracks:
                track.lost_frames += 1
                if track.lost_frames <= self.max_lost:
                    updated_tracks.append(track)

        self.tracks = updated_tracks
        return [{'bbox': t.bbox, 'id': t.id} for t in self.tracks]

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)
