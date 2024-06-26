from  ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball(self, ball_detections):
        ball_detections = [x.get(1,[]) for x in ball_detections]
        df_ball_detections = pd.DataFrame(ball_detections, columns=["x1", "y1", "x2", "y2"])

        # interpolate
        df_ball_detections = df_ball_detections.interpolate()
        df_ball_detections = df_ball_detections.bfill()

        ball_detections = [{1:x} for x in df_ball_detections.to_numpy().tolist()]

        return ball_detections

    def get_ball_shot_frames(self, ball_detections):
        ball_detections = [x.get(1,[]) for x in ball_detections]
        df_ball_detections = pd.DataFrame(ball_detections, columns=["x1", "y1", "x2", "y2"])

        df_ball_detections['mid_y'] = (df_ball_detections['y1']+df_ball_detections['y2'])/2
        df_ball_detections['mid_y_rolling_mean'] = df_ball_detections['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_detections['delta_y'] = df_ball_detections['mid_y_rolling_mean'].diff()

        df_ball_detections['hits'] = 0

        min_frames_for_hit = 25
        for i in range(1, len(df_ball_detections)-int(min_frames_for_hit*1.2)):
            neg_change = df_ball_detections['delta_y'].iloc[i] > 0 and df_ball_detections['delta_y'].iloc[i+1] <0
            pos_change = df_ball_detections['delta_y'].iloc[i] < 0 and df_ball_detections['delta_y'].iloc[i+1] >0

            if neg_change or pos_change:
                change_count = 0
                for change_frame in range(i+1, i+int(min_frames_for_hit*1.2)+1):
                    neg_change_nxt_frame = df_ball_detections['delta_y'].iloc[i] > 0 and df_ball_detections['delta_y'].iloc[change_frame] <0
                    pos_change_nxt_frame = df_ball_detections['delta_y'].iloc[i] < 0 and df_ball_detections['delta_y'].iloc[change_frame] >0

                    if neg_change and neg_change_nxt_frame:
                        change_count += 1
                    elif pos_change and pos_change_nxt_frame:
                        change_count +=1
                
                if change_count>min_frames_for_hit-1:
                    df_ball_detections['hits'].iloc[i]= 1
        
        frames_hits = df_ball_detections[df_ball_detections['hits']==1].index.tolist()
        return frames_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result    
        return ball_dict
    
    def draw_bboxes(self, video_frames, ball_detections):
        output_frames = []

        for frame, ball_dict in zip(video_frames, ball_detections):
            # draw bboxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_frames.append(frame)

        return output_frames
        