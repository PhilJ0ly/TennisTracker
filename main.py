from utils import (read_video,
                   save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

def main():
    #reading video
    print("Reading the video...")
    input_video_path = "input_video/input_video.mp4"
    video_frames = read_video(input_video_path)

    #detecting players
    print("Detecting players...")
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="trackers_stubs/player_detections.pkl")

    #detecting ball
    print("Detecting ball...")
    ball_tracker = BallTracker(model_path="models/yolov5_best.pt")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="trackers_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball(ball_detections)

    #detecting keypoints
    print("Detecting court lines...")
    court_model_path = "models/key_pts.pth"
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    court_kps = court_line_detector.predict(video_frames[0])

    # Draw outputs
    print("Drawing Anotations...")
    ## draw player and ball bbox
    output_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections)

    ## Draw court lines
    output_frames = court_line_detector.draw_kps_video(output_frames, court_kps)

    # Save video
    print("Saving Video...")
    save_video(output_frames, "output_video/output_video.avi")

    print("Done!")

if __name__ == "__main__":
    main()