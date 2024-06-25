from utils import (read_video,
                   save_video)
from trackers import PlayerTracker, BallTracker

def main():
    #reading video
    print("Reading the video...")
    input_video_path = "input_video/input_video.mp4"
    video_frames = read_video(input_video_path)

    #detecting players
    print("Detecting players...")
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="trackers_stubs/player_detections.pkl")

    # Draw outputs
    print("Drawing Anotations...")
    ## draw player bbox
    output_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    # Save video
    print("Saving Video...")
    save_video(output_frames, "output_video/output_video.avi")

    print("Done!")

if __name__ == "__main__":
    main()