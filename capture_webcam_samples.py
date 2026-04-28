"""
Webcam Video Capture Script for Training Data Collection

This script helps you record sample videos for your 6 actions:
- smoking
- sitting
- standing
- walking
- calling
- playing_phone

Usage:
    python capture_webcam_samples.py
"""

import cv2
import os
import time
from datetime import datetime
from pathlib import Path

# Configuration
OUTPUT_ROOT = "data/custom_actions"
DEFAULT_DURATION = 5  # seconds per video
DEFAULT_FPS = 30
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

# Actions to record
ACTIONS = ["smoking", "sitting", "standing", "walking", "calling", "playing_phone"]


class WebcamRecorder:
    def __init__(self, camera_id=0):
        """Initialize webcam"""
        print("Initializing webcam...")
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            exit(1)

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)

        # Get actual camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"✓ Webcam initialized: {self.width}x{self.height} @ {self.fps}fps")
        print(f"✓ Camera ID: {camera_id}")

    def show_preview(self):
        """Show camera preview"""
        print("\nShowing preview. Press 'q' to stop preview.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from webcam")
                break

            # Add overlay
            cv2.rectangle(frame, (10, 10), (400, 70), (0, 0, 0), -1)
            cv2.putText(
                frame,
                "PREVIEW MODE",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Press 'q' to stop",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Webcam Preview", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def record_video(self, action, duration=DEFAULT_DURATION, person_id=1, sample_id=1):
        """Record a video for a specific action"""

        # Create output path
        output_dir = os.path.join(OUTPUT_ROOT, "train", action)
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"person{person_id:02d}_{action}_{timestamp}_s{sample_id:03d}.mp4"
        output_path = os.path.join(output_dir, filename)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        print(f"\n{'=' * 60}")
        print(f"Recording: {action}")
        print(f"Duration: {duration} seconds")
        print(f"Output: {output_path}")
        print(f"{'=' * 60}")
        print(f"\n⚠️  Instructions:")
        print(f"  - Perform the '{action}' action clearly")
        print(f"  - Stay in frame")
        print(f"  - Make sure action is visible")
        print(f"\n⏳  Recording starts in 3 seconds...")

        # Countdown
        for i in range(3, 0, -1):
            ret, frame = self.cap.read()
            if ret:
                # Overlay countdown
                cv2.rectangle(frame, (0, 0), (self.width, self.height), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    str(i),
                    (self.width // 2 - 50, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,
                    (0, 255, 0),
                    5,
                )
                cv2.putText(
                    frame,
                    f"Action: {action}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Recording", frame)
                cv2.waitKey(1000)

        # Recording
        start_time = time.time()
        frame_count = 0

        print("🔴 Recording...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Add recording indicator
            elapsed = time.time() - start_time
            remaining = duration - elapsed

            # Red recording indicator
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)

            # Info overlay
            cv2.rectangle(frame, (50, 10), (400, 80), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"REC {action}",
                (60, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Time: {elapsed:.1f}s / {duration}s",
                (60, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            out.write(frame)
            cv2.imshow("Recording", frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n⚠️  Recording stopped by user")
                break

            if elapsed >= duration:
                break

        out.release()
        cv2.destroyAllWindows()

        print(f"✓ Recording complete!")
        print(f"  Frames captured: {frame_count}")
        print(f"  Actual duration: {elapsed:.2f} seconds")
        print(f"  Saved to: {output_path}")

        return output_path

    def record_multiple_samples(
        self, action, num_samples=5, person_id=1, duration=DEFAULT_DURATION
    ):
        """Record multiple samples for one action"""
        print(f"\n{'=' * 60}")
        print(f"Recording {num_samples} samples for: {action}")
        print(f"Person ID: {person_id}")
        print(f"Duration: {duration} seconds each")
        print(f"{'=' * 60}")

        for i in range(1, num_samples + 1):
            print(f"\nSample {i}/{num_samples}")

            # Option to skip this sample
            response = input("  Press Enter to start, or 's' to skip: ").strip().lower()
            if response == "s":
                print("  Skipped")
                continue

            # Record
            self.record_video(action, duration, person_id, i)

            # Option to review
            review = input("  Review recording? (y/n): ").strip().lower()
            if review == "y":
                self.review_last_recording(action, person_id, i)

    def review_last_recording(self, action, person_id, sample_id):
        """Review the last recorded video"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # This is approximate - find the most recent file
        action_dir = os.path.join(OUTPUT_ROOT, "train", action)
        videos = [f for f in os.listdir(action_dir) if f.endswith(".mp4")]
        if videos:
            last_video = os.path.join(action_dir, videos[-1])
            print(f"\n  Playing: {last_video}")
            print("  Press 'q' to stop playback")

            cap = cv2.VideoCapture(last_video)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow(f"Review: {action}", frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()

    def record_all_actions(self, samples_per_action=5, duration=DEFAULT_DURATION):
        """Record samples for all actions"""
        print(f"\n{'=' * 60}")
        print(f"Recording Samples for All Actions")
        print(f"{'=' * 60}")
        print(f"\nYou will record {samples_per_action} samples for each action.")
        print(f"Each sample: {duration} seconds")
        print(
            f"Total recording time: {len(ACTIONS) * samples_per_action * duration / 60:.1f} minutes"
        )
        print(f"\nPress Enter to continue...")
        input()

        person_id = 1  # Can be changed

        for action in ACTIONS:
            print(f"\n{'#' * 60}")
            print(f"# Action: {action.upper()}")
            print(f"{'#' * 60}")

            # Show action-specific tips
            self.show_action_tips(action)

            # Show preview
            preview = input("\n  Show camera preview? (y/n): ").strip().lower()
            if preview == "y":
                self.show_preview()

            # Record samples
            self.record_multiple_samples(
                action, samples_per_action, person_id, duration
            )

    def show_action_tips(self, action):
        """Show tips for recording a specific action"""
        tips = {
            "smoking": [
                "• Hold cigarette/vape clearly visible",
                "• Show hand-to-mouth motion 2-3 times",
                "• Include both inhale and exhale",
                "• Keep upper body in frame",
                "• Natural smoking motion",
            ],
            "sitting": [
                "• Sit comfortably on chair",
                "• Show clear sitting posture",
                "• Include slight natural movements",
                "• Keep upper body visible",
                "• Arms can rest naturally",
            ],
            "standing": [
                "• Stand upright naturally",
                "• Show stable standing posture",
                "• Include slight swaying (natural)",
                "• Arms at sides or natural position",
                "• Keep whole body in frame if possible",
            ],
            "walking": [
                "• Walk naturally across frame",
                "• Include 3-5 steps",
                "• Show front, side, or back view",
                "• Natural walking pace",
                "• Keep person centered in frame",
            ],
            "calling": [
                "• Hold phone to ear clearly",
                "• Show phone visible if possible",
                "• Include head tilt if natural",
                "• Keep upper body visible",
                "• Stay relatively still",
            ],
            "playing_phone": [
                "• Hold phone/tablet visible",
                "• Show phone usage (texting, browsing)",
                "• Include head looking down at phone",
                "• Keep upper body visible",
                "• Natural interaction with phone",
            ],
        }

        if action in tips:
            print(f"\n  Tips for {action}:")
            for tip in tips[action]:
                print(f"  {tip}")

    def close(self):
        """Release camera"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Webcam released")


def main():
    """Main function"""
    print("=" * 60)
    print("Webcam Video Capture for Training Data")
    print("=" * 60)
    print("\nThis script helps you record sample videos for action recognition.")
    print(f"Actions: {', '.join(ACTIONS)}")
    print("=" * 60)

    # Initialize camera
    try:
        recorder = WebcamRecorder(camera_id=0)
    except Exception as e:
        print(f"\n❌ Error initializing webcam: {e}")
        print("\nPossible solutions:")
        print("1. Check if webcam is connected")
        print("2. Try different camera ID (e.g., camera_id=1)")
        print("3. Close other applications using the webcam")
        return

    # Show preview
    print("\n" + "=" * 60)
    print("Step 1: Camera Preview")
    print("=" * 60)
    show_preview = input("\nShow camera preview? (y/n): ").strip().lower()
    if show_preview == "y":
        recorder.show_preview()

    # Recording mode
    print("\n" + "=" * 60)
    print("Step 2: Choose Recording Mode")
    print("=" * 60)
    print("\n1. Record all actions (automated)")
    print("2. Record specific action")
    print("3. Quick recording (single sample)")

    mode = input("\nSelect mode (1/2/3): ").strip()

    if mode == "1":
        # Record all actions
        samples = input("\nNumber of samples per action (default 5): ").strip()
        samples = int(samples) if samples else 5
        duration = input("Duration per sample in seconds (default 5): ").strip()
        duration = int(duration) if duration else 5

        recorder.record_all_actions(samples_per_action=samples, duration=duration)

    elif mode == "2":
        # Record specific action
        print(f"\nAvailable actions: {', '.join(ACTIONS)}")
        action = input("\nWhich action to record? ").strip().lower()

        if action not in ACTIONS:
            print(f"\n❌ Invalid action. Please choose from: {', '.join(ACTIONS)}")
            recorder.close()
            return

        samples = input("Number of samples (default 5): ").strip()
        samples = int(samples) if samples else 5
        duration = input("Duration per sample in seconds (default 5): ").strip()
        duration = int(duration) if duration else 5
        person_id = input("Person ID (default 1): ").strip()
        person_id = int(person_id) if person_id else 1

        recorder.record_multiple_samples(action, samples, person_id, duration)

    elif mode == "3":
        # Quick recording
        print(f"\nAvailable actions: {', '.join(ACTIONS)}")
        action = input("Which action to record? ").strip().lower()

        if action not in ACTIONS:
            print(f"\n❌ Invalid action. Please choose from: {', '.join(ACTIONS)}")
            recorder.close()
            return

        duration = input("Duration in seconds (default 5): ").strip()
        duration = int(duration) if duration else 5

        recorder.record_video(action, duration)

    else:
        print("\n❌ Invalid mode")
        recorder.close()
        return

    # Summary
    print("\n" + "=" * 60)
    print("Recording Complete!")
    print("=" * 60)

    # Count recorded videos
    total_videos = 0
    for action in ACTIONS:
        action_dir = os.path.join(OUTPUT_ROOT, "train", action)
        if os.path.exists(action_dir):
            videos = [f for f in os.listdir(action_dir) if f.endswith(".mp4")]
            count = len(videos)
            total_videos += count
            print(f"  {action}: {count} videos")

    print(f"\n  Total: {total_videos} videos")
    print(f"\n  Location: {OUTPUT_ROOT}/train/")
    print("=" * 60)

    # Next steps
    print("\nNext steps:")
    print("1. Review recorded videos")
    print("2. Remove any bad recordings")
    print("3. Organize 20% into val/ folder")
    print("4. Run: python prepare_dataset.py")
    print("5. Train model")
    print("=" * 60)

    # Cleanup
    recorder.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Recording interrupted by user")
        print("\nAny recorded videos are saved in:", OUTPUT_ROOT)
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
