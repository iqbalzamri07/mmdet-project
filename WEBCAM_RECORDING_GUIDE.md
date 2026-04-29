# Webcam Recording Guide for Training Data

## Overview
Capture training videos directly from your webcam for all 6 actions.

## Quick Start

```bash
cd /home/newadmin/mmdet-project
source venv/bin/activate
python capture_webcam_samples.py
```

## Features

### 1. Preview Mode
See your camera feed before recording to adjust lighting, position, and frame.

### 2. Automated Recording
Record all actions in sequence with specified number of samples.

### 3. Interactive Recording
Record specific actions with custom duration and samples.

### 4. Review Option
Watch back each recording immediately to verify quality.

## Recording Modes

### Mode 1: Record All Actions (Automated)
```
Select mode: 1
Number of samples per action: 5
Duration per sample (seconds): 5

→ Records 5 samples for each of the 6 actions (30 videos total)
→ Total time: ~15 minutes
```

### Mode 2: Record Specific Action
```
Select mode: 2
Which action to record? smoking
Number of samples: 10
Duration per sample: 6
Person ID: 1

→ Records 10 smoking samples from person 1
```

### Mode 3: Quick Recording
```
Select mode: 3
Which action to record? standing
Duration (seconds): 8

→ Records 1 standing sample (8 seconds)
```

## Action-Specific Tips

### 🚬 Smoking
- Hold cigarette/vape clearly visible
- Show hand-to-mouth motion 2-3 times
- Include both inhale and exhale
- Keep upper body in frame
- Natural smoking motion

### 🪑 Sitting
- Sit comfortably on chair
- Show clear sitting posture
- Include slight natural movements
- Keep upper body visible
- Arms can rest naturally

### 🧍 Standing
- Stand upright naturally
- Show stable standing posture
- Include slight swaying (natural)
- Arms at sides or natural position
- Keep whole body in frame if possible

### 🚶 Walking
- Walk naturally across frame
- Include 3-5 steps
- Show front, side, or back view
- Natural walking pace
- Keep person centered in frame

### 📞 Calling
- Hold phone to ear clearly
- Show phone visible if possible
- Include head tilt if natural
- Keep upper body visible
- Stay relatively still

### 📱 Playing Phone
- Hold phone/tablet visible
- Show phone usage (texting, browsing)
- Include head looking down at phone
- Keep upper body visible
- Natural interaction with phone

## Camera Setup Tips

### Positioning
- Place camera at eye level or slightly above
- Distance: 1.5-2 meters from person
- Angle: Front or slight side angle works best

### Lighting
- Good lighting is crucial for pose detection
- Avoid backlighting (light behind person)
- Even lighting on face and body

### Background
- Simple, uncluttered background
- Contrasting color from clothing
- Avoid busy patterns

### Framing
- Upper body (head to waist) for sitting/standing/calling/smoking/phone
- Full body for walking
- Leave some margin around edges

## Recording Guidelines

### Video Quality
- ✅ Clear, unambiguous action
- ✅ Single person in frame (mostly)
- ✅ Stable camera (no shaky footage)
- ✅ Good lighting
- ✅ Person centered in frame

### Action Clarity
- ✅ Action is clearly visible
- ✅ No mixed actions
- ✅ Sufficient duration (5-10 seconds)
- ✅ Shows temporal patterns (for smoking)

### What to Avoid
- ❌ Multiple people doing different actions
- ❌ Unclear or ambiguous actions
- ❌ Too short (< 2 seconds)
- ❌ Too long (> 15 seconds)
- ❌ Poor lighting
- ❌ Blurred footage
- ❌ Person mostly occluded

## Directory Structure

After recording, your data will look like:

```
data/custom_actions/train/
├── smoking/
│   ├── person01_smoking_20260428_165234_s001.mp4
│   ├── person01_smoking_20260428_165245_s002.mp4
│   └── ...
├── sitting/
├── standing/
├── walking/
├── calling/
└── playing_phone/
```

## Organizing Data for Training

After recording, you need to:

1. **Review all videos** and delete bad ones

2. **Split into train/val**:
   ```bash
   # Create validation directory
   mkdir -p data/custom_actions/val/{smoking,sitting,standing,walking,calling,playing_phone}
   
   # Move 20% of videos to val (manually or script)
   # Example: for each action, move 1 in 5 videos to val/
   ```

3. **Ensure balance**: Similar number of videos per action

## Recommended Dataset Size

| Target | Minimum | Recommended | Ideal |
|--------|---------|-------------|-------|
| Per action | 30-50 | 100-200 | 200+ |
| Total videos | 180-300 | 600-1200 | 1200+ |
| Training time | 2-3 hours | 4-8 hours | 8-12 hours |
| Expected accuracy | 70-80% | 85-90% | 90-95% |

## Troubleshooting

### Webcam not opening
- Check if camera is connected
- Try different camera ID (edit script: camera_id=1)
- Close other apps using webcam

### Poor video quality
- Improve lighting
- Check camera focus
- Clean camera lens
- Reduce motion blur (slower movements)

### Action not clear
- Ensure person is fully visible
- Check camera framing
- Make action more pronounced
- Review tips for specific action

### Recording interrupted
- Press 'q' to stop recording early
- Videos up to that point are saved
- Can re-record that sample

## Example Recording Session

```bash
# Start script
python capture_webcam_samples.py

# Choose Mode 1 (all actions)
Number of samples per action: 5
Duration per sample: 5

# Script will:
# 1. Show preview (optional)
# 2. For each action:
#    - Show tips
#    - Record 5 samples
#    - Allow review after each
# 3. Show summary
```

## Tips for Better Results

### For Each Action:
1. **Vary angles**: Record front, side, back views
2. **Vary lighting**: Day, indoor, mixed
3. **Vary people**: 3-5+ different people
4. **Vary background**: Different locations

### For Similar Actions (Standing vs Calling vs Smoking):
1. **Standing**: Clear, stable posture, no other action
2. **Calling**: Phone clearly at ear, consistent position
3. **Smoking**: Clear hand-to-mouth motion, 2-3 cycles

### Quality Check:
- Is the action clearly visible?
- Is the lighting good?
- Is the person properly framed?
- Is the duration appropriate (5-10 seconds)?

## Next Steps After Recording

1. **Review** all recorded videos
2. **Delete** any poor quality recordings
3. **Organize** into train (80%) and val (20%)
4. **Run** `python prepare_dataset.py`
5. **Train** model with `python -m mmaction.train configs/custom_action_recognition.py`

## Support

If you encounter issues:
1. Check camera permissions
2. Verify disk space
3. Ensure OpenCV is installed correctly
4. Try running with verbose output

Happy recording! 🎥



# workflow ==============================================================
1. Add new videos to dataset
cp new_video.mp4 data/custom_actions/train/walking/

2. Re-extract keypoints
python prepare_dataset.py

3. Retrain model (optional - only if adding significant new data)
python train_custom_model.py

4. Use the updated model for inference
python inference_custom_action.py