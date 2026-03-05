# Background Music Assets

The AI editor applies tone-matched background music to every export.
You must provide the following MP3 files in this directory:

| File | Used for tone |
|------|--------------|
| `cinematic.mp3` | cinematic |
| `energetic.mp3` | energetic |
| `epic.mp3`      | epic |
| `sentimental.mp3` | sentimental |
| `calm.mp3`      | calm |

## Requirements
- Format: MP3 (44.1 kHz, stereo, 128 kbps minimum)
- Length: at least 3 minutes recommended (shorter tracks are looped automatically)
- The files are mixed at 20% volume under the original clip audio by default

## If files are missing
The backend silently skips music — video generation still completes, but:
- Background music is not added
- If `mix_original_audio=False`, the output will have silent audio gaps

## Getting royalty-free music
- https://www.pixabay.com/music/ (free, no attribution required)
- https://freemusicarchive.org/
- https://artlist.io / https://epidemicsound.com (paid, licensed for commercial use)
