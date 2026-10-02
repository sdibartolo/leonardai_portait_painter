from dataclasses import dataclass
@dataclass
class PaintConfig:
    work_size:int=384
    sketch_strokes:int=180
    paint_strokes:int=2600
    candidates:int=28
    video_every:int=6
    fps:int=30
    seed:int=7
    palette=((220,35,35),(245,205,35),(35,75,205),(20,20,20),(240,238,230))
    brush_sizes=(4,12,28)

    # Phase 1.5 structured portrait teacher
    feature_cycles:int=3
    eye_marks:int=42
    nose_marks:int=34
    mouth_marks:int=38
    jaw_marks:int=30
    feature_candidates:int=52
    feature_crop_margin:float=0.35
