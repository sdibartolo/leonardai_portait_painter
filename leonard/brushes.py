import cv2, numpy as np, math
from dataclasses import dataclass

@dataclass
class Stroke:
    x:float; y:float; length:float; angle:float; width:int
    curvature:float; alpha:float; color:tuple; brush:str="filbert"

def mix_palette(weights,palette):
    w=np.clip(np.asarray(weights,np.float32),0,None)
    if w.sum()==0:w[4]=1
    w/=w.sum()
    c=(w[:,None]*np.asarray(palette,np.float32)).sum(0)
    diversity=1-float(w.max())
    c=c*(1-.12*diversity)+np.array([74,67,58],np.float32)*(.12*diversity)
    return tuple(np.clip(c,0,255).astype(np.uint8))

def _curve_points(s,rng,h,w,n):
    pts=[]; dx,dy=math.cos(s.angle),math.sin(s.angle); nx,ny=-dy,dx
    phase=rng.uniform(0,2*math.pi)
    for u in np.linspace(0,1,n):
        bend=math.sin(math.pi*u)*s.curvature*s.length
        wobble=math.sin(u*math.pi*2.3+phase)*s.width*.055
        jitter=rng.normal(0,max(.08,s.width*.018))
        px=s.x+(u-.5)*s.length*dx+(bend+wobble+jitter)*nx
        py=s.y+(u-.5)*s.length*dy+(bend+wobble+jitter)*ny
        pts.append((float(np.clip(px,0,w-1)),float(np.clip(py,0,h-1))))
    return pts

def render_stroke(canvas,s,rng):
    h,w=canvas.shape[:2]
    pts=_curve_points(s,rng,h,w,max(12,min(54,int(s.length*1.25))))
    mask=np.zeros((h,w),np.float32)
    for i,(px,py) in enumerate(pts):
        u=i/max(1,len(pts)-1)
        pressure=.30+.70*(math.sin(math.pi*u)**.42)
        depletion=1-.22*u
        radius=max(.65,s.width*.50*pressure)
        a=pressure*depletion
        if s.brush=="flat":
            dx,dy=math.cos(s.angle),math.sin(s.angle); nx,ny=-dy,dx
            strands=max(3,min(11,int(s.width*.55)))
            for b in range(strands):
                off=((b/max(1,strands-1))-.5)*s.width+rng.normal(0,.12)
                qx=int(np.clip(px+nx*off,0,w-1)); qy=int(np.clip(py+ny*off,0,h-1))
                cv2.circle(mask,(qx,qy),max(1,int(radius/max(2,strands*.42))),
                           float(a*rng.uniform(.72,1.0)),-1,cv2.LINE_AA)
        elif s.brush=="filbert":
            cv2.circle(mask,(int(px),int(py)),max(1,int(radius)),float(a),-1,cv2.LINE_AA)
        else:
            cv2.circle(mask,(int(px),int(py)),max(1,int(radius*.72)),float(a),-1,cv2.LINE_AA)
    if s.brush=="filbert":
        mask=cv2.GaussianBlur(mask,(0,0),max(.28,s.width*.035))
    yy,xx=np.where(mask>0)
    if len(xx):
        holes=rng.random(len(xx)) < (.015 + .035*(s.brush=="flat"))
        if holes.any():
            mask[yy[holes],xx[holes]]*=rng.uniform(.05,.45,size=holes.sum())
    alpha=np.clip(mask*s.alpha,0,1)[...,None]
    # Pigment variation only where paint exists; avoids perfectly digital fill.
    variation=rng.normal(0,1.3,(h,w,1)).astype(np.float32)
    paint=np.clip(np.asarray(s.color,np.float32)[None,None,:]+variation,0,255)
    out=canvas.astype(np.float32)*(1-alpha)+paint*alpha
    return np.clip(out,0,255).astype(np.uint8)


# ============================================================
# Fast Teacher V2 local/crop rendering
# ============================================================

def stroke_bounds(stroke, canvas_shape, margin=8):
    """Conservative bounds containing the complete rendered stroke."""
    h, w = canvas_shape[:2]
    curve_extent = abs(stroke.curvature * stroke.length)
    radius = (
        stroke.length * 0.58
        + stroke.width * 1.8
        + curve_extent
        + margin
    )
    x0 = max(0, int(math.floor(stroke.x - radius)))
    y0 = max(0, int(math.floor(stroke.y - radius)))
    x1 = min(w, int(math.ceil(stroke.x + radius + 1)))
    y1 = min(h, int(math.ceil(stroke.y + radius + 1)))
    return x0, y0, x1, y1


def render_stroke_local(canvas, stroke, rng):
    """
    Render only the small crop touched by a candidate.
    Returns (painted_crop, bounds). The caller can paste the winning crop
    directly into the full canvas, avoiding a second full-canvas render.
    """
    x0, y0, x1, y1 = stroke_bounds(stroke, canvas.shape)
    if x1 <= x0 or y1 <= y0:
        return None, (x0, y0, x1, y1)

    crop = canvas[y0:y1, x0:x1].copy()
    local = Stroke(
        x=stroke.x-x0, y=stroke.y-y0,
        length=stroke.length, angle=stroke.angle, width=stroke.width,
        curvature=stroke.curvature, alpha=stroke.alpha,
        color=stroke.color, brush=stroke.brush
    )
    return render_stroke(crop, local, rng), (x0, y0, x1, y1)
