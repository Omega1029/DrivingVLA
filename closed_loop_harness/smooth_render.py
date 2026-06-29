import json, numpy as np, requests, io, os, cv2, time
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

SC="/tmp/claude-1012/-home-justin-williams1-OpenDriveVLA/daebd0c0-5f2e-4b21-9ce2-529f0f6f3334/scratchpad"
ROOT="/home/justin_williams1/neuroncap/neuro-ncap/output/rerecord"
OUT="/home/justin_williams1/OpenDriveVLA/videos"; os.makedirs(OUT,exist_ok=True)
cam2ego=np.load(f"{SC}/cam2ego.npy",allow_pickle=True).item()
CAMS=["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT"]
N=5            # subframes per 0.5s interval -> 5x density
FPS=15
URL="http://localhost:8000/render_image"

def interp_poses(ep):
    ts=sorted(ep.keys(),key=lambda x:int(x))
    P=[np.array(ep[t]) for t in ts]; T=[int(t) for t in ts]
    out=[]   # (pose4x4_ego2world, timestamp)
    for i in range(len(P)-1):
        Ra,Rb=Rotation.from_matrix(P[i][:3,:3]),Rotation.from_matrix(P[i+1][:3,:3])
        slerp=Slerp([0,1],Rotation.concatenate([Ra,Rb]))
        ta,tb=P[i][:3,3],P[i+1][:3,3]
        for k in range(N):
            f=k/N
            M=np.eye(4); M[:3,:3]=slerp([f])[0].as_matrix(); M[:3,3]=ta*(1-f)+tb*f
            out.append((M,int(T[i]*(1-f)+T[i+1]*f)))
    out.append((P[-1],T[-1]))
    return out

def render(ego2world,timestamp,cam):
    pose=ego2world@cam2ego[cam]
    r=requests.post(URL,json={"pose":pose.tolist(),"timestamp":int(timestamp),
                              "camera_name":cam,"image_format":"png"},timeout=120)
    img=np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

def panel(imgs):  # imgs dict cam->bgr ; front big top, L/R bottom
    fr=imgs["CAM_FRONT"]; h,w=fr.shape[:2]
    TW=1200; s=TW/w; fr=cv2.resize(fr,(TW,int(h*s)))
    bw=TW//2
    def rs(x): 
        hh,ww=x.shape[:2]; return cv2.resize(x,(bw,int(hh*(bw/ww))))
    l=rs(imgs["CAM_FRONT_LEFT"]); r=rs(imgs["CAM_FRONT_RIGHT"])
    bh=max(l.shape[0],r.shape[0])
    bottom=np.zeros((bh,TW,3),np.uint8); bottom[:l.shape[0],:bw]=l; bottom[:r.shape[0],bw:bw+r.shape[1]]=r
    return np.vstack([fr,bottom])

def banner(img,text,color,h=64):
    w=img.shape[1]; bar=np.zeros((h,w,3),np.uint8)
    cv2.rectangle(bar,(0,0),(w,h),(25,25,25),-1); cv2.rectangle(bar,(0,0),(12,h),color,-1)
    cv2.putText(bar,text,(28,int(h*0.64)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
    return np.vstack([bar,img])

CONFIGS=[("fp16","FP16 - drives forward (79% coherent)",(40,180,40)),
         ("naive_w4","naive per-channel W4 - planner FROZEN (0%) -> collision",(40,40,220)),
         ("group_w4_g128","group-wise W4 g128 - drives forward (71%) [THE FIX]",(40,180,40))]

allframes={}
for cfg,label,color in CONFIGS:
    ep=json.load(open(f"{ROOT}/{cfg}/ego_poses.json"))
    poses=interp_poses(ep)
    print(f"[{cfg}] {len(poses)} interpolated poses",flush=True)
    frames=[]
    t0=time.time()
    for j,(M,tsmp) in enumerate(poses):
        imgs={c:render(M,tsmp,c) for c in CAMS}
        frames.append(panel(imgs))
    print(f"[{cfg}] rendered {len(frames)} frames in {time.time()-t0:.0f}s",flush=True)
    allframes[cfg]=(frames,label,color)
    framed=[banner(f,label,color) for f in frames]
    h,w=framed[0].shape[:2]
    p=f"{OUT}/0103_seed0_{cfg}_SMOOTH.mp4"
    vw=cv2.VideoWriter(p,cv2.VideoWriter_fourcc(*"mp4v"),FPS,(w,h))
    for f in framed: vw.write(f)
    vw.release(); print(f"wrote {os.path.basename(p)} {os.path.getsize(p)//1024}KB {w}x{h}",flush=True)

# side-by-side (CAM_FRONT only row for compactness)
TH=420
def col(cfg):
    out=[]
    for f in allframes[cfg][0]:
        s=TH/f.shape[0]; out.append(cv2.resize(f,(int(f.shape[1]*s),TH)))
    return out
cols={c:col(c) for c,_,_ in CONFIGS}
n=min(len(v) for v in cols.values())
short={"fp16":"FP16","naive_w4":"naive W4 (FROZEN)","group_w4_g128":"group W4 (FIX)"}
rows=[]
for i in range(n):
    panels=[banner(cols[c][i],short[c],col_) for c,_,col_ in CONFIGS]
    rows.append(np.hstack(panels))
h,w=rows[0].shape[:2]
p=f"{OUT}/0103_seed0_sidebyside_SMOOTH.mp4"
vw=cv2.VideoWriter(p,cv2.VideoWriter_fourcc(*"mp4v"),FPS,(w,h))
for f in rows: vw.write(f)
vw.release(); print(f"wrote {os.path.basename(p)} {os.path.getsize(p)//1024}KB {w}x{h}",flush=True)
print("ALL DONE",flush=True)
