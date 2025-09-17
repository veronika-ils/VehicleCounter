import cv2
import time
import math

VIDEO = r'C:\Users\ilios\Downloads\56310-479197605_small.mp4'

def center(x, y, w, h):
    return int(x + w/2), int(y + h/2)

def region_of(cy, top, bot):

    if cy < top: return -1
    if cy > bot: return 1
    return 0

def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("Cannot open video")
        return

    VIEW_W, VIEW_H = 640, 480
    CROP_TOP = 200
    line_y = 160
    gate_half = 28
    gate_top = line_y - gate_half
    gate_bot = line_y + gate_half


    backsub = cv2.createBackgroundSubtractorMOG2(history=350, varThreshold=26, detectShadows=False)


    min_area, max_area = 500, 30000
    min_w = 12
    match_radius = 75
    max_age_frames = 12
    cooldown_sec = 0.4

    cars_down = 0
    cars_up = 0


    objects = {}
    next_id = 1

    flashes = []

    def match_object(cx, cy):
        best_id, best_d = None, 1e9
        for oid, o in objects.items():
            d = math.hypot(cx - o["cx"], cy - o["cy"])
            if d < best_d:
                best_d, best_id = d, oid
        return best_id if best_d <= match_radius else None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        now = time.time()

        frame_r = cv2.resize(frame, (VIEW_W, VIEW_H))
        crop = frame_r[CROP_TOP:VIEW_H, 0:VIEW_W]


        fg = backsub.apply(crop)

        _, fg = cv2.threshold(fg, 160, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
        fg = cv2.dilate(fg, None, iterations=1)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        seen_ids = set()
        candidates = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < min_w:
                continue

            cx, cy = center(x, y, w, h)
            candidates += 1

            oid = match_object(cx, cy)
            if oid is None:

                oid = next_id
                next_id += 1
                r = region_of(cy, gate_top, gate_bot)
                entered_from = None
                if r == 0:

                    entered_from = None
                elif r == -1:
                    entered_from = 'above'
                else:
                    entered_from = 'below'

                objects[oid] = {
                    "cx": cx, "cy": cy,
                    "last_seen": frame_idx,
                    "state": r,
                    "entered_from": entered_from,
                    "last_count_time": 0.0
                }
            else:
                o = objects[oid]
                prev_state = o["state"]
                curr_state = region_of(cy, gate_top, gate_bot)


                if prev_state == -1 and curr_state == 0:
                    o["entered_from"] = 'above'
                elif prev_state == 1 and curr_state == 0:
                    o["entered_from"] = 'below'


                counted = False
                if o["entered_from"] == 'above' and prev_state == 0 and curr_state == 1:
                    if now - o["last_count_time"] > cooldown_sec:
                        cars_down += 1
                        counted = True
                elif o["entered_from"] == 'below' and prev_state == 0 and curr_state == -1:
                    if now - o["last_count_time"] > cooldown_sec:
                        cars_up += 1
                        counted = True

                if counted:
                    o["last_count_time"] = now
                    flashes.append({"rect": (x, y, w, h), "ttl": 6})

                    o["entered_from"] = None


                o["cx"], o["cy"] = cx, cy
                o["state"] = curr_state
                o["last_seen"] = frame_idx

            seen_ids.add(oid)


        for oid in [k for k,v in objects.items() if frame_idx - v["last_seen"] > max_age_frames]:
            del objects[oid]


        cv2.line(crop, (0, gate_top), (crop.shape[1], gate_top), (0, 0, 255), 2)
        cv2.line(crop, (0, gate_bot), (crop.shape[1], gate_bot), (0, 0, 255), 2)
        cv2.putText(frame_r, f"DOWN:{cars_down}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(frame_r, f"UP:{cars_up}",   (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(frame_r, f"Cands:{candidates}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


        kept = []
        for f in flashes:
            x,y,w,h = f["rect"]
            cv2.rectangle(crop, (x, y), (x+w, y+h), (0,255,255), 2)
            f["ttl"] -= 1
            if f["ttl"] > 0:
                kept.append(f)
        flashes = kept

        frame_r[CROP_TOP:VIEW_H, 0:VIEW_W] = crop
        cv2.imshow("Vehicle Counter (robust gate FSM)", frame_r)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
