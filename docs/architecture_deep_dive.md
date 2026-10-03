# SENTINEL: Architecture & Methodology Deep Dive

> **Note for Team:** Ye document specially design kiya gaya hai taake aapko apne project ke har single component ki deep technical working pata ho. Agar evaluator kisi bhi tool ya model ke andar ki working (Input -> Process -> Output) puche, toh aap yahan se tayyari kar sakte hain.

---

## 1. Video Ingestion (OpenCV)
**Kya hai?** OpenCV (Open Source Computer Vision) ek library hai jo images aur videos ko process karne ke liye use hoti hai. SENTINEL mein ye hamari "aankh" (eye) ka kaam karta hai.

*   **Input:** CCTV camera ki RTSP stream ya normal webcam feed (video format mein).
*   **Processing:** Video essentially images (frames) ka ek sequence hoti hai. OpenCV is continuous video file/stream ko pakar kar usay alag alag pictures (frames) mein todta (split karta) hai. Phir ye in frames ko BGR format se RGB format mein convert karta hai kyunke AI models RGB samajhte hain.
*   **Output:** Ek single 3D array of pixels (Height × Width × 3 Colors) jo aage AI models ko feed ki jati hai.

---

## 2. Spatial Stream: Weapon Detection (YOLOv8s)
**Kya hai?** YOLO (You Only Look Once) ek state-of-the-art Object Detection model hai. Hum iska 'Small' (s) variant use kar rahe hain jo speed aur accuracy ka best balance deta hai.

*   **Input:** OpenCV se aane wala single RGB image frame.
*   **Processing:** 
    *   YOLOv8 image ko chote grids mein divide karta hai.
    *   Iska **Feature Pyramid Network (FPN)** image ke andar se features (edges, shapes, colors) nikalta hai. Small variant choti objects (jaise door khare insaan ke hath mein gun) ke features nikalne mein bohot acha hai.
    *   Model calculation karta hai ke kya in features ki shape hamare trained dataset (Guns aur Knives) se match karti hai?
*   **Output:** 
    1.  **Bounding Box Coordinates:** (x, y, width, height) — yani weapon screen par kahan hai.
    2.  **Class ID:** Ye Gun hai ya Knife?
    3.  **Confidence Score:** Model kitne percent sure hai? (e.g., 0.85 yani 85% sure).

---

## 3. Pose Extraction (YOLOv8m-Pose)
**Kya hai?** Ye bhi YOLO ka hi ek variant hai, lekin ye objects dhoondne ke bajaye insaan ka skeleton (haddiyon ka dhancha) dhoondta hai. Hum 'Medium' (m) variant use kar rahe hain taake agar frame mein insaan adha chupa (occluded) bhi ho toh joints detect ho jayein.

*   **Input:** OpenCV se aane wala wohi same RGB image frame (jo YOLOv8s mein gaya tha, dono parallel chal rahe hain).
*   **Processing:** 
    *   Model pehle frame mein Insaan (Person) ko dhoondta hai.
    *   Phir us insaan ki body par 17 specific points (Nose, Left Eye, Right Shoulder, Left Elbow, Knees, Ankles etc.) ko locate karta hai.
*   **Output:** Har insaan ke liye ek 51-dimensional list (vector) nikalti hai. 17 joints hain, aur har joint ka (X coordinate, Y coordinate, aur Confidence score) hota hai. *[17 × 3 = 51 values]*.

---

## 4. Temporal Stream: Action Recognition (LSTM)
**Kya hai?** LSTM (Long Short-Term Memory) ek Recurrent Neural Network (RNN) hai. Normal AI models pichli baat bhool jate hain, lekin LSTM ko yaad rehta hai ke pichle frames mein kya hua tha. Ye sequential data ke liye perfect hai.

*   **Input:** YOLO-Pose se aane wale 51-dimensional vectors. Lekin 1 frame nahi, **30 frames ki ek sequence** (1 second ka data). Toh input shape banta hai: `(30, 51)`.
*   **Processing:** 
    *   Pehle ye 51 coordinates normalize kiye jate hain (taake insaan camera ke qareeb ho ya door, data scale barabar rahay).
    *   LSTM is 30 frames ke data ko step-by-step parhta hai. Ye dekhta hai ke Frame 1 mein haath kahan tha, Frame 10 mein kahan gaya, aur Frame 30 mein kahan punchua.
    *   Is movement pattern ko wo apne andaray 'Hidden State' mein store karta hai aur check karta hai ke ye movement pattern kiss class (Fighting, Robbery, Shooting, ya Normal) se sab se zyada match karta hai.
*   **Output:** Ek Softmax layer ke through 4 probability percentages aati hain. E.g., `[Fighting: 80%, Normal: 10%, Robbery: 5%, Shooting: 5%]`. Jo sab se high hai, wo final prediction hoti hai.

---

## 5. Temporal Consistency Buffer (False Alarm Filter)
**Kya hai?** Ye koi AI model nahi hai, ye hamari apni banayi hui ek purely logical programming script hai.

*   **Input:** YOLOv8 (Weapon) aur LSTM (Behavior) ka Output. 
*   **Processing:** 
    *   Jab dono mein se koi ek model "Anomaly" (Threat) flag karta hai, toh buffer alert nahi bajata.
    *   Wo us threat ke bounding box ko track karna shuru karta hai using **IoU (Intersection over Union)**. IoU ka matlab hai ke agle frame mein bhi wo object pichle frame wali jagah ke aas paas hi hai.
    *   Buffer condition check karta hai: Kya pichle 10 frames mein se at least 6 frames mein ye threat consistently detect hua hai?
*   **Output:** 
    *   Agar **YES** -> Threat is REAL. Alert trigger kardo (Database mein log karo, frontend pe bhej do).
    *   Agar **NO** -> Threat is FAKE/GLITCH. Ignore kardo.

---

## 6. Backend (FastAPI)
**Kya hai?** Ye hamare poore system ka "Brain" ya "Manager" hai.

*   **Input:** Camera, YOLO, Pose, LSTM, aur Buffer sab FastAPI ke andar chal rahe hote hain.
*   **Processing:** 
    *   Kyunke FastAPI **Asynchronous** (async) hai, ye ek waqt mein multiple kaam handle kar sakta hai. Jab ek frame YOLO se process ho raha hota hai, FastAPI frame processing ko rokta nahi, balke agla frame camera se nikal kar line mein laga deta hai. (Agar Django hota toh wo hang ho jata).
    *   Jab buffer threat confirm karta hai, FastAPI us incident ki detail (Time, Snapshot picture, Threat type) ko SQLite Database mein save kar deta hai.
*   **Output:** Ye processed video frame aur detection ka data (Boxes, Skeleton, Text) tayyar karta hai taake Frontend ko bhej sake.

---

## 7. Communication (WebSocket)
**Kya hai?** Ye backend aur frontend ke darmiyan ek "Fast Lane" (highway) hai. 

*   **Input:** FastAPI se aane wala processed data.
*   **Processing:** Normal HTTP (jaise hum website kholte hain) mein frontend ko bar bar backend se puchna parta hai "Koi naya frame aaya?". Is mein delay (latency) aata hai. WebSocket ek dafa connection bana leta hai, aur phir backend continuously bina ruke data frontend par push karta rehta hai (live streaming ki tarah).
*   **Output:** Live frames aur JSON payload (jisme bounding boxes ki info hoti hai) frontend tak milliseconds mein pohnch jati hai.

---

## 8. Frontend Dashboard (React JS & HTML5 Canvas)
**Kya hai?** Ye user interface hai jahan security guard ya operator video dekhta hai.

*   **Input:** WebSockets ke through aane wali Live Video aur JSON Data (Coordinates).
*   **Processing:** 
    *   React website ka structure (buttons, history table, alerts) banata hai.
    *   **HTML5 Canvas:** Normal websites par agar aap video ke upar 10 bounding boxes (dibbay) draw karein toh browser hang ho jata hai (DOM overload). Canvas ek drawing board ki tarah hai jo direct aapke GPU ko use karke dibbay draw karta hai.
    *   Toh jab WebSocket se coordinates aate hain, Canvas unko raw video frame ke bilkul upar exactly draw kar deta hai.
*   **Output:** Ek smooth, lag-free live video feed jisme weapons aur skeletons draw ho rahe hote hain. Agar threat aaye toh frontend automatically ek hooter (Siren) sound play karta hai aur screen par red alert banner show karta hai.
