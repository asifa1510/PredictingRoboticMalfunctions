# 🩺 Predicting Errors in Robotic-Assisted Cholecystectomy  

## 📘 Project Overview  
This project develops a **hybrid deep learning system** to predict **surgical errors and hazardous events** during **robotic-assisted laparoscopic cholecystectomy** (gallbladder removal).  

It extends and builds upon the study:  
> **“Predicting Errors in Robotic-Assisted Surgery Using Temporal and Contextual Signals”**  
> [arXiv:2412.12238v1](https://arxiv.org/html/2412.12238v1)

That research inspired the **rule-based hazard modeling** and **GRU/LNN comparison** in [Notebook 2](https://github.com/asifa1510/PredictingRoboticMalfunctions/blob/main/Robotic_errorsGRU.ipynb), which was further extended with **ECRCD kinematic data** and a **hybrid multi-domain fusion model** in [Notebook 3](https://github.com/asifa1510/PredictingRoboticMalfunctions/blob/main/hybrid-ecrcd-cholec80x.ipynb).  
The original Cholec80 GRU baseline setup can be found in [Notebook 1](https://github.com/asifa1510/PredictingRoboticMalfunctions/blob/main/Surical-error_Cholec80GRU.ipynb).  

---

## 💡 Why This Project Matters  
In robotic-assisted surgery, even subtle deviations in motion, control, or workflow timing can cascade into serious surgical errors.  
However, existing systems mainly **detect** errors *after* they occur — none focus on **predicting** them in real time.  

This project addresses that gap by:  
1. **Anticipating surgical hazards 10–30 seconds before they occur.**  
2. Combining **workflow intelligence** (from Cholec80) with **robot kinematic signals** (from ECRCD).  
3. Enabling **real-time, proactive risk mitigation** in surgical robotics.  

This capability can help reduce human error, alert operating surgeons earlier, and contribute to safer, semi-autonomous robotic systems.

---

## 📦 Datasets  

### **1️⃣ Cholec80 (Contextual Base Dataset)**  
- **Source:** [IRCAD / University Hospital of Strasbourg](http://camma.u-strasbg.fr/datasets)  
- **Subset Used:** 15 annotated surgical videos  
- **Annotations:**  
  - 7 surgical phases (workflow context)  
  - 7 tool presence indicators  
- **Purpose:** Rule-based error modeling, workflow consistency validation, and GRU/LNN baseline training  
- **Reference Implementation:** [Notebook 2](https://github.com/asifa1510/PredictingRoboticMalfunctions/blob/main/Robotic_errorsGRU.ipynb)

---

## 🧮 Cholec80 Rules & Preprocessing  

### **Feature Extraction (per-second features):**  
- One-hot encoded surgical phase (7)  
- Binary tool presence (7)  
- Time-in-phase counter (1)  
- Active tool count (1)

---

### **Rule-Based Error Detection:**  
The following rule set was used to infer *error_now* flags when the dataset lacked explicit error labels:  

- **Tool–Phase Mismatch:** Tool usage inconsistent with current surgical phase.  
- **Too Many Tools Active (>2):** Exceeding safe parallel operation.  
- **Too Short Phases (<5s):** Indicates premature phase transitions.  
- **Phase Order Violation:** Deviations from expected surgical sequence.  
- **Required Tool Missing:** Critical phase executed without necessary instruments.  
- **Prolonged Inactivity (>30s):** No active tool detected for extended periods.  
- **Abrupt Tool Switches (>3 in 5s):** Indicates operational instability.  
- **Phase Overrun:** Phases lasting significantly longer than expected.  
- **Forbidden Tool Combinations:** Unsafe or unrealistic instrument pairings.  
- **Tool Appears Too Early:** Tool introduced before its expected procedural phase.  

---

### **2️⃣ Hazard Labeling**  
After error detection, temporal hazard labels are created to enable predictive modeling.

1. **Immediate Error Detection**  
   - Mark frames with `error_now = 1` if any rule condition triggers.  

2. **Hazard Persistence**  
   - Define “hazard starts” as sequences persisting ≥2 seconds.  

3. **Predictive Horizons**  
   - `y_10s`: hazard likely within the next 10 seconds  
   - `y_20s`: hazard likely within the next 20 seconds  
   - `y_30s`: hazard likely within the next 30 seconds  

This structure allows models to **anticipate** errors instead of merely classifying them post hoc.

---

### **2️⃣ ECRCD / EXTCRCD (Kinematic Dataset)**  
- **Type:** Robotic control and joint telemetry logs (Parquet format)  
- **Signals:** ECM, MTML, MTMR, PSM1, PSM2 arms, and pedal states  
- **Used For:** Learning precise motion signatures correlated with Cholec80-derived hazard patterns  
- **Reference Implementation:** [Notebook 3](https://github.com/asifa1510/PredictingRoboticMalfunctions/blob/main/hybrid-ecrcd-cholec80x.ipynb)

---

## ⚙️ Methodology (Overview)  

### **1. Feature Engineering (ECRCD)**  
- Mean and standard deviation per joint and control point  
- Speed, jerk, and oscillation measures  
- Pedal state encoding  
- Per-video normalization using robust MAD  

### **2. Temporal Modeling**  
- Sliding windows of **300 frames (~12s)**  
- Weighted sampling to balance hazard classes  
- Computed per-class `pos_weight` for BCE loss balancing  

---

## 🧠 Training Setup  

- **Loss Function:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)  
- **Class Balancing:** `pos_weight` calculated dynamically from label ratios  
- **Optimizer:** AdamW  
- **Learning Rate Scheduler:** ReduceLROnPlateau (adaptive halving on plateau)  
- **Precision:** Mixed precision (AMP with FP16 or BF16 on GPU)  
- **Training Schedule:**  
  1. Freeze pretrained GRU backbones  
  2. Train hybrid fusion head + adapter  
  3. Gradually unfreeze for fine-tuning  
  4. Early stopping (patience = 4 epochs)  

---

## 📈 Evaluation Metrics  

| Horizon | AUROC | AUPRC | F1 | Precision | Recall |
|----------|-------|-------|----|------------|---------|
| **10 s** | 0.985 | 0.977 | 0.935 | 0.918 | 0.954 |
| **20 s** | 0.984 | 0.965 | 0.942 | 0.939 | 0.944 |
| **30 s** | 0.991 | 0.980 | 0.955 | 0.931 | 0.979 |

- High AUROC (>0.98) and AUPRC (>0.96) demonstrate exceptional reliability.  
- Consistent precision and recall across all horizons prove stability and generalization.  
- Model effectively anticipates errors **10–30 seconds before occurrence**.  

---

## ⚖️ Baseline Comparison (Notebook 2 / Cholec80)  

| Model | AUROC (10/20/30 s) | AUPRC (10/20/30 s) |
|--------|--------------------|--------------------|
| **GRU** | 0.940 / 0.925 / 0.917 | 0.914 / 0.896 / 0.887 |
| **LNN** | 0.685 / 0.681 / 0.592 | 0.590 / 0.587 / 0.544 |

As verified in [Notebook 2](https://github.com/asifa1510/PredictingRoboticMalfunctions/blob/main/Robotic_errorsGRU.ipynb)  
and discussed in [arXiv:2412.12238v1](https://arxiv.org/html/2412.12238v1),  
the **GRU model** consistently outperforms Liquid Neural Networks (LNNs) in both stability and early-warning accuracy.  

---
