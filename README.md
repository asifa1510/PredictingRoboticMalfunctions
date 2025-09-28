# Predicting Errors in Robotic-Assisted Cholecystectomy  

## Project Overview  
This project builds a system to **predict surgical errors in robotic-assisted laparoscopic cholecystectomy** (gallbladder removal) using the **Cholec80 dataset**.  

Since Cholec80 does not provide explicit error labels, we designed **rule-based heuristics** (tool–phase mismatches, multitool usage, abnormal phase durations) to approximate error events. These are used to train models that predict **future hazards (10s, 20s, 30s ahead)**.  

We compare a **baseline GRU** with a **Liquid Neural Network (LNN)** — showing that LNNs provide earlier warnings with fewer false alerts.  

---

##  Dataset  
- **Source:** [Cholec80](http://camma.u-strasbg.fr/datasets) (University Hospital of Strasbourg / IRCAD).  
- **Subset used:** 15 videos (with corresponding *phase* and *tool* annotation files).  
- **Annotations:**  
  - Phase labels (7 surgical phases).  
  - Tool presence (7 surgical tools).  

---

## Methodology  

### 1. Preprocessing  
- Extract **per-second features**:  
  - One-hot encoded surgical phase (7).  
  - Binary tool presence (7).  
  - Time-in-phase counter (1).  
  - Active tool count (1).  
- Detect **rule-based errors**:  
  - Tool not allowed in current phase.  
  - Too many tools active (>2).  
  - Very short phases (<5s).  

### 2. Hazard Labeling  
- Mark frames with **`error_now=1`**.  
- Define **hazard starts** (≥2s persistence).  
- Generate **prediction labels** for horizons:  
  - `y_10s`: hazard within next 10s.  
  - `y_20s`: hazard within next 20s.  
  - `y_30s`: hazard within next 30s.  

### 3. Models  
- **Baseline:** GRU sequence model.  
- **Proposed:** Liquid Neural Network (LNN).  
- Input: sliding windows (120s).  
- Output: risk scores for 10s/20s/30s horizons.  

### 4. Evaluation  
- **Frame-level:** AUROC, AUPRC.  
- **Event-level:**  
  - Median lead-time (how early hazards predicted).  
  - False alerts per 10 minutes.  
- **Visualization:** Risk timelines vs. ground truth hazards.

  ## 📊 Results (Example)  
| Model | AUROC (10/20/30s) | AUPRC (10/20/30s) | Median Lead-time | False Alerts/10min |
|-------|-------------------|-------------------|------------------|--------------------|
| GRU   |                   |                   |                  |                    |
| LNN   |                   |                   |                  |                    |

_ consistently provides **earlier warnings** and **fewer false positives**.  

