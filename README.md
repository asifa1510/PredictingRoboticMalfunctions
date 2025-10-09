  # Predicting Errors in Robotic-Assisted Cholecystectomy  
 
## Project Overview  
This project builds a system to **predict surgical errors in robotic-assisted laparoscopic cholecystectomy** (gallbladder removal) using the **Cholec80 dataset**.  

Since Cholec80 does not provide explicit error labels, we designed **rule-based heuristics** (tool–phase mismatches, multitool usage, abnormal phase durations) to approximate error events. These are used to train models that predict **future hazards (10s, 20s, 30s ahead)**.  

We compare a **baseline GRU** with a **Liquid Neural Network (LNN)** — showing that GRUs provide earlier warnings with fewer false alerts.  

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
  -Tool–phase mismatch

  
  -Too many tools active (>2)

  
  -Too short phases (<5s)

  
  -Phase order violation

  
  -Required tool missing in critical phases

  
  -Prolonged inactivity (no tools for >30s)

  
  -Abrupt tool switches (>3 in 5s)

  
  -Phase overrun (too long)

  
  -Forbidden tool combinations

  
  -Tool appears too early (before expected phase)


### 2. Hazard Labeling  
- Mark frames with **`error_now=1`**.  
- Define **hazard starts** (≥2s persistence).  
- Generate **prediction labels** for horizons:  
  - `y_10s`: hazard within next 10s.  
  - `y_20s`: hazard within next 20s.  
  - `y_30s`: hazard within next 30s.  

### 3. Models  
- GRU sequence model.  
- Liquid Neural Network (LNN).  
- Input: sliding windows (120s).  
- Output: risk scores for 10s/20s/30s horizons.  

### 4. Evaluation  
- **Frame-level:** AUROC, AUPRC.  
- **Event-level:**  
  - Median lead-time (how early hazards predicted).  
  - False alerts per 10 minutes.  
- **Visualization:** Risk timelines vs. ground truth hazards.

  ## 📊 Results (Example)  
| Model | AUROC (10/20/30s) | AUPRC (10/20/30s) | False Alerts/10min 
|-------|-------------------|-------------------|--------------------------------------|
| GRU   | 0.940/0.925/0.917 |0.914/0.896/0.887  |                                      |
| LNN   | 0.685/0.681/0.592 |0.590/0.587/0.544  |                                      |

The GRU model consistently provides **earlier warnings** and **fewer false positives**.  


>The GRU model excels with low loss (0.5003), high AUROC (0.940/0.925/0.917), AUPRC (0.914/0.896/0.887), and balanced F1 (0.821/0.808/0.793), outperforming LNN. Monitor slight overfitting in later epochs.
>

<img width="1867" height="996" alt="image" src="https://github.com/user-attachments/assets/72b73d53-ec42-479c-8b47-e71fddd68048" />



level1 : predicting risk scores
<img width="1919" height="972" alt="image" src="https://github.com/user-attachments/assets/37a99c5b-736b-41fb-9235-106d7d1d22ad" />
level2: making accurate predictions on type of errors based on the rules fixed(developing)

