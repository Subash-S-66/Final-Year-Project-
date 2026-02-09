# PHASE-6B — Mode B2 (Webcam + Real-Time Extraction)
## Inputs
- Dataset: `data/processed/manual_v1` (capture output)
- Collection plan: `reports/collection_plan_v1.csv`

## Strategy
- Single signer, multi-session (C)
- 3 sessions: S1 baseline, S2 mild angle/light shift, S3 faster signing
- Included tokens: P0 + P1 only

## Token checklist (P0 + P1)
```text
priority count extra S1  S2  S3  y   label                 
-------- ----- ----- --- --- --- --- ----------------------
P0       0     20    8   6   6   44  DOCTOR                
P0       0     20    8   6   6   1   HELLO                 
P0       0     20    8   6   6   27  HERE                  
P0       0     20    8   6   6   24  HOW                   
P0       0     20    8   6   6   50  I_LOVE_YOU            
P0       0     20    8   6   6   0   NO_SIGN               
P0       0     20    8   6   6   37  THIRSTY               
P0       0     20    8   6   6   22  WHEN                  
P1       1     15    5   5   5   47  EMERGENCY             
P1       1     15    5   5   5   35  FOOD                  
P1       1     15    5   5   5   13  GO                    
P1       1     15    5   5   5   9   HELP                  
P1       1     15    5   5   5   39  HOME                  
P1       1     15    5   5   5   36  HUNGRY                
P1       1     15    5   5   5   14  ME                    
P1       1     15    5   5   5   17  MY                    
P1       1     15    5   5   5   8   NO                    
P1       1     15    5   5   5   29  NOW                   
P1       1     15    5   5   5   40  SCHOOL                
P1       1     15    5   5   5   26  THAT                  
P1       1     15    5   5   5   28  THERE                 
P1       1     15    5   5   5   16  WE                    
P1       1     15    5   5   5   20  WHAT                  
P1       1     15    5   5   5   21  WHERE                 
P1       1     15    5   5   5   7   YES                   
P1       2     15    5   5   5   12  COME                  
P1       2     15    5   5   5   42  FRIEND                
P1       2     15    5   5   5   5   PLEASE                
P1       2     15    5   5   5   10  STOP                  
P1       2     15    5   5   5   25  THIS                  
P1       2     15    5   5   5   33  TIME                  
P1       2     15    5   5   5   30  TODAY                 
P1       2     15    5   5   5   38  TOILET                
P1       2     15    5   5   5   31  TOMORROW              
P1       2     15    5   5   5   23  WHY                   
P1       2     15    5   5   5   32  YESTERDAY             
P1       2     15    5   5   5   15  YOU                   
P1       2     15    5   5   5   18  YOUR                  
```

## Session schedule
| Session | Goal | Setup | Notes |
|---|---|---|---|
| S1 | Baseline lighting + angle | Same spot, stable lighting | Prioritize clean, consistent form |
| S2 | Mild angle change + lighting shift | Slight left/right shift; slightly brighter/dimmer | Keep background similar |
| S3 | Slightly faster signing speed | Same as S1 or S2 | Increase tempo, keep clarity |

## Expected workload
- S1 samples: 214
- S2 samples: 198
- S3 samples: 198
- Total new samples: 610

## Capture (PowerShell)
Run the generated script:
- `reports/capture_manual_v1_mode_b2.ps1`
