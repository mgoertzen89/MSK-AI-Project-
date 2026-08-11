import streamlit as st
import pandas as pd
from groq import Groq
import json
import os
import requests
import base64

# ==============================================================================
# 1. API & REPO CONFIGURATION & DIAGNOSTICS
# ==============================================================================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "").strip()
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "").strip()
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "").strip()  # Expected format: "username/repo-name"

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL_NAME = "llama-3.1-8b-instant"
DATA_FILE = "cases.json"

OBJECTIVE_CATEGORIES = [
    "Observation",
    "Active Range of Motion (AROM)",
    "Passive Range of Motion (PROM)",
    "Strength / Resisted Isometrics",
    "Functional Testing",
    "Palpation",
    "Special Tests"
]

# --- DEEP AUTH DIAGNOSTIC CHECK IN SIDEBAR ---
st.sidebar.markdown("### 🔍 GitHub Auth Debugger")
st.sidebar.caption("Verifying token and secret configuration:")

clean_repo = GITHUB_REPO.replace("https://github.com/", "").replace(".git", "").strip("/")
st.sidebar.text(f"Repo read: '{clean_repo}'")
st.sidebar.text(f"Token length: {len(GITHUB_TOKEN)} chars")
st.sidebar.text(f"Token prefix: {GITHUB_TOKEN[:4]}..." if GITHUB_TOKEN else "Token prefix: NONE")

if GITHUB_TOKEN and clean_repo:
    url = f"https://api.github.com/repos/{clean_repo}"
    
    # Test 1: Unauthenticated Public Check
    unauth_r = requests.get(url)
    st.sidebar.text(f"Public API Status: {unauth_r.status_code}")
    
    # Test 2: Authenticated Check
    auth_headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    auth_r = requests.get(url, headers=auth_headers)
    st.sidebar.text(f"Auth API Status: {auth_r.status_code}")
    
    if auth_r.status_code == 200:
        st.sidebar.success(f" Connected to {clean_repo}")
    else:
        st.sidebar.error(f"GitHub Error ({auth_r.status_code}): {auth_r.json().get('message', 'Unknown Error')}")
else:
    st.sidebar.warning(" Missing GITHUB_TOKEN or GITHUB_REPO in secrets.")

st.sidebar.markdown("---")

# ==============================================================================
# 2. REGION-SPECIFIC DEFAULT OBJECTIVE FINDINGS TEMPLATES
# ==============================================================================
def get_default_objective_template_for_region(region_name):
    r = str(region_name).lower()
    
    if "cervical" in r or "neck" in r:
        return {
            "Observation": "Forward head posture, protracted shoulder girdle, hypertonic upper trapezius visual bulk.",
            "Active Range of Motion (AROM)": "Flexion: 40° (Full = 50°). Extension: 45° (Full = 60°). Left Rotation: 60° painful at end-range (Full = 80°). Right Rotation: 70°. Side Bending B/L: 30° painful.",
            "Passive Range of Motion (PROM)": "Flexion: Full range with tissue-stretch end-feel. Extension: Full range. Rotation: 70° with muscular tightness end-feel.",
            "Strength / Resisted Isometrics": "Cervical Flexion: 4/5 painful. Cervical Extension: 5/5 pain-free. Cervical Side Bending B/L: 4+/5. Deep Cervical Flexors (CCFT): Impaired endurance.",
            "Functional Testing": "Sustained Neck Flexion (Desk posture simulation): Reproduces familiar neck/upper back ache at 60 seconds.",
            "Palpation": "Cervical Paraspinals (C4-C6): Moderately tender. Upper Trapezius & Levator Scapulae: Markedly tender with active trigger points. Spinous processes: Non-tender.",
            "Special Tests": "Spurling Test: Positive for localized neck pain (Negative for arm radiculopathy). Cervical Distraction Test: Reduces feeling of heaviness. Upper Limb Tension Tests (ULTT 1/Median): Negative."
        }
    elif "lumbar" in r or "back" in r:
        return {
            "Observation": "Flattened lumbar lordosis, antalgic posture, guarded transfer movements.",
            "Active Range of Motion (AROM)": "Flexion: 40° painful (Finger-to-floor distance 25cm). Extension: 10° painful. Side Bending B/L: 15° restricted.",
            "Passive Range of Motion (PROM)": "Flexion: Limited by muscle guarding. Extension: Limited with firm end-feel.",
            "Strength / Resisted Isometrics": "Lumbar Extension: 4/5 painful. Hip Flexion B/L: 5/5. Knee Extension (L4): 5/5. Great Toe Extension (L5): 5/5. Plantarflexion (S1): 5/5.",
            "Functional Testing": "Sit-to-Stand transfer: Slow, uses arms for assistance. Repeated Forward Bending: Increases lumbar pain.",
            "Palpation": "Lumbar Erector Spinae (L3-L5): Bilateral hypertonicity and tenderness. Quadratus Lumborum: Moderately tender. L4/L5 Spinous processes: Mildly tender.",
            "Special Tests": "Straight Leg Raise (SLR): Negative for radicular shooting pain below knee. Slump Test: Negative. Lumbar Quadrant Test: Reproduces localized L/S pain."
        }
    elif "shoulder" in r:
        return {
            "Observation": "Slight anterior hitch of humeral head, mild sulcus asymmetry, muscle guarding.",
            "Active Range of Motion (AROM)": "Flexion: 130° painful (Full = 180°). Abduction: 90° with painful arc between 60-120°. External Rotation: 50°. Internal Rotation: 45°.",
            "Passive Range of Motion (PROM)": "Flexion: 150° end-range discomfort. Abduction: 130° painful. External Rotation: Full. Internal Rotation: Full.",
            "Strength / Resisted Isometrics": "Flexion: 4/5. Abduction: 3+/5 painful. External Rotation: 4-/5 painful. Internal Rotation: 5/5 pain-free.",
            "Functional Testing": "Overhead reaching test: Reproduces primary pain at 90°. Hand-behind-back reach: Limited to L4 level.",
            "Palpation": "Supraspinatus insertion at greater tubercle: Markedly tender. Bicipital groove: Non-tender. AC Joint: Non-tender.",
            "Special Tests": "Hawkins-Kennedy Test: Positive. Neer Impingement Test: Positive. Empty Can (Jobe) Test: Positive for weakness and pain. Apprehension Test: Negative."
        }
    elif "elbow" in r:
        return {
            "Observation": "Carrying angle normal (10-15°), no gross joint effusion, holding arm guarded in 90° flexion.",
            "Active Range of Motion (AROM)": "Flexion: 135° (Full = 145°). Extension: -5° (Full = 0°). Pronation: 80°. Supination: 75°.",
            "Passive Range of Motion (PROM)": "Flexion: Full with soft tissue-approximation. Extension: Full with hard end-feel.",
            "Strength / Resisted Isometrics": "Wrist Extension (Cozen's): 3+/5 painful. Wrist Flexion: 5/5 pain-free. Elbow Flexion: 5/5. Elbow Extension: 5/5.",
            "Functional Testing": "Grip strength testing (Dynamometer): Significantly reduced on affected side due to pain at elbow.",
            "Palpation": "Lateral Epicondyle: Exquisitely tender to touch. Medial Epicondyle: Non-tender. Radial Head: Non-tender.",
            "Special Tests": "Cozen's Test (Resisted Wrist Extension): Positive. Mill's Test (Passive Wrist Flexion/Pronation): Positive. Golfer's Elbow Test: Negative."
        }
    elif "wrist" in r or "hand" in r:
        return {
            "Observation": "Mild localized swelling over radial wrist, normal muscle bulk in thenar/hypothenar eminences.",
            "Active Range of Motion (AROM)": "Wrist Flexion: 60° (Full = 80°). Wrist Extension: 55° (Full = 70°). Radial Deviation: 10° painful (Full = 20°). Ulnar Deviation: 20°.",
            "Passive Range of Motion (PROM)": "Wrist Flexion/Extension: Full. Radial Deviation: Limited by sharp localized pain.",
            "Strength / Resisted Isometrics": "Resisted Thumb Abduction: 4/5 painful. Grip Strength: 80% of unaffected side.",
            "Functional Testing": "Pinch grip test (Key & Tip pinch): Reproduces thumb-side wrist pain. Jar opening: Unable due to sharp pain.",
            "Palpation": "1st Dorsal Compartment (Abductor Pollicis Longus / Extensor Pollicis Brevis tendons): Highly tender. Anatomical Snuffbox: Non-tender.",
            "Special Tests": "Finkelstein's Test: Positive (sharp pain over radial styloid). Eichhoff's Test: Positive. Tinel's at Carpal Tunnel: Negative."
        }
    elif "hip" in r:
        return {
            "Observation": "Antalgic gait with shortened stance phase on affected side, Trendelenburg sign negative.",
            "Active Range of Motion (AROM)": "Flexion: 100° painful (Full = 120°). Extension: 10° (Full = 20°). Abduction: 30° painful. Internal Rotation: 15° limited (Full = 45°). External Rotation: 35°.",
            "Passive Range of Motion (PROM)": "Flexion: 110° with hard/capsular end-feel. Internal Rotation: 20° with deep groin pinching.",
            "Strength / Resisted Isometrics": "Hip Abduction: 4/5 painful. Hip Flexion: 4+/5. Hip Extension: 5/5. Internal Rotation: 4/5 painful.",
            "Functional Testing": "Single-Leg Stance: Stable for 10s. Deep Squat: Limited to 60° knee bend due to groin pinching.",
            "Palpation": "Greater Trochanter: Moderately tender. Deep Groin / Femoral Triangle: Tender to deep palpation. Ischial Tuberosity: Non-tender.",
            "Special Tests": "FADDIR Test (Flexion, Adduction, Internal Rotation): Positive for deep groin pain. FABER / Patrick's Test: Positive for lateral/groin pain. Thomas Test: Positive for hip flexor tightness."
        }
    elif "knee" in r:
        return {
            "Observation": "Mild intra-articular joint effusion (1+ sweep test), no visible alignment deformity (Genu Varum/Valgum normal).",
            "Active Range of Motion (AROM)": "Flexion: 115° painful (Full = 135°). Extension: -5° lack of full extension.",
            "Passive Range of Motion (PROM)": "Flexion: 125° with tissue-stretch end-feel. Extension: 0° with springy block end-feel.",
            "Strength / Resisted Isometrics": "Quadriceps (Knee Extension): 4/5 painful. Hamstrings (Knee Flexion): 5/5 pain-free.",
            "Functional Testing": "Single-leg Hop: Hesitant, unable to perform smoothly. Step-down test: Painful at 45° flexion.",
            "Palpation": "Medial Joint Line: Point tender. Patellar Tendon: Non-tender. Lateral Joint Line: Non-tender. Anserine Bursa: Non-tender.",
            "Special Tests": "McMurray Test: Positive for medial joint line click/pain. Lachman Test: Negative (Firm end-point). Anterior Drawer: Negative. Patellar Apprehension: Negative."
        }
    elif "ankle" in r or "foot" in r:
        return {
            "Observation": "Ecchymosis and edema localized below lateral malleolus, antalgic gait favoring heel-strike.",
            "Active Range of Motion (AROM)": "Dorsiflexion: 10° (Full = 20°). Plantarflexion: 35° (Full = 50°). Inversion: 15° painful (Full = 30°). Eversion: 15°.",
            "Passive Range of Motion (PROM)": "Inversion: Limited by sharp pain over lateral ligaments. Dorsiflexion: Limited by Achilles tightness.",
            "Strength / Resisted Isometrics": "Ankle Eversion (Peroneals): 4/5. Inversion: 4/5 painful. Plantarflexion (Gastrocnemius): 5/5.",
            "Functional Testing": "Single-leg Heel Raise: Able to perform 3 reps with mild wobble. Tandem gait: Unstable.",
            "Palpation": "Anterior Talofibular Ligament (ATFL): Highly tender. Calcaneofibular Ligament (CFL): Moderately tender. Lateral Malleolus Bone: Non-tender.",
            "Special Tests": "Anterior Drawer Test (Ankle): Positive for mild laxity compared to contralateral side. Talar Tilt Test: Positive for pain. Thompson Squeeze Test: Negative (Achilles intact)."
        }
    else:
        return {
            "Observation": "Postural alignment: Guarded position. Slight asymmetry noted on affected side.",
            "Active Range of Motion (AROM)": "Flexion: 75% available with pain at end-range. Extension: Full, pain-free. Lateral movements: Mildly restricted.",
            "Passive Range of Motion (PROM)": "Flexion: Full range with tissue-stretch end-feel and mild discomfort. Extension: Unrestricted.",
            "Strength / Resisted Isometrics": "Primary movers: 4/5 with pain elicited on strong contraction. Surrounding stabilizers: 5/5 non-tender.",
            "Functional Testing": "Functional movement test: Reproduces primary complaint at end-range loading.",
            "Palpation": "Point tenderness noted over local tendinous insertion. Surrounding muscular hypertonicity present.",
            "Special Tests": "Primary provocative test: Positive. Secondary stability tests: Negative."
        }

# ==============================================================================
# 3. COMPLETE DEFAULT CASE LIBRARY (48 PATIENT CASES)
# ==============================================================================
DEFAULT_CASE_LIBRARY = {
    "Cervical spine": {
        "Case 1": {
            "name": "Arthur", "region_label": "Cervical spine", "forthcomingness": 1,
            "demeanor": "Rubbing neck, sits slouched forward.",
            "chief_complaint": "Persistent ache across upper back and neck after long hours at computer.",
            "history_present_illness": "Dull ache developed over 3 months as work demands increased.",
            "location_pain": "Bilateral upper trapezius and mid-cervical paraspinals.",
            "onset_pain": "Insidious onset over 12 weeks.",
            "type_pain": "Constant heavy ache, muscular tightness.",
            "aggravating_factors": "Prolonged desk work, head-forward postures.",
            "easing_factors": "Heat packs, gentle stretching, lying down flat.",
            "radiation": "None into arms.",
            "red_flags": "Denies upper extremity numbness, weakness, or clumsiness.",
            "social_history": "Software developer, works 10-hour days.",
            "past_medical_history": "None.",
            "diff_dx": "Mechanical Neck Pain (Postural Strain)"
        },
        "Case 2": {
            "name": "Brenda", "region_label": "Cervical spine", "forthcomingness": 1,
            "demeanor": "Holding neck stiffly, avoids turning head.",
            "chief_complaint": "Sharp neck pain radiating down right arm to thumb.",
            "history_present_illness": "Sudden onset 1 week ago after lifting heavy container.",
            "location_pain": "Right side of neck shooting into shoulder and lateral forearm.",
            "onset_pain": "Acute onset 7 days ago.",
            "type_pain": "Sharp, burning, electric-shock quality.",
            "aggravating_factors": "Cervical extension, side-bending to the right, coughing.",
            "easing_factors": "Placing right hand on top of head (Distraction/Relief sign).",
            "radiation": "C6 dermatome down to index/thumb.",
            "red_flags": "Reports mild numbness in thumb, denies gait changes or bowel/bladder issues.",
            "social_history": "Warehouse supervisor.",
            "past_medical_history": "Hypertension.",
            "diff_dx": "Cervical Radiculopathy (Right C6)"
        },
        "Case 3": {
            "name": "Charles", "region_label": "Cervical spine", "forthcomingness": 1,
            "demeanor": "Guarded movements, turns whole torso to look sideways.",
            "chief_complaint": "Whiplash injury after rear-end car collision 2 weeks ago.",
            "history_present_illness": "Was stopped at red light when struck from behind; neck felt sore next morning.",
            "location_pain": "Suboccipital region down to thoracic spine.",
            "onset_pain": "Acute post-traumatic onset 14 days ago.",
            "type_pain": "Stiff, aching pain with occasional sharp twinges.",
            "aggravating_factors": "Any rapid neck rotation or bumpy car rides.",
            "easing_factors": "Soft cervical collar temporary use, OTC ibuprofen.",
            "radiation": "Refers across bilateral shoulders.",
            "red_flags": "Denies dizziness, diplopia, dysarthria, dysphagia, or drop attacks.",
            "social_history": "Accountant, high stress.",
            "past_medical_history": "None.",
            "diff_dx": "Cervical Spine Acceleration-Deceleration Injury (Whiplash Associated Disorder)"
        },
        "Case 4": {
            "name": "Diana", "region_label": "Cervical spine", "forthcomingness": 1,
            "demeanor": "Frequent massaging of suboccipital area and temples.",
            "chief_complaint": "Daily headaches starting from back of neck and wrapping around head.",
            "history_present_illness": "Neck tightness evolving into daily headaches for past 2 months.",
            "location_pain": "Suboccipital neck radiating to forehead and behind right eye.",
            "onset_pain": "Gradual progression over 8 weeks.",
            "type_pain": "Dull band-like pressure, fluctuating intensity.",
            "aggravating_factors": "Sustained neck flexion, reading, desk work.",
            "easing_factors": "Suboccipital pressure, dark room, neck traction.",
            "radiation": "Unilateral temporoparietal and retro-orbital region.",
            "red_flags": "No visual scotomas, no nausea/vomiting, no focal neuro deficits.",
            "social_history": "High school teacher.",
            "past_medical_history": "Migraines in early adulthood.",
            "diff_dx": "Cervicogenic Headache"
        },
        "Case 5": {
            "name": "Edward", "region_label": "Cervical spine", "forthcomingness": 1,
            "demeanor": "Elderly male, walks with slightly wide base, cautious.",
            "chief_complaint": "Clumsy hands, stiffness in neck, and difficulty buttoning shirt.",
            "history_present_illness": "Over 6 months noticed declining hand dexterity and unsteadiness walking.",
            "location_pain": "Central neck stiffness, diffuse hand paresthesias.",
            "onset_pain": "Progressive chronic onset.",
            "type_pain": "Aching stiffness with bilateral glove-like tingling.",
            "aggravating_factors": "Cervical extension.",
            "easing_factors": "Resting seated in recliner.",
            "radiation": "Bilateral upper extremities and unsteadiness down legs.",
            "red_flags": "Positive Hoffmann sign potential, hyperreflexia, loss of fine motor skills.",
            "social_history": "Retired Carpenter.",
            "past_medical_history": "Osteoarthritis, Cervical Spondylosis.",
            "diff_dx": "Cervical Spondylotic Myelopathy"
        },
        "Case 6": {
            "name": "Fiona", "region_label": "Cervical spine", "forthcomingness": 1,
            "demeanor": "Young athlete, holding neck to left side.",
            "chief_complaint": "Acute onset of severe neck pain after awkward sleeping position.",
            "history_present_illness": "Woke up 2 days ago completely unable to turn head to the right.",
            "location_pain": "Unilateral sternocleidomastoid and splenius cervicis.",
            "onset_pain": "Acute onset 48 hours ago.",
            "type_pain": "Sharp spasm on attempted motion.",
            "aggravating_factors": "Right rotation and lateral flexion.",
            "easing_factors": "Heat, topical analgesic creams, muscle relaxants.",
            "radiation": "None.",
            "red_flags": "Denies fever, night sweats, or neurological deficits.",
            "social_history": "University student.",
            "past_medical_history": "None.",
            "diff_dx": "Acute Cervical Facet Torticollis"
        }
    },
    "Lumbar spine": {
        "Case 1": {
            "name": "George", "region_label": "Lumbar spine", "forthcomingness": 1,
            "demeanor": "Stands up slowly using hands on thighs (Gower sign).",
            "chief_complaint": "Lower back pain after bending to lift heavy box at home.",
            "history_present_illness": "Felt a 'pop' in lower back 3 days ago followed by intense spasms.",
            "location_pain": "L4-S1 lumbar region extending to gluteal folds.",
            "onset_pain": "Acute onset 3 days ago.",
            "type_pain": "Deep aching spasm, sharp with movement.",
            "aggravating_factors": "Forward bending, lifting, sitting in deep chair.",
            "easing_factors": "Walking short distances, supine with knees elevated.",
            "radiation": "Gluteal region bilateral, does not go past knees.",
            "red_flags": "Denies urinary incontinence, saddle anesthesia, or foot drop.",
            "social_history": "Construction estimator.",
            "past_medical_history": "None.",
            "diff_dx": "Acute Lumbar Strain / Discogenic Low Back Pain"
        },
        "Case 2": {
            "name": "Hannah", "region_label": "Lumbar spine", "forthcomingness": 1,
            "demeanor": "Leaning to the left when standing (antalgic shift).",
            "chief_complaint": "Sharp pain shooting down back of left leg to big toe.",
            "history_present_illness": "Back pain started 2 weeks ago, transformed into severe leg pain 3 days ago.",
            "location_pain": "Left L5 distribution down posterior-lateral leg.",
            "onset_pain": "Subacute onset over 14 days.",
            "type_pain": "Electric, burning, shooting pain.",
            "aggravating_factors": "Sitting, coughing, sneezing, bending forward.",
            "easing_factors": "Standing, walking, extension positioning.",
            "radiation": "Posterolateral thigh, calf, dorsum of foot.",
            "red_flags": "Mild weakness in dorsiflexion, denies bowel/bladder changes.",
            "social_history": "Nurse, frequent patient transfers.",
            "past_medical_history": "Low back pain history.",
            "diff_dx": "Lumbar Radiculopathy (Left L5 Disc Herniation)"
        },
        "Case 3": {
            "name": "Ian", "region_label": "Lumbar spine", "forthcomingness": 1,
            "demeanor": "Stooped forward posture while pushing shopping cart.",
            "chief_complaint": "Leg heaviness and cramping after walking 5 minutes.",
            "history_present_illness": "Gradual reduction in walking tolerance over past year.",
            "location_pain": "Bilateral thighs, calves, and lower lumbar region.",
            "onset_pain": "Insidious chronic progression over 12 months.",
            "type_pain": "Dull ache, numbness, cramping heaviness.",
            "aggravating_factors": "Walking, lumbar extension, standing straight.",
            "easing_factors": "Sitting, bending forward ('shopping cart sign').",
            "radiation": "Bilateral lower extremities.",
            "red_flags": "Neurogenic claudication symptoms; vascular pulses intact.",
            "social_history": "Retired postman.",
            "past_medical_history": "Hypertension, Osteoarthritis.",
            "diff_dx": "Lumbar Spinal Stenosis"
        },
        "Case 4": {
            "name": "Julia", "region_label": "Lumbar spine", "forthcomingness": 1,
            "demeanor": "Arching back frequently, points to localized spot on back.",
            "chief_complaint": "Localized lower back pain worse when standing and twisting.",
            "history_present_illness": "Persistent back pain after starting gymnastics rotation 1 month ago.",
            "location_pain": "Unilateral lower lumbar (L5 area).",
            "onset_pain": "Gradual onset over 4 weeks.",
            "type_pain": "Sharp focal pain with extension.",
            "aggravating_factors": "Lumbar extension, rotation, single-leg stance loading.",
            "easing_factors": "Flexion, bed rest.",
            "radiation": "None.",
            "red_flags": "Denies neurological symptoms in legs.",
            "social_history": "High school gymnast.",
            "past_medical_history": "None.",
            "diff_dx": "Lumbar Spondylolysis / Facet Joint Arthropathy"
        },
        "Case 5": {
            "name": "Kevin", "region_label": "Lumbar spine", "forthcomingness": 1,
            "demeanor": "Shifting weight side to side while standing.",
            "chief_complaint": "Deep aching pain over posterior pelvis near beltline.",
            "history_present_illness": "Pain developed after stepping unexpectedly off a curb 3 weeks ago.",
            "location_pain": "Sacroiliac joint sulcus, dimple of Venus area.",
            "onset_pain": "Acute traumatic onset 21 days ago.",
            "type_pain": "Dull ache with sharp catch during weight transfer.",
            "aggravating_factors": "Single leg standing, rolling over in bed, sit-to-stand.",
            "easing_factors": "SI belt support, lying supine, ice.",
            "radiation": "Posterior thigh down to knee level.",
            "red_flags": "Denies radicular pain below knee.",
            "social_history": "Delivery driver.",
            "past_medical_history": "Hamstring strain.",
            "diff_dx": "Sacroiliac Joint Dysfunction"
        },
        "Case 6": {
            "name": "Laura", "region_label": "Lumbar spine", "forthcomingness": 1,
            "demeanor": "Anxious, holding lumbar region with both hands.",
            "chief_complaint": "Constant dull lumbar ache that wakes her up at night.",
            "history_present_illness": "Unremitting back pain for 6 weeks, not responding to rest.",
            "location_pain": "Mid-lumbar spine.",
            "onset_pain": "Insidious onset 6 weeks ago.",
            "type_pain": "Deep, boring constant pain.",
            "aggravating_factors": "Nighttime rest, recumbent positioning.",
            "easing_factors": "None significant.",
            "radiation": "Girdle-like around abdomen.",
            "red_flags": "Unexplained 10lb weight loss, constant night pain, fatigue.",
            "social_history": "Graphic designer.",
            "past_medical_history": "Breast cancer survivor in remission.",
            "diff_dx": "Non-Mechanical Spinal Pathology (Red Flag Screening Required)"
        }
    },
    "Shoulder": {
        "Case 1": {
            "name": "Sarah", "region_label": "Shoulder", "forthcomingness": 1,
            "demeanor": "Holding right arm close to side, avoids overhead reach.",
            "chief_complaint": "Anterior shoulder pain when reaching into upper cabinets.",
            "history_present_illness": "Pain started 6 weeks ago after painting garage ceiling.",
            "location_pain": "Anterolateral shoulder radiating down to mid-deltoid.",
            "onset_pain": "Gradual onset over 6 weeks.",
            "type_pain": "Sharp with overhead activity, dull ache at rest.",
            "aggravating_factors": "Reaching overhead, dressing, lying on affected side.",
            "easing_factors": "Rest, ice, holding arm supported.",
            "radiation": "Lateral arm down to insertion of deltoid.",
            "red_flags": "Denies neck pain, chest pain, or systemic weakness.",
            "social_history": "Recreational tennis player, office worker.",
            "past_medical_history": "None.",
            "diff_dx": "Subacromial Pain Syndrome / Rotator Cuff Tendinopathy"
        },
        "Case 2": {
            "name": "Marcus", "region_label": "Shoulder", "forthcomingness": 1,
            "demeanor": "Left arm supported in sling or cradled by right arm.",
            "chief_complaint": "Inability to lift left arm after fall on shoulder playing rugby.",
            "history_present_illness": "Tackled 4 days ago, felt sharp tear in shoulder.",
            "location_pain": "Lateral shoulder and upper arm.",
            "onset_pain": "Acute traumatic onset 4 days ago.",
            "type_pain": "Severe sharp pain on movement, weakness.",
            "aggravating_factors": "Attempted active elevation or abduction.",
            "easing_factors": "Sling immobilization, ice.",
            "radiation": "Deltoid muscle belly.",
            "red_flags": "True weakness on active elevation vs pain inhibition.",
            "social_history": "Rugby player, electrician.",
            "past_medical_history": "Prior shoulder subluxation.",
            "diff_dx": "Full-Thickness Rotator Cuff Tear (Supraspinatus)"
        },
        "Case 3": {
            "name": "Nancy", "region_label": "Shoulder", "forthcomingness": 1,
            "demeanor": "Rigid posture, arm held glued to torso.",
            "chief_complaint": "Severe global stiffness in right shoulder, cannot fasten bra.",
            "history_present_illness": "Gradual pain onset 4 months ago followed by progressive freezing.",
            "location_pain": "Deep shoulder joint, anterolateral capsule.",
            "onset_pain": "Insidious 4-month progression.",
            "type_pain": "Throb at night, severe sharp pain at end-range.",
            "aggravating_factors": "Reaching behind back, quick sudden movements.",
            "easing_factors": "Heat, gentle pendulum movements.",
            "radiation": "Down lateral upper arm.",
            "red_flags": "Denies acute trauma.",
            "social_history": "Executive secretary.",
            "past_medical_history": "Type 2 Diabetes Mellitus.",
            "diff_dx": "Adhesive Capsulitis (Frozen Shoulder)"
        },
        "Case 4": {
            "name": "Oliver", "region_label": "Shoulder", "forthcomingness": 1,
            "demeanor": "Young swimmer, frequently clicking shoulder.",
            "chief_complaint": "Shoulder feels like it slips out of place during swim stroke.",
            "history_present_illness": "Sensation of instability and dead arm feeling during overhead activity.",
            "location_pain": "Anterior shoulder joint line.",
            "onset_pain": "Recurrent over 6 months.",
            "type_pain": "Apprehension, slipping catch.",
            "aggravating_factors": "Abduction and external rotation position.",
            "easing_factors": "Resting arm at side.",
            "radiation": "Transient numbness down arm during slipping episode.",
            "red_flags": "Prior dislocation 1 year ago.",
            "social_history": "Competitive collegiate swimmer.",
            "past_medical_history": "Anterior shoulder dislocation.",
            "diff_dx": "Anterior Shoulder Instability / Labral Pathology (Bankart/SLAP)"
        },
        "Case 5": {
            "name": "Patricia", "region_label": "Shoulder", "forthcomingness": 1,
            "demeanor": "Points directly with finger to top of shoulder joint.",
            "chief_complaint": "Pain at top of shoulder when carrying heavy shopping bags or crossing arm.",
            "history_present_illness": "Developing over 3 months after starting cross-fit training.",
            "location_pain": "Superior acromioclavicular joint region.",
            "onset_pain": "Gradual onset over 12 weeks.",
            "type_pain": "Sharp focal ache.",
            "aggravating_factors": "Cross-body adduction, bench press, sleeping on side.",
            "easing_factors": "Avoiding overhead chest presses, NSAIDs.",
            "radiation": "Neck and upper trapezius localized.",
            "red_flags": "No deformity or step-off present.",
            "social_history": "Cross-fit enthusiast.",
            "past_medical_history": "None.",
            "diff_dx": "Acromioclavicular Joint Osteoarthritis / Sprain"
        },
        "Case 6": {
            "name": "Quentin", "region_label": "Shoulder", "forthcomingness": 1,
            "demeanor": "Rubbing front of biceps arm area.",
            "chief_complaint": "Throbbing pain in front of shoulder with audible clicking when rotating arm.",
            "history_present_illness": "Started after repetitive wood splitting 3 weeks ago.",
            "location_pain": "Anterior bicipital groove.",
            "onset_pain": "Subacute onset 21 days ago.",
            "type_pain": "Localized ache and snapping sensation.",
            "aggravating_factors": "Repetitive lifting, forearm supination with elbow flexed.",
            "easing_factors": "Ice, rest.",
            "radiation": "Down anterior biceps belly.",
            "red_flags": "No 'Popeye' muscle deformity noted.",
            "social_history": "Landscape gardener.",
            "past_medical_history": "None.",
            "diff_dx": "Biceps Tendinopathy / Tenosynovitis"
        }
    },
    "Elbow": {
        "Case 1": {
            "name": "Rachel", "region_label": "Elbow", "forthcomingness": 1,
            "demeanor": "Rubs lateral aspect of elbow, flexes wrist tentatively.",
            "chief_complaint": "Pain on outer elbow when lifting coffee cup or turning doorknobs.",
            "history_present_illness": "Developing over 2 months after increased computer mouse usage and tennis playing.",
            "location_pain": "Lateral epicondyle extending into wrist extensor mass.",
            "onset_pain": "Insidious onset 8 weeks ago.",
            "type_pain": "Burning ache, sharp with gripping.",
            "aggravating_factors": "Gripping objects, wrist extension under resistance, shaking hands.",
            "easing_factors": "Counterforce elbow strap, rest, ice.",
            "radiation": "Dorsal forearm down toward wrist.",
            "red_flags": "Denies hand weakness or numbness.",
            "social_history": "Recreational tennis player, administrative manager.",
            "past_medical_history": "None.",
            "diff_dx": "Lateral Elbow Tendinopathy (Lateral Epicondylalgia / Tennis Elbow)"
        },
        "Case 2": {
            "name": "Steven", "region_label": "Elbow", "forthcomingness": 1,
            "demeanor": "Cradles medial elbow with opposite hand.",
            "chief_complaint": "Inner elbow pain during golf swing impact and carrying heavy buckets.",
            "history_present_illness": "Pain worsened over 6 weeks of golf practice.",
            "location_pain": "Medial epicondyle and flexor-pronator mass origin.",
            "onset_pain": "Gradual onset over 6 weeks.",
            "type_pain": "Aching pain, sharp with wrist flexion/pronation.",
            "aggravating_factors": "Golfing, forceful wrist flexion, lifting palm up.",
            "easing_factors": "Rest, thermal packs.",
            "radiation": "Anteromedial forearm.",
            "red_flags": "Denies ulnar nerve tingling in 4th/5th digits.",
            "social_history": "Avid golfer, plumber.",
            "past_medical_history": "None.",
            "diff_dx": "Medial Elbow Tendinopathy (Medial Epicondylalgia / Golfer's Elbow)"
        },
        "Case 3": {
            "name": "Tanya", "region_label": "Elbow", "forthcomingness": 1,
            "demeanor": "Shakes out small finger and ring finger repeatedly.",
            "chief_complaint": "Numbness and tingling in pinky finger and elbow pain after flexed sitting.",
            "history_present_illness": "Tingling started 1 month ago, worse when leaning elbow on desk.",
            "location_pain": "Posteromedial elbow (cubital tunnel area).",
            "onset_pain": "Gradual onset 4 weeks ago.",
            "type_pain": "Paresthesias, dysesthesias, dull ache.",
            "aggravating_factors": "Sustained elbow flexion, sleeping with arms folded, leaning on elbows.",
            "easing_factors": "Extending elbow, elbow night splinting.",
            "radiation": "Medial forearm to 4th and 5th digits.",
            "red_flags": "Mild intrinsic hand muscle weakness developing.",
            "social_history": "Call center representative.",
            "past_medical_history": "None.",
            "diff_dx": "Cubital Tunnel Syndrome (Ulnar Nerve Entrapment)"
        },
        "Case 4": {
            "name": "Ulysses", "region_label": "Elbow", "forthcomingness": 1,
            "demeanor": "Large soft egg-shaped swelling visible behind tip of elbow.",
            "chief_complaint": "Swelling behind elbow joint after leaning on hard desk for weeks.",
            "history_present_illness": "Fluid bag appeared 2 weeks ago, mild dull pressure.",
            "location_pain": "Posterior olecranon process.",
            "onset_pain": "Gradual painless swelling over 14 days.",
            "type_pain": "Fullness, mild tender pressure.",
            "aggravating_factors": "Direct pressure resting elbow on table.",
            "easing_factors": "Compression sleeve, elevation.",
            "radiation": "None.",
            "red_flags": "No erythema, skin warmth, systemic fever, or purulent fluid.",
            "social_history": "Draftsman.",
            "past_medical_history": "Gout.",
            "diff_dx": "Olecranon Bursitis (Aseptic)"
        },
        "Case 5": {
            "name": "Victoria", "region_label": "Elbow", "forthcomingness": 1,
            "demeanor": "Arm held in neutral flexion, cautious on rotation.",
            "chief_complaint": "Deep anterior elbow pain after sudden heavy deadlift flex.",
            "history_present_illness": "Felt popping sensation in anterior elbow during weightlifting 5 days ago.",
            "location_pain": "Anterior cubital fossa over radial tuberosity.",
            "onset_pain": "Acute traumatic onset 5 days ago.",
            "type_pain": "Deep tear pain, weakness in supination.",
            "aggravating_factors": "Forearm supination against resistance, elbow flexion.",
            "easing_factors": "Flexed arm support.",
            "radiation": "Proximal anterior forearm.",
            "red_flags": "Mild ecchymosis, palpable distal biceps tendon present with strain.",
            "social_history": "Weightlifter.",
            "past_medical_history": "Anabolic use history.",
            "diff_dx": "Distal Biceps Tendon Strain / Partial Tear"
        },
        "Case 6": {
            "name": "Walter", "region_label": "Elbow", "forthcomingness": 1,
            "demeanor": "Holds elbow in 90 degrees flexion, unable to extend.",
            "chief_complaint": "Elbow pain and mechanical locking after fall on outstretched hand.",
            "history_present_illness": "Fell off bicycle 1 week ago, elbow jammed into extension.",
            "location_pain": "Lateral radial head and posterior elbow.",
            "onset_pain": "Acute onset 7 days ago.",
            "type_pain": "Sharp catching pain.",
            "aggravating_factors": "Forearm pronation/supination, terminal elbow extension.",
            "easing_factors": "Immobilization.",
            "radiation": "Lateral forearm.",
            "red_flags": "True mechanical block to full passive motion.",
            "social_history": "Cyclist.",
            "past_medical_history": "None.",
            "diff_dx": "Radial Head Fracture / Intra-articular Loose Body"
        }
    },
    "Wrist and Hand": {
        "Case 1": {
            "name": "Xena", "region_label": "Wrist and Hand", "forthcomingness": 1,
            "demeanor": "Shaking wrist and rubbing palm, flicking hand.",
            "chief_complaint": "Nighttime numbness in thumb, index, and middle fingers waking her up.",
            "history_present_illness": "Symptoms worsening over 3 months; drops small items like needles.",
            "location_pain": "Palmar aspect of wrist radiating into radial digits.",
            "onset_pain": "Insidious 12-week progression.",
            "type_pain": "Burning tingling paresthesias, nocturnal ache.",
            "aggravating_factors": "Sustained wrist flexion, driving, holding phone.",
            "easing_factors": "Shaking hand ('flick sign'), wrist cock-up splint.",
            "radiation": "Median nerve distribution (Digits 1-3 and radial half of 4).",
            "red_flags": "Thenar eminence mild atrophy, loss of two-point discrimination.",
            "social_history": "Assembly line worker.",
            "past_medical_history": "Hypothyroidism, Pregnancy 3rd trimester.",
            "diff_dx": "Carpal Tunnel Syndrome (Median Nerve Entrapment)"
        },
        "Case 2": {
            "name": "Yusuf", "region_label": "Wrist and Hand", "forthcomingness": 1,
            "demeanor": "Holding thumb rigid, avoids thumb movement.",
            "chief_complaint": "Sharp radial wrist pain when lifting baby or wringing out cloths.",
            "history_present_illness": "Onset 4 weeks ago after birth of new grandchild.",
            "location_pain": "Radial styloid process and 1st dorsal compartment.",
            "onset_pain": "Gradual onset over 1 month.",
            "type_pain": "Sharp catching pain over thumb tendon.",
            "aggravating_factors": "Thumb ulnar deviation with flexed thumb (Finkelstein motion), lifting.",
            "easing_factors": "Thumb spica splint, resting thumb.",
            "radiation": "Dorsal thumb and lateral forearm.",
            "red_flags": "No swelling over anatomical snuffbox bone.",
            "social_history": "New parent / primary caregiver.",
            "past_medical_history": "None.",
            "diff_dx": "De Quervain's Tenosynovitis"
        },
        "Case 3": {
            "name": "Zachary", "region_label": "Wrist and Hand", "forthcomingness": 1,
            "demeanor": "Massaging base of thumb continuously.",
            "chief_complaint": "Deep ache at base of thumb when opening jars or turning keys.",
            "history_present_illness": "Chronic progressive thumb stiffness over past 2 years.",
            "location_pain": "1st Carpometacarpal (CMC) joint base of thumb.",
            "onset_pain": "Chronic insidious onset.",
            "type_pain": "Deep grinding ache.",
            "aggravating_factors": "Pinching, gripping, twisting jar lids.",
            "easing_factors": "Heat, topical NSAID gel, thumb immobilization.",
            "radiation": "Thenar eminence.",
            "red_flags": "Prominent square deformity at 1st CMC joint.",
            "social_history": "Retired seamstress.",
            "past_medical_history": "General Osteoarthritis.",
            "diff_dx": "First Carpometacarpal (CMC) Joint Osteoarthritis"
        },
        "Case 4": {
            "name": "Abigail", "region_label": "Wrist and Hand", "forthcomingness": 1,
            "demeanor": "Points to smooth cyst bump on back of wrist.",
            "chief_complaint": "Visible bump on back of wrist causing dull ache when pressing up off floor.",
            "history_present_illness": "Noticed small fluid-filled lump 2 months ago that fluctuates in size.",
            "location_pain": "Dorsal scapholunate joint area of wrist.",
            "onset_pain": "Gradual onset over 8 weeks.",
            "type_pain": "Dull localized pressure ache.",
            "aggravating_factors": "End-range wrist extension (e.g. push-ups).",
            "easing_factors": "Rest, neutral wrist positioning.",
            "radiation": "None.",
            "red_flags": "Transilluminates with light, soft non-tender mobile mass.",
            "social_history": "Yoga instructor.",
            "past_medical_history": "None.",
            "diff_dx": "Dorsal Wrist Ganglion Cyst"
        },
        "Case 5": {
            "name": "Brian", "region_label": "Wrist and Hand", "forthcomingness": 1,
            "demeanor": "Holding wrist splinted, tender anatomical snuffbox.",
            "chief_complaint": "Pain at thumb base after falling onto outstretched hand playing basketball.",
            "history_present_illness": "Fell 10 days ago; thought it was a simple sprain, but pain persists.",
            "location_pain": "Anatomical snuffbox / Scaphoid bone.",
            "onset_pain": "Acute traumatic onset 10 days ago.",
            "type_pain": "Deep dull ache, point tenderness.",
            "aggravating_factors": "Wrist extension, radial deviation, axial loading of thumb.",
            "easing_factors": "Immobilization.",
            "radiation": "Radial side of wrist.",
            "red_flags": "Exquisite tenderness over scaphoid tubercle and snuffbox.",
            "social_history": "Student athlete.",
            "past_medical_history": "None.",
            "diff_dx": "Scaphoid Fracture"
        },
        "Case 6": {
            "name": "Catherine", "region_label": "Wrist and Hand", "forthcomingness": 1,
            "demeanor": "Finger flexed into palm, uses other hand to snap it straight.",
            "chief_complaint": "Ring finger gets stuck in palm and pops open with sharp pain.",
            "history_present_illness": "Finger catching began 3 weeks ago, worse in early mornings.",
            "location_pain": "A1 pulley area at distal palmar crease.",
            "onset_pain": "Subacute onset 21 days ago.",
            "type_pain": "Sharp catching pain and palpable nodule click.",
            "aggravating_factors": "Full fist closure, gripping steering wheel.",
            "easing_factors": "Manually extending finger using opposite hand.",
            "radiation": "Up flexor tendon into digit.",
            "red_flags": "Palpable tender nodule at A1 pulley.",
            "social_history": "Gardener, heavy pruning sheath user.",
            "past_medical_history": "Rheumatoid Arthritis.",
            "diff_dx": "Stenosing Tenosynovitis (Trigger Finger)"
        }
    },
    "Hip": {
        "Case 1": {
            "name": "Robert", "region_label": "Hip", "forthcomingness": 1,
            "demeanor": "Walking with slight limp, rubs anterior groin when sitting.",
            "chief_complaint": "Deep groin pinching pain when getting out of car or squatting.",
            "history_present_illness": "Deep groin stiffness developed over 4 months.",
            "location_pain": "Anterior groin and lateral hip ('C-sign').",
            "onset_pain": "Insidious onset.",
            "type_pain": "Deep pinching ache.",
            "aggravating_factors": "Deep hip flexion, prolonged sitting, twisting on planted foot.",
            "easing_factors": "Walking on flat ground, NSAIDs.",
            "radiation": "Anterior thigh down toward knee.",
            "red_flags": "Denies night pain, unexplained weight loss, or fever.",
            "social_history": "Former recreational soccer player.",
            "past_medical_history": "None.",
            "diff_dx": "Femoroacetabular Impingement (FAI) / Labral Tear"
        },
        "Case 2": {
            "name": "Deborah", "region_label": "Hip", "forthcomingness": 1,
            "demeanor": "Massaging lateral outer thigh, unable to sleep on affected side.",
            "chief_complaint": "Sharp pain on outside of hip when lying down or climbing stairs.",
            "history_present_illness": "Pain started 2 months ago after increasing walking program.",
            "location_pain": "Greater trochanter lateral hip extending to lateral IT band.",
            "onset_pain": "Gradual onset over 8 weeks.",
            "type_pain": "Sharp when lying directly on side, dull ache after activity.",
            "aggravating_factors": "Direct pressure lying on side, stair climbing, single-leg stance.",
            "easing_factors": "Sleeping with pillow between knees, rest.",
            "radiation": "Lateral thigh down to lateral knee.",
            "red_flags": "No true intra-articular groin pinching or hip joint locking.",
            "social_history": "Active retiree, power walker.",
            "past_medical_history": "None.",
            "diff_dx": "Greater Trochanteric Pain Syndrome (Gluteal Tendinopathy / Bursitis)"
        },
        "Case 3": {
            "name": "Ethan", "region_label": "Hip", "forthcomingness": 1,
            "demeanor": "Elderly male using walking cane, stiff gait.",
            "chief_complaint": "Severe hip stiffness in morning making putting on socks difficult.",
            "history_present_illness": "Progressive groin stiffness and hip pain over past 3 years.",
            "location_pain": "Deep anterior groin and buttocks.",
            "onset_pain": "Chronic insidious progression.",
            "type_pain": "Grinding dull ache, severe start-up stiffness.",
            "aggravating_factors": "Weight-bearing, cold damp weather, prolonged walking.",
            "easing_factors": "Warm shower, gentle movement after start-up, analgesics.",
            "radiation": "Anterior thigh to anterior medial knee.",
            "red_flags": "Capsular pattern restriction (Internal Rotation markedly reduced).",
            "social_history": "Retired farmer.",
            "past_medical_history": "Primary Osteoarthritis.",
            "diff_dx": "Hip Joint Osteoarthritis"
        },
        "Case 4": {
            "name": "Faith", "region_label": "Hip", "forthcomingness": 1,
            "demeanor": "Young long-distance runner, pointing to ischial tuberosity.",
            "chief_complaint": "Deep buttock pain when sitting on hard chairs and accelerating during runs.",
            "history_present_illness": "Ache developed 5 weeks ago during marathon training expansion.",
            "location_pain": "Ischial tuberosity and proximal hamstring origin.",
            "onset_pain": "Gradual onset over 5 weeks.",
            "type_pain": "Deep aching strain, sharp with high-speed running.",
            "aggravating_factors": "Prolonged sitting on firm surfaces, deep forward lunges, hill running.",
            "easing_factors": "Standing, sitting on donut cushion, ice.",
            "radiation": "Posterior upper thigh.",
            "red_flags": "Denies radicular leg tingling or foot numbness.",
            "social_history": "Marathon runner.",
            "past_medical_history": "None.",
            "diff_dx": "Proximal Hamstring Tendinopathy"
        },
        "Case 5": {
            "name": "Gavin", "region_label": "Hip", "forthcomingness": 1,
            "demeanor": "Hand placed deep in anterior groin flexor fold.",
            "chief_complaint": "Snapping sensation in front of hip when swinging leg back and forth.",
            "history_present_illness": "Audible pop and catch in groin during dance practice for 1 month.",
            "location_pain": "Anterior iliopsoas tendon area.",
            "onset_pain": "Insidious onset 4 weeks ago.",
            "type_pain": "Painless or mildly aching audible snap.",
            "aggravating_factors": "Moving hip from flexed-abducted position into extension.",
            "easing_factors": "Rest, gentle hip flexor stretches.",
            "radiation": "None.",
            "red_flags": "No true joint locking or giving way.",
            "social_history": "Ballet dancer.",
            "past_medical_history": "None.",
            "diff_dx": "Internal Coxa Saltans (Iliopsoas Snapping Hip)"
        },
        "Case 6": {
            "name": "Heather", "region_label": "Hip", "forthcomingness": 1,
            "demeanor": "Antalgic gait, non-weight bearing on left side with crutches.",
            "chief_complaint": "Severe deep groin pain on weight-bearing after high-energy fall.",
            "history_present_illness": "Slipped on ice 2 days ago, landed hard on lateral hip.",
            "location_pain": "Deep anterior groin and trochanteric region.",
            "onset_pain": "Acute traumatic onset 48 hours ago.",
            "type_pain": "Severe sharp pain with weight-bearing.",
            "aggravating_factors": "Any attempt to bear weight or rotate hip passively.",
            "easing_factors": "Complete rest, supine positioning.",
            "radiation": "Thigh and groin.",
            "red_flags": "Inability to bear weight (>4 steps), limb shortened and externally rotated.",
            "social_history": "Retired administrative assistant.",
            "past_medical_history": "Osteopenia.",
            "diff_dx": "Femoral Neck Stress Fracture / Femoral Fracture"
        }
    },
    "Knee": {
        "Case 1": {
            "name": "Ian", "region_label": "Knee", "forthcomingness": 1,
            "demeanor": "Holding knee flexed, visible anterior joint swelling.",
            "chief_complaint": "Knee popped and swelled immediately after pivoting in basketball.",
            "history_present_illness": "Pivoted on planted foot 24 hours ago; felt pop and knee gave way.",
            "location_pain": "Deep knee joint, general effusion.",
            "onset_pain": "Acute traumatic onset 1 day ago.",
            "type_pain": "Instability, deep pressure, sharp with twisting.",
            "aggravating_factors": "Attempted weight-bearing, twisting, full knee extension.",
            "easing_factors": "Rest, ice, compression, elevation (RICE), crutches.",
            "radiation": "None.",
            "red_flags": "Immediate hemarthrosis effusion (<2 hours), sensation of giving way.",
            "social_history": "Basketball player.",
            "past_medical_history": "None.",
            "diff_dx": "Anterior Cruciate Ligament (ACL) Tear"
        },
        "Case 2": {
            "name": "Jessica", "region_label": "Knee", "forthcomingness": 1,
            "demeanor": "Pointing to medial joint line, walks cautiously.",
            "chief_complaint": "Joint line pain and catching sensation when squatting.",
            "history_present_illness": "Knee twisted while gardening 3 weeks ago; persistent clicking inside knee.",
            "location_pain": "Medial joint line.",
            "onset_pain": "Subacute onset 21 days ago.",
            "type_pain": "Sharp catching pain, intermittent delayed swelling.",
            "aggravating_factors": "Deep squatting, twisting on flexed knee, duck walking.",
            "easing_factors": "Rest, keeping knee straight.",
            "radiation": "Posteromedial knee.",
            "red_flags": "Intermittent mechanical locking during terminal extension.",
            "social_history": "Landscape designer.",
            "past_medical_history": "None.",
            "diff_dx": "Medial Meniscal Tear"
        },
        "Case 3": {
            "name": "Klaus", "region_label": "Knee", "forthcomingness": 1,
            "demeanor": "Cupping hand over patella while walking up stairs.",
            "chief_complaint": "Aching pain behind kneecap when walking down stairs or sitting long.",
            "history_present_illness": "Vague knee ache worsened over 2 months after increasing running mileage.",
            "location_pain": "Retro-patellar and peri-patellar anterior knee.",
            "onset_pain": "Insidious onset over 8 weeks.",
            "type_pain": "Dull grinding ache, 'movie-theater sign' after long sitting.",
            "aggravating_factors": "Descending stairs, squatting, prolonged sitting with knees flexed.",
            "easing_factors": "Extending knee straight, patellar taping, ice.",
            "radiation": "Anterior knee.",
            "red_flags": "Crepitus present, no joint effusion.",
            "social_history": "Recreational runner.",
            "past_medical_history": "None.",
            "diff_dx": "Patellofemoral Pain Syndrome (PFPS)"
        },
        "Case 4": {
            "name": "Leah", "region_label": "Knee", "forthcomingness": 1,
            "demeanor": "Pointing to tendon right below kneecap.",
            "chief_complaint": "Sharp pain below kneecap during jumping and sprinting drills.",
            "history_present_illness": "Started 1 month ago during volleyball preseason training.",
            "location_pain": "Inferior pole of patella and patellar tendon.",
            "onset_pain": "Gradual onset over 4 weeks.",
            "type_pain": "Sharp focal pain with impact, stiff after rest.",
            "aggravating_factors": "Jumping, landing, heavy squats, quick decelerations.",
            "easing_factors": "Heat before activity, patellar tendon strap, rest.",
            "radiation": "Patellar tendon body to tibial tuberosity.",
            "red_flags": "No defect in tendon continuity.",
            "social_history": "Volleyball player.",
            "past_medical_history": "None.",
            "diff_dx": "Patellar Tendinopathy (Jumper's Knee)"
        },
        "Case 5": {
            "name": "Michael", "region_label": "Knee", "forthcomingness": 1,
            "demeanor": "Massaging medial shin right below knee joint line.",
            "chief_complaint": "Pain inside upper shin below joint line when rising from chair.",
            "history_present_illness": "Developing over 6 weeks in obese patient with medial knee arthritis.",
            "location_pain": "Anteromedial proximal tibia (pes anserinus region).",
            "onset_pain": "Gradual onset over 6 weeks.",
            "type_pain": "Localized tenderness and ache.",
            "aggravating_factors": "Stair climbing, crossing legs, sit-to-stand transfers.",
            "easing_factors": "Rest, local ice, sleeping with pillow between knees.",
            "radiation": "Medial upper calf.",
            "red_flags": "No intra-articular joint line tenderness.",
            "social_history": "Office clerk.",
            "past_medical_history": "Medial Knee Osteoarthritis, Obesity.",
            "diff_dx": "Pes Anserine Bursitis / Tendinopathy"
        },
        "Case 6": {
            "name": "Nina", "region_label": "Knee", "forthcomingness": 1,
            "demeanor": "Holding outer side of knee, rubbing lateral femoral condyle.",
            "chief_complaint": "Sharp stinging pain on outer knee at 30 degrees knee bend during runs.",
            "history_present_illness": "Pain appears consistently at 2-mile mark during running for past month.",
            "location_pain": "Lateral femoral epicondyle.",
            "onset_pain": "Insidious onset 4 weeks ago.",
            "type_pain": "Sharp repetitive friction sting.",
            "aggravating_factors": "Downhill running, repetitive flex-extension near 30 degrees.",
            "easing_factors": "Stopping running, straight leg walking, foam rolling tensor fasciae latae.",
            "radiation": "Up lateral thigh along IT band.",
            "red_flags": "Joint line non-tender, no joint effusion.",
            "social_history": "Distance runner.",
            "past_medical_history": "None.",
            "diff_dx": "Iliotibial Band Friction Syndrome (ITBS)"
        }
    },
    "Ankle and Foot": {
        "Case 1": {
            "name": "Oscar", "region_label": "Ankle and Foot", "forthcomingness": 1,
            "demeanor": "Limping, swollen lateral ankle visible with bruising.",
            "chief_complaint": "Twisted ankle landing on opponent's foot during basketball.",
            "history_present_illness": "Inversion trauma 2 days ago; immediate lateral ankle swelling.",
            "location_pain": "Anterior talofibular ligament (ATFL) and calcaneofibular ligament (CFL).",
            "onset_pain": "Acute traumatic onset 48 hours ago.",
            "type_pain": "Throbbing, sharp with weight-bearing or inversion.",
            "aggravating_factors": "Weight-bearing, passive plantarflexion and inversion.",
            "easing_factors": "RICE protocol, ankle brace, crutches.",
            "radiation": "Lateral foot dorsum.",
            "red_flags": "Able to walk 4 steps (Ottawa Ankle Rules negative for fracture).",
            "social_history": "Rec basketball player.",
            "past_medical_history": "None.",
            "diff_dx": "Lateral Ankle Sprain (Grade II ATFL/CFL)"
        },
        "Case 2": {
            "name": "Paula", "region_label": "Ankle and Foot", "forthcomingness": 1,
            "demeanor": "Grit teeth taking first few steps in clinic room.",
            "chief_complaint": "Excruciating stabbing heel pain during first steps out of bed in morning.",
            "history_present_illness": "Heel pain worsening over 3 months; eases slightly after walking then aches.",
            "location_pain": "Medial tubercle of calcaneus on plantar aspect.",
            "onset_pain": "Insidious onset 12 weeks ago.",
            "type_pain": "Sharp stabbing during first steps, dull throb after rest.",
            "aggravating_factors": "First morning steps, standing after sitting, bare foot walking on tile.",
            "easing_factors": "Calf stretching, supportive footwear with arch support, ice bottle rolling.",
            "radiation": "Along plantar fascia arch toward toes.",
            "red_flags": "No tingling in heel pad or sole.",
            "social_history": "Retail sales worker standing 8 hours/day.",
            "past_medical_history": "High BMI.",
            "diff_dx": "Plantar Fasciitis"
        },
        "Case 3": {
            "name": "Quinn", "region_label": "Ankle and Foot", "forthcomingness": 1,
            "demeanor": "Rubbing posterior Achilles heel region.",
            "chief_complaint": "Stiffness and burning pain behind ankle during morning walk.",
            "history_present_illness": "Developing over 6 weeks after starting plyometric jump workouts.",
            "location_pain": "Mid-portion Achilles tendon (2-6cm proximal to insertion).",
            "onset_pain": "Gradual onset over 6 weeks.",
            "type_pain": "Stiff ache, sharp with push-off phase of gait.",
            "aggravating_factors": "Pushing off, hill climbing, running, calf raises.",
            "easing_factors": "Heel lift inserts, eccentric calf loading, rest.",
            "radiation": "Up posterior lower calf.",
            "red_flags": "Tendon continuity intact, Thompson test negative.",
            "social_history": "Cross-training athlete.",
            "past_medical_history": "Fluoroquinolone antibiotic use 3 months ago.",
            "diff_dx": "Mid-Portion Achilles Tendinopathy"
        },
        "Case 4": {
            "name": "Rosa", "region_label": "Ankle and Foot", "forthcomingness": 1,
            "demeanor": "Pointing to 3rd and 4th toe interspace, taking off tight shoe.",
            "chief_complaint": "Sensation of standing on a crumpled sock or pebble in shoe.",
            "history_present_illness": "Burning ball-of-foot pain for 2 months, worse in narrow dress shoes.",
            "location_pain": "3rd intermetatarsal space on plantar aspect.",
            "onset_pain": "Gradual onset over 8 weeks.",
            "type_pain": "Burning, tingling, sharp electric shoots into 3rd and 4th toes.",
            "aggravating_factors": "High heels, narrow toe box shoes, running on ball of foot.",
            "easing_factors": "Removing shoes, massaging metatarsal heads, wide footwear.",
            "radiation": "Toes 3 and 4 digital webs.",
            "red_flags": "Positive Mulder's click sign with squeeze test.",
            "social_history": "Corporate executive.",
            "past_medical_history": "None.",
            "diff_dx": "Morton's Neuroma (Interdigital Neuroma)"
        },
        "Case 5": {
            "name": "Samuel", "region_label": "Ankle and Foot", "forthcomingness": 1,
            "demeanor": "Holding high-ankle anterior area, walking with stiff leg.",
            "chief_complaint": "Ankle pain above joint line after severe external rotation injury in football.",
            "history_present_illness": "Tackled 4 days ago with foot planted and rotated outward.",
            "location_pain": "Anterior inferior tibiofibular syndesmosis and interosseous membrane.",
            "onset_pain": "Acute traumatic onset 4 days ago.",
            "type_pain": "Severe sharp pain during weight-bearing push-off.",
            "aggravating_factors": "Passive dorsiflexion and external rotation of foot, squeezing upper calf.",
            "easing_factors": "Rigid boot immobilization, non-weight bearing.",
            "radiation": "Up distal anterior shin.",
            "red_flags": "Inability to perform single-leg hop, prolonged recovery expected.",
            "social_history": "Football player.",
            "past_medical_history": "None.",
            "diff_dx": "High Ankle Sprain (Syndesmotic Ligament Complex Injury)"
        },
        "Case 6": {
            "name": "Tina", "region_label": "Ankle and Foot", "forthcomingness": 1,
            "demeanor": "Massaging medial inner ankle behind medial malleolus.",
            "chief_complaint": "Inner ankle pain and flattening of foot arch over past 4 months.",
            "history_present_illness": "Noticeable loss of foot arch height and pain behind inner ankle bone.",
            "location_pain": "Posterior inferior medial malleolus and navicular tuberosity.",
            "onset_pain": "Gradual progression over 16 weeks.",
            "type_pain": "Aching weariness, sharp with heel raise.",
            "aggravating_factors": "Prolonged walking, standing, single-leg heel raise attempt.",
            "easing_factors": "Custom rigid orthotics, supportive boots.",
            "radiation": "Medial plantar arch.",
            "red_flags": "'Too many toes' sign positive from behind, inability to perform single heel raise.",
            "social_history": "Store clerk.",
            "past_medical_history": "Hypertension, Adult Acquired Flatfoot.",
            "diff_dx": "Posterior Tibial Tendon Dysfunction (PTTD)"
        }
    }
}

# ==============================================================================
# 4. PERSISTENT DISK & GITHUB STORAGE FUNCTIONS
# ==============================================================================
def save_cases_to_disk(case_data):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(case_data, f, indent=4)
    except Exception:
        pass

    if GITHUB_TOKEN and GITHUB_REPO:
        try:
            clean_repo = GITHUB_REPO.replace("https://github.com/", "").replace(".git", "").strip("/")
            url = f"https://api.github.com/repos/{clean_repo}/contents/{DATA_FILE}"
            
            headers = {
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json",
                "X-GitHub-Api-Version": "2022-11-28"
            }

            get_res = requests.get(url, headers=headers)
            sha = ""
            if get_res.status_code == 200:
                sha = get_res.json().get("sha", "")

            json_bytes = json.dumps(case_data, indent=4).encode("utf-8")
            base64_content = base64.b64encode(json_bytes).decode("utf-8")

            payload = {
                "message": "Admin update: Sync cases.json via Streamlit UI",
                "content": base64_content
            }
            if sha:
                payload["sha"] = sha

            put_res = requests.put(url, headers=headers, json=payload)
            if put_res.status_code in [200, 201]:
                st.toast("Saved permanently to GitHub repo!", icon="✅")
            elif put_res.status_code == 404:
                st.error("GitHub Sync Failed (404): Check GITHUB_REPO format (username/repo) and GITHUB_TOKEN write access.")
            else:
                st.error(f"GitHub Sync Failed ({put_res.status_code}): {put_res.json().get('message')}")
        except Exception as err:
            st.error(f"GitHub Sync Error: {err}")

def load_cases_from_disk():
    loaded_data = None
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                loaded_data = json.load(f)
        except Exception:
            loaded_data = None

    if not loaded_data or not isinstance(loaded_data, dict):
        loaded_data = DEFAULT_CASE_LIBRARY
        save_cases_to_disk(DEFAULT_CASE_LIBRARY)

    for region, cases in list(loaded_data.items()):
        region_template = get_default_objective_template_for_region(region)
        if isinstance(cases, dict):
            for case_key, cdata in cases.items():
                if isinstance(cdata, dict):
                    cdata.setdefault("forthcomingness", 1)
                    if "objective_data" not in cdata:
                        cdata["objective_data"] = region_template

    return loaded_data

# ==============================================================================
# 5. SESSION STATE INITIALIZATION
# ==============================================================================
if "ccid" not in st.session_state:
    st.session_state.ccid = None
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "case_library" not in st.session_state:
    st.session_state.case_library = load_cases_from_disk()

if "encounter_phase" not in st.session_state:
    st.session_state.encounter_phase = 1
if "subjective_messages" not in st.session_state:
    st.session_state.subjective_messages = []
if "objective_tests" not in st.session_state:
    st.session_state.objective_tests = []
if "initial_differentials" not in st.session_state:
    st.session_state.initial_differentials = ["", "", ""]

if "tx_final_dx" not in st.session_state:
    st.session_state.tx_final_dx = ""
if "tx_education" not in st.session_state:
    st.session_state.tx_education = ""
if "tx_pain_mgmt" not in st.session_state:
    st.session_state.tx_pain_mgmt = ""
if "tx_mobility" not in st.session_state:
    st.session_state.tx_mobility = ""
if "tx_strength" not in st.session_state:
    st.session_state.tx_strength = ""

# ==============================================================================
# 6. HELPER FUNCTIONS
# ==============================================================================
def get_forthcomingness_instruction(level):
    level = int(level)
    if level == 1:
        return "COMMUNICATION STYLE: Very short, reluctant answers (1-2 sentences)."
    elif level == 2:
        return "COMMUNICATION STYLE: Answer exact questions asked without extra detail."
    elif level == 3:
        return "COMMUNICATION STYLE: Answer questions naturally as a realistic patient."
    elif level == 4:
        return "COMMUNICATION STYLE: Open and verbose; share related details comfortably."
    else:
        return "COMMUNICATION STYLE: Extremely open; freely share extensive personal details."

def build_patient_instructions(c):
    return (
        f"You are a standardized patient named {c['name']} in a medical simulation.\n"
        f"PATIENT DEMEANOR: {c['demeanor']}\n"
        f"{get_forthcomingness_instruction(c.get('forthcomingness', 1))}\n\n"
        f"CHIEF COMPLAINT: {c['chief_complaint']}\n"
        f"HPI: {c['history_present_illness']}\n"
        f"LOCATION: {c['location_pain']}\n"
        f"ONSET: {c['onset_pain']}\n"
        f"TYPE: {c['type_pain']}\n"
        f"AGGRAVATING: {c['aggravating_factors']}\n"
        f"EASING: {c['easing_factors']}\n"
        f"RADIATION: {c['radiation']}\n"
        f"RED FLAGS: {c['red_flags']}\n"
        f"SOCIAL: {c['social_history']}\n"
        f"PMH: {c['past_medical_history']}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Stay in character as {c['name']}.\n"
        f"- Never state your diagnosis or medical jargon directly."
    )

def match_objective_query(query_text, case_obj_data):
    q = query_text.strip().lower()
    
    if any(k in q for k in ["strength", "resisted", "mmt", "manual muscle", "myotome"]):
        return "Strength / Resisted Isometrics", case_obj_data.get("Strength / Resisted Isometrics", "Normal strength.")
    elif any(k in q for k in ["palpate", "palpation", "touch", "tenderness", "point"]):
        return "Palpation", case_obj_data.get("Palpation", "No specific point tenderness noted.")
    elif any(k in q for k in ["special test", "provocative", "test", "spurling", "distraction", "hawkins", "faddir", "faber", "mcmurray", "slr", "lachman", "cozen", "finkelstein"]):
        return "Special Tests", case_obj_data.get("Special Tests", "Special tests negative.")
    elif any(k in q for k in ["prom", "passive"]):
        return "Passive Range of Motion (PROM)", case_obj_data.get("Passive Range of Motion (PROM)", "Full PROM.")
    elif any(k in q for k in ["arom", "active range", "active motion", "flexion", "extension", "abduction", "rotation", "side bend"]):
        return "Active Range of Motion (AROM)", case_obj_data.get("Active Range of Motion (AROM)", "Full AROM.")
    elif any(k in q for k in ["observe", "observation", "posture", "gait", "look", "alignment"]):
        return "Observation", case_obj_data.get("Observation", "No gross abnormality.")
    elif any(k in q for k in ["functional", "squat", "reach", "hop", "balance", "desk", "step"]):
        return "Functional Testing", case_obj_data.get("Functional Testing", "Functional movements intact.")
    else:
        for cat, content in case_obj_data.items():
            if q in content.lower() or any(term in content.lower() for term in q.split()):
                return f"{cat} ({query_text.strip()})", content
                
        return query_text.strip(), f"Evaluation of '{query_text.strip()}': No localized or specific pathological findings reproduced."

# ==============================================================================
# 7. STAGE 1: CCID SECURITY GATE
# ==============================================================================
if not st.session_state.ccid:
    st.title("🏥 MSK Clinical Assessment Simulator")
    st.write("Enter your CCID badge number to start your clinical simulation.")
    ccid_input = st.text_input("Institutional CCID Number:", placeholder="e.g., MGOERTZ99")
    if st.button("Access Clinical Portal", type="primary"):
        if ccid_input.strip():
            st.session_state.ccid = ccid_input.strip()
            st.rerun()
        else:
            st.warning("A valid CCID sequence is mandatory.")
    st.stop()

# ==============================================================================
# 8. STAGE 2: NAVIGATION & SIDEBAR
# ==============================================================================
st.sidebar.title("🩺 Control Center")
st.sidebar.markdown(f"**Active User:** `{st.session_state.ccid}`")

nav_options = ["Student Portal"]
if st.session_state.is_admin:
    nav_options.append("Admin/Instructor Editor")

role = st.sidebar.radio("Navigation View:", nav_options)
st.sidebar.markdown("---")

if not st.session_state.is_admin:
    with st.sidebar.expander("🔑 Admin Access"):
        admin_pass = st.text_input("Enter Admin Password:", type="password")
        if st.button("Unlock Admin Mode"):
            if admin_pass == "admin":
                st.session_state.is_admin = True
                st.success("Admin access granted!")
                st.rerun()
            else:
                st.error("Incorrect password.")
else:
    st.sidebar.success("🔓 Admin Mode Active")
    if st.sidebar.button("Lock Admin Access"):
        st.session_state.is_admin = False
        st.rerun()

if st.sidebar.button("Terminate Session"):
    st.session_state.ccid = None
    st.session_state.is_admin = False
    st.session_state.encounter_phase = 1
    st.session_state.subjective_messages = []
    st.session_state.objective_tests = []
    st.session_state.initial_differentials = ["", "", ""]
    st.session_state.tx_final_dx = ""
    st.session_state.tx_education = ""
    st.session_state.tx_pain_mgmt = ""
    st.session_state.tx_mobility = ""
    st.session_state.tx_strength = ""
    st.rerun()

# ==============================================================================
# 9. STAGE 3: ADMIN CASE EDITOR
# ==============================================================================
if role == "Admin/Instructor Editor":
    st.title("🛠️ Admin Case Management Matrix")
    
    cat_col, case_col = st.columns(2)
    with cat_col:
        admin_categories = list(st.session_state.case_library.keys())
        selected_category = st.selectbox("1. Select Joint Domain:", admin_categories)
        
    admin_category_cases = st.session_state.case_library.get(selected_category, {})
    admin_case_keys = list(admin_category_cases.keys())

    with case_col:
        if admin_case_keys:
            selected_case_key = st.selectbox(
                "2. Select Patient Case:", 
                admin_case_keys,
                format_func=lambda k: f"{k} — Patient: {admin_category_cases[k].get('name', 'Unknown')}"
            )
        else:
            st.error(f"No cases found for category: {selected_category}")
            st.stop()
        
    case_data = admin_category_cases[selected_case_key]
    if "objective_data" not in case_data:
        case_data["objective_data"] = get_default_objective_template_for_region(selected_category)

    st.markdown("---")
    
    with st.form("admin_case_form"):
        st.subheader(f"Editing {selected_case_key}: Patient {case_data.get('name', '')} ({selected_category})")
        
        tab1, tab2 = st.tabs(["🗣️ Subjective Case Parameters", "📊 Granular Objective Matrix"])
        
        with tab1:
            e_forthcoming = st.slider("Patient Forthcomingness (1-5):", 1, 5, int(case_data.get("forthcomingness", 1)))
            col1, col2 = st.columns(2)
            with col1:
                e_name = st.text_input("Patient Name", value=case_data.get("name", ""))
                e_demeanor = st.text_input("Demeanor", value=case_data.get("demeanor", ""))
                e_chief = st.text_area("Chief Complaint", value=case_data.get("chief_complaint", ""))
                e_hpi = st.text_area("HPI", value=case_data.get("history_present_illness", ""))
                e_loc = st.text_input("Location", value=case_data.get("location_pain", ""))
                e_onset = st.text_input("Onset", value=case_data.get("onset_pain", ""))
                e_type = st.text_input("Type", value=case_data.get("type_pain", ""))
            with col2:
                e_agg = st.text_area("Aggravating Factors", value=case_data.get("aggravating_factors", ""))
                e_ease = st.text_area("Easing Factors", value=case_data.get("easing_factors", ""))
                e_rad = st.text_input("Radiation", value=case_data.get("radiation", ""))
                e_red = st.text_area("Red Flags", value=case_data.get("red_flags", ""))
                e_soc = st.text_area("Social History", value=case_data.get("social_history", ""))
                e_pmh = st.text_area("Past Medical History", value=case_data.get("past_medical_history", ""))
                e_diff = st.text_input("Master Diagnosis Key", value=case_data.get("diff_dx", ""))

        with tab2:
            st.markdown(f"### Edit Objective Physical Exam Findings ({selected_category})")
            st.caption("Customize movement breakdowns, specific anatomical structures, and special tests.")
            
            edited_objective_data = {}
            for cat in OBJECTIVE_CATEGORIES:
                current_val = case_data["objective_data"].get(cat, "")
                edited_objective_data[cat] = st.text_area(f"📌 {cat}", value=current_val, height=100)

        save_submitted = st.form_submit_button("Save Case Settings & Sync to GitHub", type="primary")
        
        if save_submitted:
            st.session_state.case_library[selected_category][selected_case_key].update({
                "name": e_name,
                "forthcomingness": e_forthcoming,
                "demeanor": e_demeanor,
                "chief_complaint": e_chief,
                "history_present_illness": e_hpi,
                "location_pain": e_loc,
                "onset_pain": e_onset,
                "type_pain": e_type,
                "aggravating_factors": e_agg,
                "easing_factors": e_ease,
                "radiation": e_rad,
                "red_flags": e_red,
                "social_history": e_soc,
                "past_medical_history": e_pmh,
                "diff_dx": e_diff,
                "objective_data": edited_objective_data
            })
            save_cases_to_disk(st.session_state.case_library)

# ==============================================================================
# 10. STAGE 4: STUDENT 3-PHASE CLINICAL SIMULATOR
# ==============================================================================
else:
    st.title("🎓 Interactive 3-Phase Clinical Assessment")
    
    col_cat, col_case = st.columns(2)
    with col_cat:
        available_categories = list(st.session_state.case_library.keys())
        student_category = st.selectbox("Select Joint Category:", available_categories)

    category_cases = st.session_state.case_library.get(student_category, {})
    case_keys = list(category_cases.keys())

    with col_case:
        if case_keys:
            student_case_key = st.selectbox(
                "Select Patient Case:", 
                case_keys,
                format_func=lambda k: f"{k} — Patient: {category_cases[k].get('name', 'Unknown')}"
            )
        else:
            st.error(f"No cases found for category: {student_category}")
            st.stop()
        
    active_case = category_cases[student_case_key]
    if "objective_data" not in active_case:
        active_case["objective_data"] = get_default_objective_template_for_region(student_category)

    unique_case_id = f"{student_category}_{student_case_key}"
    if "last_chosen_case_id" not in st.session_state or st.session_state.last_chosen_case_id != unique_case_id:
        st.session_state.subjective_messages = []
        st.session_state.objective_tests = []
        st.session_state.encounter_phase = 1
        st.session_state.initial_differentials = ["", "", ""]
        st.session_state.tx_final_dx = ""
        st.session_state.tx_education = ""
        st.session_state.tx_pain_mgmt = ""
        st.session_state.tx_mobility = ""
        st.session_state.tx_strength = ""
        st.session_state.last_chosen_case_id = unique_case_id

    st.info(f"📋 **Active Encounter:** {student_category} ({student_case_key}) — Patient: **{active_case.get('name', 'Unknown')}**")

    # ENCOUNTER PROGRESS BAR
    phase_names = {
        1: "Phase 1: Subjective History",
        2: "Phase 2: Objective Physical Exam",
        3: "Phase 3: Treatment & Management Plan",
        4: "Encounter Complete"
    }
    progress_val = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}[st.session_state.encounter_phase]
    st.progress(progress_val, text=f"**Current Status:** {phase_names[st.session_state.encounter_phase]}")
    st.markdown("---")

    # --------------------------------------------------------------------------
    # PHASE 1: SUBJECTIVE HISTORY
    # --------------------------------------------------------------------------
    if st.session_state.encounter_phase == 1:
        st.subheader("🗣️ Phase 1: Subjective History Taking")
        
        for msg in st.session_state.subjective_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask your patient a subjective history question..."):
            st.session_state.subjective_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                if not client:
                    st.error("GROQ_API_KEY missing from secrets.")
                else:
                    try:
                        system_instruction = build_patient_instructions(active_case)
                        completion = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": system_instruction},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.6,
                            max_tokens=300
                        )
                        ai_text = completion.choices[0].message.content
                        st.markdown(ai_text)
                        st.session_state.subjective_messages.append({"role": "assistant", "content": ai_text})
                    except Exception as e:
                        st.error(f"Groq API Error: {e}")

        st.markdown("---")
        
        @st.dialog("Submit Initial Differential Diagnoses")
        def open_phase1_dialog():
            st.write("Enter your top 3 differential diagnoses based on the subjective history to unlock Phase 2.")
            with st.form("phase1_diff_form"):
                dx1 = st.text_input("Primary Suspected Differential:", placeholder="e.g., Primary Pathology")
                dx2 = st.text_input("Secondary Differential:", placeholder="e.g., Secondary Suspect")
                dx3 = st.text_input("Tertiary Differential:", placeholder="e.g., Alternative Suspect")
                
                if st.form_submit_button("Submit & Proceed to Objective Exam", type="primary"):
                    if not dx1.strip() or not dx2.strip() or not dx3.strip():
                        st.error("Please fill in all 3 differential fields before proceeding.")
                    else:
                        st.session_state.initial_differentials = [dx1.strip(), dx2.strip(), dx3.strip()]
                        st.session_state.encounter_phase = 2
                        st.rerun()

        if st.button("➡️ Move on to Objective Exam", type="primary", use_container_width=True):
            if not st.session_state.subjective_messages:
                st.warning("Please ask at least one subjective history question before moving on.")
            else:
                open_phase1_dialog()

    # --------------------------------------------------------------------------
    # PHASE 2: OBJECTIVE PHYSICAL EXAM
    # --------------------------------------------------------------------------
    elif st.session_state.encounter_phase == 2:
        st.subheader("🔬 Phase 2: Objective Physical Examination")
        st.write("Type what physical exam procedures or evaluations you want to perform.")

        with st.expander("📌 Your Phase 1 Initial Differential Diagnoses"):
            for i, d in enumerate(st.session_state.initial_differentials, 1):
                st.markdown(f"**{i}.** {d}")

        st.markdown("### Request Physical Examination Procedures")
        st.caption(f"Perform tests relevant to the **{student_category}** (e.g., MMT/Strength, Palpation, Special Tests, AROM/PROM)")
        
        user_test_query = st.text_input("Enter physical exam evaluation / test to perform:", key="test_input_field", placeholder=f"e.g., {student_category} strength testing or special tests")

        if st.button("Execute Physical Examination Test", type="primary"):
            if not user_test_query.strip():
                st.warning("Please type a test or evaluation request first.")
            else:
                category_name, finding_text = match_objective_query(user_test_query, active_case.get("objective_data", {}))
                
                st.session_state.objective_tests.append({
                    "requested": user_test_query.strip(),
                    "category": category_name,
                    "findings": finding_text
                })
                st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Physical Exam Charting Record")
        
        if not st.session_state.objective_tests:
            st.info("No physical exam tests executed yet. Type an examination request above to evaluate.")
        else:
            chart_df = pd.DataFrame(st.session_state.objective_tests)
            chart_df = chart_df[["requested", "category", "findings"]]
            chart_df.columns = ["Student Requested Test", "Matched Category", "Specific Clinical Findings"]
            st.dataframe(chart_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        if st.button("➡️ Move on to Treatment Phase", type="primary", use_container_width=True):
            if not st.session_state.objective_tests:
                st.warning("Please perform at least one objective evaluation before proceeding.")
            else:
                st.session_state.encounter_phase = 3
                st.rerun()

    # --------------------------------------------------------------------------
    # PHASE 3: TREATMENT & MANAGEMENT PLAN
    # --------------------------------------------------------------------------
    elif st.session_state.encounter_phase >= 3:
        st.subheader("💊 Phase 3: Treatment & Management Plan")
        st.write("Synthesize your subjective and objective findings to formulate your final diagnosis and management strategy.")

        with st.expander("🔍 Review Prior Phase Findings"):
            st.markdown("**Phase 1 Differentials:** " + ", ".join(st.session_state.initial_differentials))
            st.markdown("**Phase 2 Objective Findings:**")
            for t in st.session_state.objective_tests:
                st.markdown(f"- **{t['requested']}** ({t['category']}): {t['findings']}")

        if st.session_state.encounter_phase == 3:
            with st.form("treatment_phase_form"):
                st.markdown("### 📝 Clinical Management Plan")
                
                f_dx = st.text_input("1. Final Diagnosis:", placeholder=f"e.g., Primary {student_category} Pathology")
                f_edu = st.text_area("2. Education:", placeholder="Patient reassurance, posture/ergonomic advice, prognosis...", height=100)
                f_pain = st.text_area("3. Pain Management:", placeholder="Heat/ice, activity modification, movement breaks...", height=100)
                f_mob = st.text_area("4. Mobility:", placeholder="Range of motion exercises, joint mobilizations, stretching...", height=100)
                f_str = st.text_area("5. Strength:", placeholder="Progressive resistance exercises, stabilizer strengthening...", height=100)

                if st.form_submit_button("Submit Complete Treatment Plan", type="primary"):
                    if not f_dx.strip() or not f_edu.strip() or not f_pain.strip() or not f_mob.strip() or not f_str.strip():
                        st.error("Please complete all 5 text fields before submitting.")
                    else:
                        st.session_state.tx_final_dx = f_dx.strip()
                        st.session_state.tx_education = f_edu.strip()
                        st.session_state.tx_pain_mgmt = f_pain.strip()
                        st.session_state.tx_mobility = f_mob.strip()
                        st.session_state.tx_strength = f_str.strip()
                        st.session_state.encounter_phase = 4
                        st.rerun()

        elif st.session_state.encounter_phase == 4:
            st.success("🎉 **Clinical Encounter Complete!**")
            st.markdown(f"**Final Diagnosis:** {st.session_state.tx_final_dx}")
            
            st.markdown("---")
            st.markdown("### Submitted Treatment Plan")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Education:**")
                st.info(st.session_state.tx_education)
                st.markdown("**Pain Management:**")
                st.info(st.session_state.tx_pain_mgmt)
            with col_b:
                st.markdown("**Mobility:**")
                st.info(st.session_state.tx_mobility)
                st.markdown("**Strength:**")
                st.info(st.session_state.tx_strength)

    # ==============================================================================
    # 11. STAGE 5: FULL 3-PHASE TRANSCRIPT EXPORT
    # ==============================================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Submission Records")
    if st.sidebar.button("Compile Full 3-Phase Transcript"):
        if not st.session_state.subjective_messages:
            st.sidebar.warning("No encounter data recorded.")
        else:
            export = f"==================================================\n"
            export += f"OFFICIAL MSK 3-PHASE EVALUATION TRANSCRIPT\n"
            export += f"==================================================\n"
            export += f"Student CCID: {st.session_state.ccid}\n"
            export += f"Joint Region: {student_category} ({student_case_key})\n"
            export += f"Patient Name: {active_case.get('name', 'Unknown')}\n"
            export += f"--------------------------------------------------\n\n"
            
            export += f"--- PHASE 1: SUBJECTIVE HISTORY ---\n"
            for line in st.session_state.subjective_messages:
                spk = "STUDENT" if line["role"] == "user" else "PATIENT"
                export += f"[{spk}]: {line['content']}\n"
            
            export += f"\nPHASE 1 INITIAL DIFFERENTIALS:\n"
            for idx, dx in enumerate(st.session_state.initial_differentials, 1):
                export += f"  {idx}. {dx}\n"
            
            export += f"\n--------------------------------------------------\n"
            export += f"--- PHASE 2: OBJECTIVE FINDINGS ---\n"
            if st.session_state.objective_tests:
                for item in st.session_state.objective_tests:
                    export += f"Requested: {item['requested']}\nMatched Category: {item['category']}\nFindings: {item['findings']}\n\n"
            else:
                export += f"[No objective evaluations recorded]\n\n"

            export += f"--------------------------------------------------\n"
            export += f"--- PHASE 3: TREATMENT & MANAGEMENT PLAN ---\n\n"
            export += f"FINAL DIAGNOSIS:\n{st.session_state.tx_final_dx}\n\n"
            export += f"EDUCATION:\n{st.session_state.tx_education}\n\n"
            export += f"PAIN MANAGEMENT:\n{st.session_state.tx_pain_mgmt}\n\n"
            export += f"MOBILITY:\n{st.session_state.tx_mobility}\n\n"
            export += f"STRENGTH:\n{st.session_state.tx_strength}\n"
            export += f"==================================================\n"
                
            st.sidebar.download_button(
                label="📥 Download Full Transcript (.txt)",
                data=export,
                file_name=f"MSK_FullEncounter_{st.session_state.ccid}_{student_case_key}.txt",
                mime="text/plain"
            )