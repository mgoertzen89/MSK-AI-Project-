import streamlit as st
import pandas as pd
from groq import Groq
import json
import os

# ==============================================================================
# 1. PAGE CONFIGURATION & DARK MODE STYLING
# ==============================================================================
st.set_page_config(
    page_title="MSK Clinical Simulator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme & UI Refinements
st.markdown("""
<style>
    /* Dark Theme Backgrounds */
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30363D;
    }

    /* Cards & Containers */
    div[data-testid="stExpander"], div[data-testid="stForm"] {
        background-color: #1E222A !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    /* Status & Info Boxes */
    div.stAlert {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        color: #58A6FF !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button[kind="primary"] {
        background-color: #238636 !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #2EA043 !important;
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.4);
    }

    /* Chat Bubbles */
    div[data-testid="stChatMessage"] {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        margin-bottom: 0.5rem !important;
    }

    /* Input Fields */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div {
        background-color: #0D1117 !important;
        color: #F0F6FC !important;
        border: 1px solid #30363D !important;
        border-radius: 6px !important;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #58A6FF !important;
        box-shadow: 0 0 5px rgba(88, 166, 255, 0.3) !important;
    }
    
    /* Headers & Text */
    h1, h2, h3 {
        color: #F0F6FC !important;
        font-weight: 600 !important;
    }
    
    /* Divider */
    hr {
        border-color: #30363D !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. INITIALIZATION & API CONFIGURATION
# ==============================================================================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "").strip()
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL_NAME = "llama-3.1-8b-instant"
DATA_FILE = "cases.json"

ADMIN_PASSWORD = "mgoertze"  # Updated Admin Password

OBJECTIVE_CATEGORIES = [
    "Observation",
    "Active Range of Motion (AROM)",
    "Passive Range of Motion (PROM)",
    "Strength / Resisted Isometrics",
    "Functional Testing",
    "Palpation",
    "Special Tests"
]

# ==============================================================================
# 3. REGION-SPECIFIC DEFAULT OBJECTIVE FINDINGS TEMPLATES
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
# 4. DEFAULT CASE LIBRARY
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
        }
    }
}

# ==============================================================================
# 5. LOCAL DISK STORAGE FUNCTIONS
# ==============================================================================
def save_cases_to_disk(case_data):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(case_data, f, indent=4)
        st.toast("Case settings updated locally!", icon="✅")
    except Exception as e:
        st.error(f"Error saving cases to disk: {e}")

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
# 6. SESSION STATE INITIALIZATION
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

# Phase 3 Structured Inputs
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
# 7. HELPER FUNCTIONS
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
# 8. LOGIN GATEWAY
# ==============================================================================
if not st.session_state.ccid:
    st.markdown("<h1 style='text-align: center; color: #58A6FF;'>🩺 MSK Clinical Assessment Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8B949E;'>Standardized Patient Interaction & Case Assessment Suite</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown("### 🔐 User Authentication")
            ccid_input = st.text_input("Enter Institutional CCID:", placeholder="e.g., MGOERTZE99")
            if st.button("Access Clinical Portal", type="primary", use_container_width=True):
                if ccid_input.strip():
                    st.session_state.ccid = ccid_input.strip()
                    st.rerun()
                else:
                    st.warning("A valid CCID sequence is mandatory.")
    st.stop()

# ==============================================================================
# 9. SIDEBAR CONTROL PANEL
# ==============================================================================
st.sidebar.markdown("### 🩺 Control Center")
st.sidebar.markdown(f"**Active User:** `{st.session_state.ccid}`")

nav_options = ["Student Portal"]
if st.session_state.is_admin:
    nav_options.append("Admin/Instructor Editor")

role = st.sidebar.radio("Navigation Mode:", nav_options)
st.sidebar.markdown("---")

if not st.session_state.is_admin:
    with st.sidebar.expander("🔑 Admin Authorization"):
        admin_pass = st.text_input("Admin Password:", type="password")
        if st.button("Unlock Admin Mode", use_container_width=True):
            if admin_pass == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.success("Admin access granted!")
                st.rerun()
            else:
                st.error("Invalid password.")
else:
    st.sidebar.success("🔓 Admin Mode Unlocked")
    if st.sidebar.button("Lock Admin Access", use_container_width=True):
        st.session_state.is_admin = False
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("Terminate Session", use_container_width=True):
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
# 10. ADMIN EDITOR VIEW
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
        
        tab1, tab2 = st.tabs(["🗣️ Subjective Case Parameters", "📊 Objective Matrix"])
        
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
            st.markdown(f"### Edit Physical Exam Findings ({selected_category})")
            edited_objective_data = {}
            for cat in OBJECTIVE_CATEGORIES:
                current_val = case_data["objective_data"].get(cat, "")
                edited_objective_data[cat] = st.text_area(f"📌 {cat}", value=current_val, height=100)

        save_submitted = st.form_submit_button("Save Case Parameters", type="primary")
        
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
# 11. STUDENT ENCOUNTER PORTAL
# ==============================================================================
else:
    st.title("🎓 Interactive Clinical Assessment")
    
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
    st.progress(progress_val, text=f"**Status:** {phase_names[st.session_state.encounter_phase]}")
    st.markdown("---")

    # --------------------------------------------------------------------------
    # PHASE 1: SUBJECTIVE HISTORY
    # --------------------------------------------------------------------------
    if st.session_state.encounter_phase == 1:
        st.subheader("🗣️ Phase 1: Subjective History Taking")
        
        for msg in st.session_state.subjective_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask your patient a question..."):
            st.session_state.subjective_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                if not client:
                    st.error("GROQ_API_KEY is missing from Streamlit secrets.")
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
                dx1 = st.text_input("Primary Suspected Differential:")
                dx2 = st.text_input("Secondary Differential:")
                dx3 = st.text_input("Tertiary Differential:")
                
                if st.form_submit_button("Submit & Proceed to Objective Exam", type="primary"):
                    if not dx1.strip() or not dx2.strip() or not dx3.strip():
                        st.error("Please complete all 3 differential fields.")
                    else:
                        st.session_state.initial_differentials = [dx1.strip(), dx2.strip(), dx3.strip()]
                        st.session_state.encounter_phase = 2
                        st.rerun()

        if st.button("➡️ Move to Objective Exam", type="primary", use_container_width=True):
            if not st.session_state.subjective_messages:
                st.warning("Please conduct history taking before proceeding.")
            else:
                open_phase1_dialog()

    # --------------------------------------------------------------------------
    # PHASE 2: OBJECTIVE PHYSICAL EXAM
    # --------------------------------------------------------------------------
    elif st.session_state.encounter_phase == 2:
        st.subheader("🔬 Phase 2: Objective Physical Examination")

        with st.expander("📌 Phase 1 Initial Differentials", expanded=False):
            for i, d in enumerate(st.session_state.initial_differentials, 1):
                st.markdown(f"**{i}.** {d}")

        st.markdown("### Physical Examination Requests")
        user_test_query = st.text_input(
            "Enter evaluation or test to perform:", 
            key="test_input_field", 
            placeholder=f"e.g., {student_category} special tests, AROM, or palpation"
        )

        if st.button("Execute Exam Procedure", type="primary"):
            if not user_test_query.strip():
                st.warning("Please type a valid evaluation request.")
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
            st.info("No physical exam tests recorded yet.")
        else:
            chart_df = pd.DataFrame(st.session_state.objective_tests)
            chart_df = chart_df[["requested", "category", "findings"]]
            chart_df.columns = ["Requested Test", "Category", "Clinical Findings"]
            st.dataframe(chart_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        if st.button("➡️ Proceed to Treatment Phase", type="primary", use_container_width=True):
            if not st.session_state.objective_tests:
                st.warning("Perform at least one objective evaluation first.")
            else:
                st.session_state.encounter_phase = 3
                st.rerun()

    # --------------------------------------------------------------------------
    # PHASE 3: TREATMENT & MANAGEMENT PLAN
    # --------------------------------------------------------------------------
    elif st.session_state.encounter_phase >= 3:
        st.subheader("💊 Phase 3: Treatment & Management Plan")

        with st.expander("🔍 Review Case Summary"):
            st.markdown("**Phase 1 Differentials:** " + ", ".join(st.session_state.initial_differentials))
            st.markdown("**Phase 2 Key Findings:**")
            for t in st.session_state.objective_tests:
                st.markdown(f"- **{t['requested']}** ({t['category']}): {t['findings']}")

        if st.session_state.encounter_phase == 3:
            with st.form("treatment_phase_form"):
                st.markdown("### 📝 Clinical Management Plan")
                
                f_dx = st.text_input("1. Final Diagnosis:")
                f_edu = st.text_area("2. Patient Education:", height=80)
                f_pain = st.text_area("3. Pain Management Strategy:", height=80)
                f_mob = st.text_area("4. Mobility Prescription:", height=80)
                f_str = st.text_area("5. Strength/Rehabilitation Plan:", height=80)

                if st.form_submit_button("Submit Complete Plan", type="primary"):
                    if not f_dx.strip() or not f_edu.strip() or not f_pain.strip() or not f_mob.strip() or not f_str.strip():
                        st.error("Please fill in all 5 management fields.")
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
            st.markdown(f"**Final Diagnosis:** `{st.session_state.tx_final_dx}`")
            
            st.markdown("---")
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
    # 12. EXPORT ENCOUNTER TRANSCRIPT
    # ==============================================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Export Records")
    if st.sidebar.button("Compile Full Transcript", use_container_width=True):
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
                label="📥 Download Transcript (.txt)",
                data=export,
                file_name=f"MSK_Transcript_{st.session_state.ccid}_{student_case_key}.txt",
                mime="text/plain",
                use_container_width=True
            )