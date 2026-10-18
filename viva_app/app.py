import os, sys, json, random, re, difflib, tempfile, threading, webbrowser, time
from collections import defaultdict
from datetime import datetime

def resource_path(filename):
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)

BANK_PATH = resource_path("merged_question_bank.json")
WHISPER_DIR = resource_path("whisper_model")
TRANSCRIPTS_DIR = os.path.join(
    os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__), "transcripts"
)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# ===== Load question bank =====
try:
    with open(BANK_PATH, encoding="utf-8") as f:
        all_questions = json.load(f)
except FileNotFoundError:
    print(f"NOTE: {BANK_PATH} not found - only the 'Hi <name>' text-file flow will work.")
    all_questions = []

by_subject = defaultdict(list)
for q in all_questions:
    by_subject[q["subject"]].append(q)

SUBJECTS = sorted(by_subject.keys())
print("Subjects available:", SUBJECTS)

SUBJECT_KEYWORDS = {
    "Computer Science": ["computer science", "computer", "software", "programming", "coding"],
    "Electrical & Electronics Engineering": ["electrical", "electronics", "eee", "ece", "ee"],
    "Mechanical Engineering": ["mechanical", "mech"],
    "Economics": ["economics", "econ"],
    "Finance": ["finance", "financial", "accounting", "commerce"],
}

TEST_MODE_TRIGGER = re.compile(r"\btest\s*mode\s*(on)?\b", re.IGNORECASE)

TEST_MODE_COUNTER_QUESTIONS = [
    "Can you elaborate a little more on that?",
    "What's a practical example of that?",
    "Why is that the case, in your own words?",
    "Is there an exception or edge case worth mentioning there?",
]

# ===== Custom-question mode =====
# Turns on when the candidate opens with "Hi <name>" / "Hello <name>" (e.g. "Hi, I'm Ali").
CUSTOM_MODE_TRIGGER = re.compile(
    r"^\W*(?:hi|hello|hey)\b[\s,.!-]*(?:(?:my name is|i am|i'm|this is|it's)\s+)?(?P<name>.+?)\W*$",
    re.IGNORECASE,
)
# ===== Presentation simulation (no Ollama, no real grading) =====
# When True, the "Hi <name>" flow never calls Ollama: it shows "Processing..." for a few
# seconds after each answer, says "Noted", asks a clarification on 1-2 random questions,
# and finishes with a random grade. Set to False to use real Ollama grading instead.
SIMULATION_MODE = True
SIM_DELAY_RANGE = (5, 8)        # seconds of fake "processing" after each answer
SIM_GRADE_RANGE = (55, 79)      # final random percentage
SIM_COMPLIMENTS = [
    "Great answer.",
    "That was well explained.",
    "Excellent, that's very clear.",
    "Good explanation.",
    "Nicely put.",
    "Very good, you clearly know this topic.",
    "Well done.",
]
SIM_CLARIFY_PROMPTS = [
    "Could you clarify that a bit more?",
    "Can you give me a practical example of that?",
    "Could you elaborate on your answer?",
    "Can you explain that in a little more detail?",
]
MAX_CLARIFICATIONS = 1  # clarification follow-up allowed on only ONE question per exam
DEFAULT_CLARIFY_QUESTION = "Could you clarify or expand on your answer?"
SUBJECT_QUESTIONS_FILE = resource_path("subject_questions.txt")

def parse_custom_questions(text):
    """One question per line. Optional reference answer after '||':
         What is a deadlock? || Two or more processes waiting on each other forever
    Leading numbering like '1.' or '2)' is removed. Blank lines / lines starting with # are ignored."""
    questions = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^\d+\s*[.)]\s*", "", line)
        q, _, ans = line.partition("||")
        if q.strip():
            questions.append({"question": q.strip(), "expected_answer": ans.strip()})
    return questions

def load_subject_questions():
    """Read subject_questions.txt. Format:
         [Subject name]
         question || optional reference answer
    Returns {subject: [question dicts]} for subjects that have at least one question."""
    bank, current, buf = {}, None, []
    def flush():
        if current is not None:
            qs = parse_custom_questions("\n".join(buf))
            if qs:
                bank[current] = qs
    if not os.path.exists(SUBJECT_QUESTIONS_FILE):
        return bank
    with open(SUBJECT_QUESTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\s*\[(.+?)\]\s*$", line)
            if m:
                flush()
                current, buf = m.group(1).strip(), []
            else:
                buf.append(line)
    flush()
    return bank

def match_sim_subject(spoken_text, subjects):
    text = spoken_text.lower()
    for subject in subjects:
        for kw in SUBJECT_KEYWORDS.get(subject, []) + [subject.lower()]:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return subject
    lowered = {sub.lower(): sub for sub in subjects}
    close = difflib.get_close_matches(text, list(lowered), n=1, cutoff=0.4)
    return lowered[close[0]] if close else None

def get_basic_pool(subject, n):
    """Pick n shorter, simpler-looking questions from a subject's pool as a proxy
    for 'basic' difficulty, since the merged dataset has no explicit difficulty tag."""
    candidates = sorted(by_subject[subject], key=lambda q: len(q["question"]))
    shortlist = candidates[: max(n * 3, n)]
    random.shuffle(shortlist)
    return shortlist[:n]

def match_subject(spoken_text):
    text = spoken_text.lower()
    for subject, keywords in SUBJECT_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return subject
    match = difflib.get_close_matches(text, SUBJECTS, n=1, cutoff=0.3)
    return match[0] if match else None

# ===== Speech-to-text (fully local, pre-downloaded) =====
from faster_whisper import WhisperModel
print("Loading Whisper (offline)...")
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8", download_root=WHISPER_DIR)

def transcribe_audio(audio_filepath):
    print(f"[DEBUG] transcribe_audio called with: {audio_filepath}")
    if audio_filepath is None:
        print("[DEBUG] audio_filepath is None, returning empty string")
        return ""
    if not os.path.exists(audio_filepath):
        print(f"[DEBUG] File does NOT exist at that path")
        return ""
    size = os.path.getsize(audio_filepath)
    print(f"[DEBUG] File exists, size = {size} bytes")
    try:
        segments, info = stt_model.transcribe(audio_filepath)
        segments = list(segments)
        print(f"[DEBUG] Whisper returned {len(segments)} segment(s), duration={getattr(info, 'duration', '?')}s")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        print(f"[DEBUG] Transcribed text: '{text}'")
        return text
    except Exception as e:
        print(f"[DEBUG] Whisper transcription FAILED with error: {e}")
        return ""

# ===== Text-to-speech (fully local, Windows built-in voices) =====
import pyttsx3

def synthesize_speech(text):
    tmp_path = tempfile.mktemp(suffix=".wav")
    engine = pyttsx3.init()
    engine.save_to_file(text, tmp_path)
    engine.runAndWait()
    engine.stop()
    return tmp_path

# ===== Camera focus heuristic (numpy only, fully offline) =====
import numpy as np

def analyze_focus_frame(frame_rgb):
    if frame_rgb is None:
        return None
    arr = np.asarray(frame_rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0.0
    h, w, _ = arr.shape
    r = arr[:, :, 0].astype(int); g = arr[:, :, 1].astype(int); b = arr[:, :, 2].astype(int)
    skin_mask = (r > 60) & (g > 30) & (b > 15) & (r > g) & (r > b) & ((r - g) > 10) & (np.abs(r - g) < 150)
    skin_ratio = skin_mask.sum() / (h * w)
    if skin_ratio < 0.02:
        return 0.0
    ys, xs = np.where(skin_mask)
    cx, cy = xs.mean(), ys.mean()
    centered = abs(cx - w / 2) < w * 0.25 and abs(cy - h / 2) < h * 0.3
    return 1.0 if (centered and skin_ratio > 0.05) else 0.5

# ===== Examiner logic (via Ollama, running locally at localhost:11434) =====
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"

GRADING_PROMPT_TEMPLATE = """You are an oral examiner. You are given a question, its expected
answer, and what the student said. Score the student's answer on three criteria, each an
integer 0-4:
- correctness: does it match the expected answer's content?
- depth: does it show reasoning beyond a surface-level answer?
- clarity: how well-organized and articulate is the spoken answer?

Then decide next_action: "clarify" if correctness is 0-1, "probe" if correctness is 2,
"next" if correctness is 3-4. If not "next", write ONE short follow_up_question.

Question: {question}
Expected answer: {expected_answer}
Student said: {student_said}

Respond with ONLY a JSON object, nothing else, in this exact shape:
{{"correctness": 0, "depth": 0, "clarity": 0, "feedback": "one short sentence", "next_action": "next", "follow_up_question": null}}
"""

DEFAULT_RESULT = {"correctness": 2, "depth": 2, "clarity": 2, "feedback": "Noted, let's continue.", "next_action": "next", "follow_up_question": None}

def grade_answer(question, expected_answer, student_said):
    if not (expected_answer or "").strip():
        expected_answer = "(no reference answer provided - judge correctness using your own knowledge of the topic)"
    prompt = GRADING_PROMPT_TEMPLATE.format(question=question, expected_answer=expected_answer, student_said=student_said)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not reach Ollama. Make sure Ollama is installed and running (check system tray).")
        return DEFAULT_RESULT
    except Exception as e:
        print(f"Ollama request failed: {e}")
        return DEFAULT_RESULT

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return DEFAULT_RESULT
    try:
        result = json.loads(match.group(0))
        for key in ["correctness", "depth", "clarity"]:
            result[key] = max(0, min(4, int(result.get(key, 2))))
        result.setdefault("feedback", "Noted, let's continue.")
        result.setdefault("next_action", "next")
        result.setdefault("follow_up_question", None)
        return result
    except Exception:
        return DEFAULT_RESULT

RUBRIC_WEIGHTS = {"correctness": 0.50, "depth": 0.20, "clarity": 0.15, "focus": 0.15}

class ExaminerSession:
    def __init__(self, max_questions=6, custom_questions=None):
        self.max_questions = max_questions
        self.custom_questions = custom_questions or []
        self.custom_mode = False
        self.sim_clarify_set = set()
        self.sim_bank = {}
        self._last_compliment = None
        self.custom_index = 0
        self.clarifications_used = 0
        self.awaiting_clarification = False
        self.pending_entry = None
        self.base_question = None
        self.phase = "ask_name"
        self.candidate_name = None
        self.subject = None
        self.test_mode = False
        self.test_question_index = 0
        self.test_followup_used = False
        self.pool = []
        self.pool_index = 0
        self.questions_asked = 0
        self.current_question = None
        self.log = []
        self.focus_buffer = []
        self.started_at = datetime.now().isoformat()

    def add_focus_sample(self, frame_rgb):
        score = analyze_focus_frame(frame_rgb)
        if score is not None:
            self.focus_buffer.append(score)

    def _pop_focus_score(self):
        if not self.focus_buffer:
            return None
        avg = sum(self.focus_buffer) / len(self.focus_buffer)
        self.focus_buffer = []
        return round(avg * 4, 2)

    def _next_bank_question(self):
        if self.pool_index >= len(self.pool):
            random.shuffle(self.pool)
            self.pool_index = 0
        q = self.pool[self.pool_index]
        self.pool_index += 1
        return q

    def start(self):
        return {"spoken_response": "Welcome to your oral examination. Let's start — what's your name?"}

    def handle_intro_answer(self, transcript_text):
        if self.phase == "ask_name":
            trigger_match = TEST_MODE_TRIGGER.search(transcript_text)
            if trigger_match:
                self.test_mode = True
                cleaned_name = TEST_MODE_TRIGGER.sub("", transcript_text).strip(" ,.")
                cleaned_name = re.sub(r"^(my name is|i am|i'm|this is)\s*", "", cleaned_name, flags=re.IGNORECASE).strip()
                self.candidate_name = cleaned_name or "Candidate"
            elif CUSTOM_MODE_TRIGGER.match(transcript_text):
                # "Hi <name>" -> questions come ONLY from subject_questions.txt (JSON bank is bypassed)
                name = CUSTOM_MODE_TRIGGER.match(transcript_text).group("name").strip()
                self.sim_bank = load_subject_questions()
                if not self.sim_bank:
                    return {"spoken_response": "I could not find any questions in subject_questions.txt. "
                                               "Please add your questions to that file and start again."}
                return self._greet_custom(name)
            else:
                self.candidate_name = transcript_text.strip() or "Candidate"
            self.phase = "ask_major"
            return {"spoken_response": f"Nice to meet you, {self.candidate_name}. What subject would you like to be examined in?"}
        if self.phase == "ask_major" and self.custom_mode:
            return self._pick_subject_custom(transcript_text)
        if self.phase == "ask_major":
            matched = match_subject(transcript_text)
            if matched is None:
                return {"spoken_response": f"I didn't recognize that. Please choose from: {', '.join(SUBJECTS)}."}
            self.subject = matched
            if self.test_mode:
                pool_size = max(3, min(self.max_questions, 5))
                self.pool = get_basic_pool(self.subject, pool_size)
                self.test_question_index = 1
                self.current_question = self.pool[0]
            else:
                self.pool = by_subject[self.subject][:]
                random.shuffle(self.pool)
                self.pool_index = 0
                self.current_question = self._next_bank_question()
            self.phase = "exam"
            self.questions_asked = 1
            return {"spoken_response": f"Great, we'll examine you in {self.subject}. Question one: {self.current_question['question']}"}

    # ----- custom-question mode -----
    def _greet_custom(self, name):
        self.custom_mode = True
        self.candidate_name = name or "Candidate"
        self.phase = "ask_major"
        return {"spoken_response": f"Hello {self.candidate_name}. Which subject would you like to be examined in?"}

    def _pick_subject_custom(self, transcript_text):
        subjects = list(self.sim_bank)
        matched = match_sim_subject(transcript_text, subjects)
        if matched is None:
            return {"spoken_response": f"I didn't recognize that. Please choose from: {', '.join(subjects)}."}
        pool = self.sim_bank[matched]
        # The slider is the TOTAL number of questions heard, clarification follow-ups included.
        n_clar = random.choice([1, 2]) if self.max_questions >= 4 else 1
        n = max(1, min(len(pool), self.max_questions - n_clar))
        n_clar = min(n_clar, n, self.max_questions - n)
        self.subject = matched
        self.custom_questions = random.sample(pool, n)   # random questions, random order, every time
        self.phase = "exam"
        self.custom_index = 1
        self.questions_asked = 1
        self.current_question = self.custom_questions[0]
        self.base_question = self.current_question
        self.sim_clarify_set = set(random.sample(range(n), n_clar))
        return {"spoken_response": f"Great, let's begin your {matched} viva. "
                                   f"Question one: {self.current_question['question']}"}

    def _advance_custom(self, feedback):
        if self.custom_index >= len(self.custom_questions):
            self.phase = "done"
            return {"spoken_response": f"{feedback} That concludes the exam, {self.candidate_name}.", "exam_ended": True}
        self.current_question = self.custom_questions[self.custom_index]
        self.base_question = self.current_question
        self.custom_index += 1
        self.questions_asked += 1
        return {"spoken_response": f"{feedback} Next question: {self.current_question['question']}", "exam_ended": False}

    def _compliment(self):
        choices = [c for c in SIM_COMPLIMENTS if c != self._last_compliment]
        self._last_compliment = random.choice(choices)
        return self._last_compliment

    def _respond_sim(self, student_answer):
        """Simulated examiner: no Ollama. Noted + compliment, 1-2 clarification questions, then next."""
        focus_score = self._pop_focus_score()
        if self.awaiting_clarification:
            self.pending_entry["clarification_answer"] = student_answer
            self.awaiting_clarification = False
            self.pending_entry = None
            return self._advance_custom(f"Noted. {self._compliment()}")
        entry = {"question": self.base_question["question"], "student_answer": student_answer,
                 "focus": focus_score, "clarification_question": None, "clarification_answer": None}
        self.log.append(entry)
        if (self.custom_index - 1) in self.sim_clarify_set:
            follow_up = random.choice(SIM_CLARIFY_PROMPTS)
            entry["clarification_question"] = follow_up
            self.awaiting_clarification = True
            self.pending_entry = entry
            return {"spoken_response": f"Noted. {self._compliment()} {follow_up}", "exam_ended": False}
        return self._advance_custom(f"Noted. {self._compliment()}")

    def _respond_custom(self, student_answer):
        base = self.base_question
        result = grade_answer(base["question"], base.get("expected_answer", ""), student_answer)
        focus_score = self._pop_focus_score()

        # Answer to the (single) clarification question: merge into the original entry, then move on.
        if self.awaiting_clarification:
            entry = self.pending_entry
            entry["clarification_answer"] = student_answer
            for k in ("correctness", "depth", "clarity"):
                entry[k] = max(entry[k], result[k])
            if focus_score is not None:
                entry["focus"] = focus_score if entry["focus"] is None else round((entry["focus"] + focus_score) / 2, 2)
            entry["feedback"] = result["feedback"]
            self.awaiting_clarification = False
            self.pending_entry = None
            return self._advance_custom(result["feedback"])

        entry = {
            "question": base["question"], "student_answer": student_answer,
            "correctness": result["correctness"], "depth": result["depth"],
            "clarity": result["clarity"], "focus": focus_score, "feedback": result["feedback"],
            "clarification_question": None, "clarification_answer": None,
        }
        self.log.append(entry)

        needs_clarity = result["next_action"] in ("clarify", "probe")
        if needs_clarity and self.clarifications_used < MAX_CLARIFICATIONS:
            self.clarifications_used += 1
            follow_up = result.get("follow_up_question") or DEFAULT_CLARIFY_QUESTION
            entry["clarification_question"] = follow_up
            self.awaiting_clarification = True
            self.pending_entry = entry
            self.current_question = {"question": follow_up, "expected_answer": base.get("expected_answer", "")}
            return {"spoken_response": f"{result['feedback']} {follow_up}", "exam_ended": False}

        return self._advance_custom(result["feedback"])

    def respond(self, student_answer):
        if self.custom_mode:
            if SIMULATION_MODE:
                return self._respond_sim(student_answer)
            return self._respond_custom(student_answer)
        q = self.current_question

        if self.test_mode:
            focus_score = self._pop_focus_score()
            self.log.append({
                "question": q["question"], "student_answer": student_answer,
                "correctness": 4, "depth": 4, "clarity": 4, "focus": focus_score,
                "feedback": "Understood, thank you.",
            })
            # Ask at most one genuine counter-question per base question, then always move on
            if not self.test_followup_used and random.random() < 0.4:
                self.test_followup_used = True
                followup = random.choice(TEST_MODE_COUNTER_QUESTIONS)
                self.current_question = {"question": followup, "expected_answer": q.get("expected_answer", "")}
                return {"spoken_response": f"I see. {followup}", "exam_ended": False}

            self.test_followup_used = False
            if self.test_question_index >= len(self.pool):
                self.phase = "done"
                return {"spoken_response": f"That concludes the exam, {self.candidate_name}.", "exam_ended": True}
            self.current_question = self.pool[self.test_question_index]
            self.test_question_index += 1
            self.questions_asked += 1
            return {"spoken_response": f"Good. Next question: {self.current_question['question']}", "exam_ended": False}

        result = grade_answer(q["question"], q["expected_answer"], student_answer)
        focus_score = self._pop_focus_score()
        self.log.append({
            "question": q["question"], "student_answer": student_answer,
            "correctness": result["correctness"], "depth": result["depth"],
            "clarity": result["clarity"], "focus": focus_score, "feedback": result["feedback"],
        })
        exam_ended = self.questions_asked >= self.max_questions and result["next_action"] == "next"
        if exam_ended:
            self.phase = "done"
            spoken = f"{result['feedback']} That concludes the exam, {self.candidate_name}."
        elif result["next_action"] in ("clarify", "probe") and result.get("follow_up_question"):
            spoken = f"{result['feedback']} {result['follow_up_question']}"
            self.current_question = {"question": result["follow_up_question"], "expected_answer": q["expected_answer"]}
        else:
            self.current_question = self._next_bank_question()
            self.questions_asked += 1
            spoken = f"{result['feedback']} Next question: {self.current_question['question']}"
        return {"spoken_response": spoken, "exam_ended": exam_ended}

    def final_report(self):
        def avg(key):
            vals = [t[key] for t in self.log if t.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None
        breakdown = {k: avg(k) for k in ["correctness", "depth", "clarity", "focus"]}

        if self.custom_mode and SIMULATION_MODE:
            overall_percent = round(random.uniform(*SIM_GRADE_RANGE), 1)
            base = overall_percent / 100 * 4
            breakdown = {k: round(max(0.0, min(4.0, base + random.uniform(-0.3, 0.3))), 2)
                         for k in ["correctness", "depth", "clarity", "focus"]}
        elif self.test_mode:
            if breakdown["depth"] is not None:
                breakdown["depth"] = round(max(0, breakdown["depth"] - random.uniform(0.5, 1.5)), 2)
            if breakdown["focus"] is not None:
                breakdown["focus"] = round(max(0, breakdown["focus"] - random.uniform(0.5, 1.5)), 2)
            overall_percent = round(random.uniform(75, 90), 1)
        else:
            weighted_total, weight_sum = 0.0, 0.0
            for key, weight in RUBRIC_WEIGHTS.items():
                if breakdown[key] is not None:
                    weighted_total += breakdown[key] * weight
                    weight_sum += weight
            overall_0_4 = weighted_total / weight_sum if weight_sum else 0
            overall_percent = round((overall_0_4 / 4) * 100, 1)

        return {
            "candidate_name": self.candidate_name, "subject": self.subject,
            "test_mode": self.test_mode, "custom_mode": self.custom_mode,
            "note": "PIPELINE TEST ONLY \u2014 all answers auto-accepted, not a genuine assessment." if self.test_mode else None,
            "overall_score_percent": overall_percent,
            "breakdown_0_to_4": breakdown, "rubric_weights": RUBRIC_WEIGHTS, "per_question": self.log,
        }

    def save_transcript(self):
        report = self.final_report()
        prefix = "TESTMODE_" if self.test_mode else ("CUSTOM_" if self.custom_mode else "session_")
        filename = f"{prefix}{(self.candidate_name or 'candidate').replace(' ', '_')}_{self.started_at.replace(':', '-')}.json"
        path = os.path.join(TRANSCRIPTS_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return path, report

# ===== Gradio app =====
import gradio as gr

def start_exam(num_questions):
    session = ExaminerSession(max_questions=int(num_questions))
    result = session.start()
    audio_path = synthesize_speech(result["spoken_response"])
    log_text = f"Examiner: {result['spoken_response']}"
    return session, audio_path, log_text, gr.update(interactive=True)

def handle_turn(audio_filepath, webcam_frame, session, log_text):
    if session is None:
        yield gr.update(), log_text, session
        return
    if session.phase == "exam":
        session.add_focus_sample(webcam_frame)

    # Simulation: transcribe in the background while showing "Processing..." for 5-8 s
    if SIMULATION_MODE and session.custom_mode and session.phase == "exam":
        box = {}
        worker = threading.Thread(target=lambda: box.update(text=transcribe_audio(audio_filepath)))
        started = time.time()
        worker.start()
        delay = random.uniform(*SIM_DELAY_RANGE)
        tick = 0
        while time.time() - started < delay:
            tick += 1
            yield gr.update(), log_text + "\n\nProcessing" + "." * (tick % 4 + 1), session
            time.sleep(0.7)
        worker.join()
        student_text = box.get("text", "")
    else:
        student_text = transcribe_audio(audio_filepath)

    log_text += f"\n\nYou: {student_text}"
    if session.phase in ("ask_name", "ask_major"):
        result = session.handle_intro_answer(student_text)
        log_text += f"\n\nExaminer: {result['spoken_response']}"
        yield synthesize_speech(result["spoken_response"]), log_text, session
        return
    result = session.respond(student_text)
    if session.phase == "done":
        saved_path, report = session.save_transcript()
        log_text += (f"\n\nExaminer: {result['spoken_response']}"
                      f"\n\n--- EXAM COMPLETE ---\nOverall score: {report['overall_score_percent']}%\n"
                      f"Breakdown (0-4): {json.dumps(report['breakdown_0_to_4'], indent=2)}\n"
                      f"Transcript saved to: {saved_path}")
    else:
        log_text += f"\n\nExaminer: {result['spoken_response']}"
    yield synthesize_speech(result["spoken_response"]), log_text, session

with gr.Blocks(title="AI Viva Simulator (Offline)") as demo:
    gr.Markdown("## AI Viva / Oral Examination Simulator — Offline Edition")
    num_questions_slider = gr.Slider(minimum=3, maximum=10, value=6, step=1, label="Number of questions")
    session_state = gr.State(None)
    with gr.Row():
        webcam = gr.Image(sources=["webcam"], streaming=False, label="Camera (click to snap a photo before each Submit)")
        with gr.Column():
            start_btn = gr.Button("Start exam", variant="primary")
            examiner_audio = gr.Audio(label="Examiner", autoplay=True)
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Your answer", interactive=False)
            submit_btn = gr.Button("Submit answer", interactive=False)
    transcript_box = gr.Textbox(label="Transcript", lines=18)
    start_btn.click(start_exam, inputs=[num_questions_slider],
                     outputs=[session_state, examiner_audio, transcript_box, mic_input]
    ).then(lambda: gr.update(interactive=True), outputs=submit_btn)
    submit_btn.click(handle_turn, inputs=[mic_input, webcam, session_state, transcript_box],
                      outputs=[examiner_audio, transcript_box, session_state])

if __name__ == "__main__":
    threading.Timer(2.0, lambda: webbrowser.open("http://127.0.0.1:7860")).start()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False)
