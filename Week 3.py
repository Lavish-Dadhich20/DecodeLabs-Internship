import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    GRAY   = "\033[90m"
    WHITE  = "\033[97m"
    ORANGE = "\033[38;5;208m"

def h(text, color=C.CYAN):
    print(f"\n{color}{C.BOLD}{'─'*58}{C.RESET}")
    print(f"{color}{C.BOLD}  {text}{C.RESET}")
    print(f"{color}{C.BOLD}{'─'*58}{C.RESET}")

def label(k, v, color=C.WHITE):
    print(f"  {C.GRAY}{k:<24}{C.RESET}{color}{v}{C.RESET}")


JOB_ROLES = {
    "Data Scientist": "python sql machine learning data analysis statistics numpy pandas scikit-learn visualization regression classification deep learning jupyter",
    "Machine Learning Engineer": "python machine learning deep learning tensorflow pytorch model deployment mlops docker kubernetes scikit-learn neural networks algorithms",
    "Data Analyst": "sql python excel data analysis visualization power bi tableau statistics reporting pandas numpy data cleaning dashboards",
    "Backend Developer": "python java nodejs sql rest api microservices docker databases postgresql mongodb authentication server deployment",
    "Frontend Developer": "javascript html css react vuejs nodejs typescript ui ux responsive design web browser dom apis",
    "Full Stack Developer": "javascript python html css react nodejs sql rest api mongodb postgresql docker git deployment frontend backend",
    "DevOps Engineer": "docker kubernetes aws cloud linux ci cd jenkins git automation bash terraform ansible monitoring deployment",
    "Cloud Architect": "aws azure google cloud docker kubernetes terraform networking security iam cloud infrastructure devops scalability",
    "Cybersecurity Analyst": "networking linux security firewalls ethical hacking vulnerability assessment penetration testing encryption protocols siem",
    "Android Developer": "java kotlin android studio mobile ui xml apis sqlite firebase rest retrofit git sdk",
    "iOS Developer": "swift objective c xcode ios mobile ui apis core data firebase git sdk apple",
    "Data Engineer": "python sql spark hadoop etl data pipelines airflow kafka postgresql aws big data databases",
    "AI Research Scientist": "python deep learning pytorch tensorflow neural networks mathematics statistics nlp computer vision research papers algorithms",
    "NLP Engineer": "python nlp spacy transformers bert huggingface text classification sentiment analysis language models tokenization",
    "Computer Vision Engineer": "python opencv deep learning cnn pytorch tensorflow image processing object detection segmentation cuda",
    "Blockchain Developer": "solidity ethereum web3 javascript smart contracts cryptography defi blockchain nodejs react python",
    "Embedded Systems Engineer": "c c++ microcontrollers rtos arduino raspberry pi hardware protocols iot firmware debugging assembly",
    "QA Engineer": "testing selenium pytest java automation test cases bug tracking agile git ci cd performance testing",
    "Product Manager": "agile scrum product roadmap analytics user research stakeholder management jira communication business strategy",
    "Business Intelligence Analyst": "sql power bi tableau data analysis business reporting dashboards etl excel statistics kpis visualization",
}


h("PHASE 1 : INPUT — USER SKILL INGESTION", C.BLUE)

print(f"\n  {C.YELLOW}Available skills (examples):{C.RESET}")
sample_skills = ["Python", "SQL", "Machine Learning", "Docker", "React", "AWS",
                 "TensorFlow", "Java", "JavaScript", "NLP", "Kubernetes", "Tableau"]
print("  " + "  ".join(f"{C.CYAN}{s}{C.RESET}" for s in sample_skills))

print(f"\n  {C.GRAY}Enter at least 3 skills you know (comma-separated).{C.RESET}")
print(f"  {C.GRAY}Example: python, machine learning, sql{C.RESET}\n")

demo_input = "python, machine learning, sql, deep learning"
try:
    raw = input(f"  {C.WHITE}Your skills: {C.RESET}").strip()
    if not raw:
        raw = demo_input
        print(f"  {C.GRAY}(using demo: {demo_input}){C.RESET}")
except EOFError:
    raw = demo_input

user_skills_list = [s.strip().lower() for s in raw.split(",") if s.strip()]

if len(user_skills_list) < 3:
    print(f"  {C.YELLOW}Less than 3 skills entered — adding demo skills for density.{C.RESET}")
    user_skills_list += ["data analysis", "statistics"]

user_profile_text = " ".join(user_skills_list)

print(f"\n  {C.GREEN}Skills captured: {C.RESET}{', '.join(user_skills_list)}")
label("Total skills entered", str(len(user_skills_list)))
label("Profile vector built", "Yes — joined as text document")


h("PHASE 2 : PROCESS — TF-IDF VECTORIZATION", C.CYAN)

job_titles = list(JOB_ROLES.keys())
job_descriptions = list(JOB_ROLES.values())

all_documents = job_descriptions + [user_profile_text]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_documents)

job_vectors  = tfidf_matrix[:-1]
user_vector  = tfidf_matrix[-1]

label("Total job roles",     str(len(job_titles)))
label("Vocabulary size",     str(len(vectorizer.vocabulary_)))
label("Vectorizer",          "TF-IDF  (penalizes generic, rewards specific terms)")
label("Similarity metric",   "Cosine Similarity  (angle-based, magnitude-invariant)")

print(f"\n  {C.YELLOW}Why TF-IDF over binary?{C.RESET}")
print(f"  {C.GRAY}Binary treats 'python' same as 'kubernetes' — both get 1.{C.RESET}")
print(f"  {C.GRAY}TF-IDF gives 'python' a lower weight (appears everywhere){C.RESET}")
print(f"  {C.GRAY}and 'NLP' a higher weight (specific, rare) — smarter matching.{C.RESET}")

print(f"\n  {C.YELLOW}Why Cosine over Euclidean?{C.RESET}")
print(f"  {C.GRAY}Euclidean is fooled by document length. A long job description{C.RESET}")
print(f"  {C.GRAY}scores far from a short user profile even if perfectly aligned.{C.RESET}")
print(f"  {C.GRAY}Cosine only measures the ANGLE between vectors — not the size.{C.RESET}")


h("PHASE 2b : SCORING — COSINE SIMILARITY ENGINE", C.CYAN)

scores = cosine_similarity(user_vector, job_vectors).flatten()

scored_roles = list(zip(job_titles, scores))
scored_roles.sort(key=lambda x: x[1], reverse=True)

print(f"\n  {C.YELLOW}All roles scored (descending):{C.RESET}")
print(f"  {C.GRAY}{'Rank':<6}{'Job Role':<32}{'Similarity Score':<18}{'Match Bar'}{C.RESET}")
print(f"  {'─'*70}")
for i, (title, score) in enumerate(scored_roles, 1):
    bar_len = int(score * 35)
    bar = "█" * bar_len
    color = C.GREEN if i <= 3 else C.GRAY
    marker = f"{C.YELLOW} ← TOP 3{C.RESET}" if i <= 3 else ""
    print(f"  {color}#{i:<5}{title:<32}{score:.4f}{'':12}{bar}{C.RESET}{marker}")


h("PHASE 3 : OUTPUT — TOP-N RECOMMENDATIONS", C.GREEN)

TOP_N = 3
top_recommendations = scored_roles[:TOP_N]

print(f"\n  {C.BOLD}{C.GREEN}🎯  Your Top {TOP_N} Career Recommendations{C.RESET}\n")

for rank, (title, score) in enumerate(top_recommendations, 1):
    match_pct = score * 100
    bar = "█" * int(score * 40)
    role_skills = JOB_ROLES[title].split()

    matched = [s for s in user_skills_list if s in role_skills]
    missing = [s for s in role_skills[:6] if s not in user_skills_list][:3]

    rank_color = [C.YELLOW, C.WHITE, C.CYAN][rank - 1]
    print(f"  {rank_color}{C.BOLD}#{rank}  {title}{C.RESET}")
    print(f"  {C.GRAY}{'─'*50}{C.RESET}")
    print(f"  {'Match Score':<20} {C.GREEN}{match_pct:.1f}%{C.RESET}  {C.GREEN}{bar}{C.RESET}")
    print(f"  {'Skills matched':<20} {C.CYAN}{', '.join(matched) if matched else 'none directly'}{C.RESET}")
    print(f"  {'Skills to add':<20} {C.YELLOW}{', '.join(missing)}{C.RESET}")
    print()

label("Algorithm",         "Content-Based Filtering")
label("Feature extraction","TF-IDF Weighted Vectors")
label("Similarity method", "Cosine Similarity (A·B / ||A||||B||)")
label("Pipeline",          "Ingestion → Scoring → Sorting → Filtering")
label("Cold start handled","Yes — onboarding survey (minimum 3 inputs)")

print(f"\n{'═'*60}")
print(f"{C.GREEN}{C.BOLD}  Project 3 Complete — DecodeLabs Batch 2026 ✓{C.RESET}")
print(f"{'═'*60}\n")
