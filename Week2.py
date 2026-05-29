import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

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

def h(text, color=C.CYAN):
    print(f"\n{color}{C.BOLD}{'─'*55}{C.RESET}")
    print(f"{color}{C.BOLD}  {text}{C.RESET}")
    print(f"{color}{C.BOLD}{'─'*55}{C.RESET}")

def label(k, v, color=C.WHITE):
    print(f"  {C.GRAY}{k:<22}{C.RESET}{color}{v}{C.RESET}")


h("PHASE 1 : INPUT — THE IRIS BENCHMARK", C.BLUE)

iris = load_iris()
X = iris.data
y = iris.target
CLASS_NAMES = iris.target_names

label("Dataset",        "Iris Benchmark")
label("Total samples",  f"{X.shape[0]}")
label("Features (dims)",f"{X.shape[1]}  →  {list(iris.feature_names)}")
label("Classes",        f"{len(CLASS_NAMES)}  →  {list(CLASS_NAMES)}")
label("Samples/class",  "50 each (perfectly balanced)")

print(f"\n  {C.GRAY}First 5 rows of raw data:{C.RESET}")
headers = ["Sepal L", "Sepal W", "Petal L", "Petal W", "Label"]
print(f"  {C.YELLOW}{'  '.join(f'{h:<9}' for h in headers)}{C.RESET}")
for i in range(5):
    row = "  ".join(f"{v:<9.2f}" for v in X[i])
    print(f"  {row}  {CLASS_NAMES[y[i]]}")


h("PHASE 1b : FEATURE SCALING — STANDARDSCALER", C.BLUE)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

label("Split ratio",    "80% train / 20% test")
label("Train samples",  f"{X_train.shape[0]}")
label("Test samples",   f"{X_test.shape[0]}")
label("Stratified",     "Yes — class balance preserved")
label("Scaler",         "StandardScaler (Mean=0, Variance=1)")
label("Data leakage",   "Prevented — scaler fit on train only")

print(f"\n  {C.GRAY}Before scaling → sample[0]: {X_train_raw[0].round(2)}{C.RESET}")
print(f"  {C.GREEN}After  scaling → sample[0]: {X_train[0].round(2)}{C.RESET}")


h("PHASE 2 : PROCESS — KNN ALGORITHM  (K=5)", C.CYAN)

model = KNeighborsClassifier(
    n_neighbors=5,
    metric='euclidean',
    weights='uniform'
)
label("Algorithm",      "K-Nearest Neighbors (KNN)")
label("K value",        "5  (optimal elbow — not too high/low)")
label("Distance metric","Euclidean")
label("Vote weight",    "Uniform (all neighbours equal)")

model.fit(X_train, y_train)
label("Training status","Model fitted on 120 samples ✓")

predictions = model.predict(X_test)
label("Predictions",    f"Generated for {len(predictions)} test samples ✓")


h("PHASE 3 : OUTPUT — MODEL VALIDATION", C.GREEN)

accuracy = accuracy_score(y_test, predictions)
f1       = f1_score(y_test, predictions, average='weighted')
cm       = confusion_matrix(y_test, predictions)

label("Accuracy",       f"{accuracy*100:.2f}%  ← (can be a mirage!)")
label("F1 Score",       f"{f1*100:.2f}%  ← (true performance metric)")

print(f"\n  {C.YELLOW}Confusion Matrix:{C.RESET}")
print(f"  {C.GRAY}{'':12}", end="")
for name in CLASS_NAMES:
    print(f"  {C.CYAN}{name[:10]:<12}{C.RESET}", end="")
print()
for i, row in enumerate(cm):
    print(f"  {C.CYAN}{CLASS_NAMES[i][:10]:<12}{C.RESET}", end="")
    for j, val in enumerate(row):
        color = C.GREEN if i == j else C.RED
        print(f"  {color}{val:<12}{C.RESET}", end="")
    print()

print(f"\n  {C.YELLOW}Per-class Classification Report:{C.RESET}")
report = classification_report(y_test, predictions, target_names=CLASS_NAMES)
for line in report.strip().split('\n'):
    print(f"  {line}")

print(f"\n  {C.YELLOW}Sample Predictions (first 10 test items):{C.RESET}")
print(f"  {C.GRAY}{'Actual':<18}{'Predicted':<18}{'Result'}{C.RESET}")
for i in range(min(10, len(y_test))):
    actual    = CLASS_NAMES[y_test[i]]
    predicted = CLASS_NAMES[predictions[i]]
    ok = f"{C.GREEN}✓ CORRECT{C.RESET}" if y_test[i] == predictions[i] else f"{C.RED}✗ WRONG{C.RESET}"
    print(f"  {actual:<18}{predicted:<18}{ok}")


h("BONUS : PREDICT A NEW FLOWER", C.YELLOW)
print(f"  {C.GRAY}Enter measurements to classify a new iris flower.{C.RESET}")
print(f"  {C.GRAY}(Press Enter to skip and use a demo sample){C.RESET}\n")

demo = [5.1, 3.5, 1.4, 0.2]
try:
    raw = input(f"  {C.WHITE}Sepal Length (cm) [{demo[0]}]: {C.RESET}").strip()
    sl = float(raw) if raw else demo[0]
    raw = input(f"  {C.WHITE}Sepal Width  (cm) [{demo[1]}]: {C.RESET}").strip()
    sw = float(raw) if raw else demo[1]
    raw = input(f"  {C.WHITE}Petal Length (cm) [{demo[2]}]: {C.RESET}").strip()
    pl = float(raw) if raw else demo[2]
    raw = input(f"  {C.WHITE}Petal Width  (cm) [{demo[3]}]: {C.RESET}").strip()
    pw = float(raw) if raw else demo[3]
except (EOFError, ValueError):
    sl, sw, pl, pw = demo

sample = np.array([[sl, sw, pl, pw]])
scaled_sample = scaler.transform(sample)
result = model.predict(scaled_sample)[0]
proba  = model.predict_proba(scaled_sample)[0]

print(f"\n  {C.CYAN}Input:      {C.RESET}[sepal_l={sl}, sepal_w={sw}, petal_l={pl}, petal_w={pw}]")
print(f"  {C.GREEN}Prediction: {CLASS_NAMES[result].upper()}{C.RESET}")
print(f"  {C.YELLOW}Confidence breakdown:{C.RESET}")
for i, cls in enumerate(CLASS_NAMES):
    bar = "█" * int(proba[i] * 30)
    print(f"  {cls:<14} {bar:<32} {proba[i]*100:.1f}%")

print(f"\n{'═'*57}")
print(f"{C.GREEN}{C.BOLD}  Project 2 Complete — DecodeLabs Batch 2026 ✓{C.RESET}")
print(f"{'═'*57}\n")
