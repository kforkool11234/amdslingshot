"""
patch_notebook.py
Applies all flat-line prediction fixes to 2.0-JD-energy_load_prediction.ipynb.
Run from the amdhack root:
    python AGES/patch_notebook.py
"""

import json, copy, re

NB_PATH = r"AGES/notebooks/2.0-JD-energy_load_prediction.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

def src(cell):
    return "".join(cell.get("source", []))

def set_src(cell, code: str):
    # Store as list of lines (Jupyter convention)
    lines = code.splitlines(keepends=True)
    cell["source"] = lines

def new_code_cell(code: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }

changed = []

for i, cell in enumerate(cells):
    s = src(cell)

    # -----------------------------------------------------------------
    # 1. REGULARIZATION  1e-4 → 1e-5
    # -----------------------------------------------------------------
    if s.strip() == "REGULARIZATION = 1e-4":
        set_src(cell, "REGULARIZATION = 1e-5  # was 1e-4 — too strong, collapsed weights to mean")
        changed.append("REGULARIZATION")

    # -----------------------------------------------------------------
    # 2. DROP  0.1 → 0.15
    # -----------------------------------------------------------------
    elif s.strip() == "DROP = 0.1":
        set_src(cell, "DROP = 0.15  # was 0.1 — regularize via dropout not L2")
        changed.append("DROP")

    # -----------------------------------------------------------------
    # 3. MODEL — replace plain GRU with Bidirectional GRU
    # -----------------------------------------------------------------
    elif "GRU(128" in s and "Sequential" in s:
        new_model = (
            "model = Sequential([\n"
            "    Input(shape=(STEPS, FEATURES)),\n"
            "\n"
            "    # Bidirectional GRU Block 1 (was plain GRU(128))\n"
            "    Bidirectional(GRU(64, activation='tanh',\n"
            "                      kernel_regularizer=l2(REGULARIZATION),\n"
            "                      return_sequences=True)),\n"
            "    BatchNormalization(),\n"
            "    Dropout(DROP),\n"
            "\n"
            "    # Bidirectional GRU Block 2 (was plain GRU(64))\n"
            "    Bidirectional(GRU(32, activation='tanh',\n"
            "                      kernel_regularizer=l2(REGULARIZATION))),\n"
            "    BatchNormalization(),\n"
            "    Dropout(DROP),\n"
            "\n"
            "    Dense(32, activation='relu', kernel_initializer='he_uniform'),\n"
            "    Dense(OUTPUT)\n"
            "])"
        )
        set_src(cell, new_model)
        changed.append("MODEL → BiGRU")

    # -----------------------------------------------------------------
    # 4. OPTIMIZER  1e-3 → 5e-4
    # -----------------------------------------------------------------
    elif s.strip() == "OPTIMIZER = Adam(learning_rate=1e-3)":
        set_src(cell, "OPTIMIZER = Adam(learning_rate=5e-4)  # was 1e-3 — too high, skipped patterns")
        changed.append("OPTIMIZER LR")

    # -----------------------------------------------------------------
    # 5. COMPILE — loss log_cosh → mse
    # -----------------------------------------------------------------
    elif "log_cosh" in s and "model.compile" in s:
        set_src(cell,
            "model.compile(\n"
            "    optimizer=OPTIMIZER,\n"
            "    loss='mse',  # was log_cosh — too flat near mean; hides collapse\n"
            "    metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]\n"
            ")"
        )
        changed.append("LOSS log_cosh → mse")

    # -----------------------------------------------------------------
    # 6. LR SCHEDULER  factor 0.2, patience 7 → factor 0.5, patience 5
    # -----------------------------------------------------------------
    elif "ReduceLROnPlateau" in s and "factor=0.2" in s:
        set_src(cell,
            "LR_SCHEDULER = ReduceLROnPlateau(\n"
            "    monitor='val_loss', factor=0.5,  # was 0.2 — too aggressive\n"
            "    patience=5, min_lr=1e-6, verbose=1\n"
            ")"
        )
        changed.append("LR SCHEDULER")

    # -----------------------------------------------------------------
    # 7. INSERT lag-features cell RIGHT BEFORE getSequence cell
    # -----------------------------------------------------------------
    elif "def getSequence" in s:
        lag_cell = new_code_cell(
            "# ── Lag features (autoregressive signal) ────────────────────────────\n"
            "# Most impactful fix: gives the model 'memory' of recent values.\n"
            "# Without this, the model sees no autocorrelation → predicts mean.\n"
            "for lag in [1, 4, 8, 16, 24]:   # 15 min, 1 hr, 2 hr, 4 hr, 6 hr ago\n"
            "    for col in ['Solar Power (kW)', 'Wind Power (kW)', 'Power Consumption (kW)']:\n"
            "        TRAINING_DATASET[f'{col}_lag{lag}'] = TRAINING_DATASET[col].shift(lag)\n"
            "        TESTING_DATASET[f'{col}_lag{lag}']  = TESTING_DATASET[col].shift(lag)\n"
            "\n"
            "TRAINING_DATASET.dropna(inplace=True)\n"
            "TESTING_DATASET.dropna(inplace=True)\n"
            "print(f'Train shape after lags: {TRAINING_DATASET.shape}')\n"
            "print(f'Test  shape after lags: {TESTING_DATASET.shape}')",
            "lag_features_fix"
        )
        cells.insert(i, lag_cell)   # insert BEFORE getSequence
        changed.append("LAG FEATURES inserted before getSequence")
        break   # index shifted; remaining fixes already done above

# Also fix DROP column — need TARGET to be recalculated after lag columns added
# Find TARGET cell and update it to recalculate after lags
for cell in cells:
    s = src(cell)
    if "TARGET_COLUMNS" in s and "get_loc" in s:
        set_src(cell,
            "TARGET_COLUMNS = ['Power Consumption (kW)', 'Solar Power (kW)', 'Wind Power (kW)']\n"
            "\n"
            "# Recalculate indices AFTER lag columns are added\n"
            "TARGET = [TRAINING_DATASET.columns.get_loc(col) for col in TARGET_COLUMNS]"
        )
        changed.append("TARGET indices recalculated after lags")
        break

# -----------------------------------------------------------------
# Write patched notebook
# -----------------------------------------------------------------
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nPatch complete! Changes applied:")
for c in changed:
    print(f"  [OK]  {c}")
print(f"\nNotebook saved: {NB_PATH}")
print("Re-run the notebook from the top to retrain with fixes.")
