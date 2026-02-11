# ============================================================================
# MIXED GASSES MODEL - Copy these cells into your notebook
# ============================================================================

# ==== CELL 1: MARKDOWN ====
"""
# Mixed Gasses Model
## Two-Stage Detection Pipeline:
## Stage 1: Pattern Classifier (Gas vs Interference vs Clean)
## Stage 2a: Multi-output LEL Regression (6 gases)
## Stage 2b: Interference Classifier (Dirty/Drops/Fog)
"""

# ==== CELL 2: DATA GENERATION ====
# MIXED GAS DATA GENERATION - Synthetic + Real Data Loader
# Generates training data with multiple gases at different concentrations
# Labels: 6 LEL values [CH4, C4H8, C3H8, C2H6, n-Butane, CO2] + pattern type for Stage 1

Labels_Mixed = ["CH4", "C4H8", "C3H8", "C2H6", "n-Butane", "CO2"]  # 6 combustible gases (no H2O)
N_GASES = 6
spec_interp_list_mixed = [globals()[f"spec_interp{i+2}"] for i in range(N_GASES)]  # First 6 gases only

def generate_mixed_gas_sample(concentrations, add_noise=True, noise_level=0.02):
    """Generate sensor ratios for a mixture of gases at given concentrations (0-1 = 0-100% LEL)"""
    T_total = np.ones_like(UWV)  # Start with full transmission
    for gas_idx, conc in enumerate(concentrations):
        if conc > 0:
            spec = spec_interp_list_mixed[gas_idx]
            T_gas = np.power(10, -1e6 * conc * spec)
            T_total *= T_gas
    Transmit_signal = newspec1n * T_total * detector_interp
    sA = np.dot(filter_A, Transmit_signal)
    sB = np.dot(filter_B, Transmit_signal)
    sC = np.dot(filter_C, Transmit_signal)
    sD = np.dot(filter_D, Transmit_signal)
    sAR, sBR = sA / signal_A_NG, sB / signal_B_NG
    sCR, sDR = sC / signal_C_NG, sD / signal_D_NG
    norm_AB, norm_CD = max(sAR, sBR), max(sCR, sDR)
    sAN, sBN = sA / norm_AB, sB / norm_AB
    sCN, sDN = sC / norm_CD, sD / norm_CD
    ratios = np.array([sDN/sBN, sDN/sCN, sDN/sAN, sAN/sBN, sAN/sCN, sCN/sBN], dtype=np.float32)
    norm_ratios = ratios / Ratio_NG
    if add_noise:
        norm_ratios += np.random.normal(0, noise_level, 6)
    data = (norm_ratios - norm_ratios.min()) / (norm_ratios.max() - norm_ratios.min() + 1e-8)
    return data

def generate_training_data_mixed(n_samples=5000, max_gases_per_sample=3):
    """Generate synthetic mixed gas training data"""
    X_data, Y_lel = [], []
    Y_pattern = []  # 0=Clean, 1=Gas, 2=Interference (for Stage 1)
    for _ in range(n_samples // 3):  # Clean samples (no gas)
        concentrations = np.zeros(N_GASES)
        X_data.append(generate_mixed_gas_sample(concentrations, noise_level=0.01))
        Y_lel.append(concentrations * 100)  # Convert to % LEL
        Y_pattern.append(0)  # Clean
    for _ in range(n_samples // 3):  # Single gas samples
        concentrations = np.zeros(N_GASES)
        gas_idx = np.random.randint(0, N_GASES)
        concentrations[gas_idx] = np.random.uniform(0.05, 1.0)  # 5-100% LEL
        X_data.append(generate_mixed_gas_sample(concentrations))
        Y_lel.append(concentrations * 100)
        Y_pattern.append(1)  # Gas
    for _ in range(n_samples // 3):  # Mixed gas samples (2-3 gases)
        concentrations = np.zeros(N_GASES)
        n_gases = np.random.randint(2, max_gases_per_sample + 1)
        gas_indices = np.random.choice(N_GASES, n_gases, replace=False)
        for idx in gas_indices:
            concentrations[idx] = np.random.uniform(0.05, 0.6)  # Lower max for mixtures
        X_data.append(generate_mixed_gas_sample(concentrations))
        Y_lel.append(concentrations * 100)
        Y_pattern.append(1)  # Gas
    return np.array(X_data, dtype=np.float32), np.array(Y_lel, dtype=np.float32), np.array(Y_pattern)

# ============ REAL DATA LOADER ============
def load_real_gas_data(file_path):
    """Load real recorded gas data from Excel file"""
    df = pd.read_excel(file_path)
    nim = np.array(df)
    X_real, Y_lel_real, Y_pattern_real = [], [], []
    for i in range(len(df)):
        ratios = np.array(nim[i, 1:7], dtype=np.float32)  # 6 ratio columns
        # Modify these indices based on your actual file format:
        # lel_values = np.array(nim[i, 7:13], dtype=np.float32)  # If you have LEL columns
        # pattern = nim[i, 13]  # If you have pattern label
        X_real.append(ratios)
        # Y_lel_real.append(lel_values)
        # Y_pattern_real.append(pattern)
    return np.array(X_real), np.array(Y_lel_real), np.array(Y_pattern_real)

def load_real_interference_data(dirty_file, drops_file, fog_file):
    """Load real interference data from separate Excel files"""
    X_interference, Y_interference = [], []
    
    # Load dirty lens data (label = 0)
    if dirty_file:
        df = pd.read_excel(dirty_file)
        nim = np.array(df)
        for i in range(len(df)):
            ratios = np.array(nim[i, 1:7], dtype=np.float32)
            X_interference.append(ratios)
            Y_interference.append(0)  # Dirty
    
    # Load water drops data (label = 1)
    if drops_file:
        df = pd.read_excel(drops_file)
        nim = np.array(df)
        for i in range(len(df)):
            ratios = np.array(nim[i, 1:7], dtype=np.float32)
            X_interference.append(ratios)
            Y_interference.append(1)  # Drops
    
    # Load fog data (label = 2)
    if fog_file:
        df = pd.read_excel(fog_file)
        nim = np.array(df)
        for i in range(len(df)):
            ratios = np.array(nim[i, 1:7], dtype=np.float32)
            X_interference.append(ratios)
            Y_interference.append(2)  # Fog
    
    return np.array(X_interference, dtype=np.float32), np.array(Y_interference)

# ============ GENERATE DATA ============
print("Generating synthetic mixed gas training data...")
X_mixed, Y_lel_mixed, Y_pattern_mixed = generate_training_data_mixed(n_samples=6000)

print(f"✓ Generated {len(X_mixed)} samples")
print(f"  - Input shape: {X_mixed.shape} (6 ratios)")
print(f"  - LEL output shape: {Y_lel_mixed.shape} (6 gas LEL values)")
print(f"  - Pattern labels: {np.bincount(Y_pattern_mixed)} [Clean, Gas, Interference]")


# ==== CELL 3: STAGE 1 MODEL ====
# STAGE 1: PATTERN CLASSIFIER MODEL
# Classifies ratio patterns into: 0=Clean, 1=Gas Pattern, 2=Interference Pattern

def Stage1_Pattern_Classifier():
    """Classifies ratio deviation pattern: Clean vs Gas vs Interference"""
    model = keras.models.Sequential([
        keras.Input(shape=(6,)),  # 6 normalized ratios
        keras.layers.Dense(16, activation="relu", name="s1_dense1"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(12, activation="relu", name="s1_dense2"),
        keras.layers.Dropout(0.15),
        keras.layers.Dense(8, activation="relu", name="s1_dense3"),
        keras.layers.Dense(3, activation="softmax", name="s1_output")  # 3 classes
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model

# Build and show Stage 1 model
stage1_model = Stage1_Pattern_Classifier()
print("STAGE 1: Pattern Classifier")
stage1_model.summary()

# Prepare Stage 1 training data
X_s1 = X_mixed
Y_s1 = Y_pattern_mixed

# Shuffle data
shuffle_idx = np.random.permutation(len(X_s1))
X_s1, Y_s1 = X_s1[shuffle_idx], Y_s1[shuffle_idx]

# Train/test split
split = int(0.8 * len(X_s1))
X_s1_train, X_s1_test = X_s1[:split], X_s1[split:]
Y_s1_train, Y_s1_test = Y_s1[:split], Y_s1[split:]

print(f"\n✓ Stage 1 Data: Train={len(X_s1_train)}, Test={len(X_s1_test)}")


# ==== CELL 4: TRAIN STAGE 1 ====
# TRAIN STAGE 1 MODEL
s1_callbacks = [
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_delta=1e-4),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ModelCheckpoint('stage1_pattern_classifier.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_accuracy')
]

history_s1 = stage1_model.fit(
    X_s1_train, Y_s1_train,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
    callbacks=s1_callbacks,
    verbose=1
)

# Evaluate
s1_loss, s1_acc = stage1_model.evaluate(X_s1_test, Y_s1_test, verbose=0)
print(f"\n✓ Stage 1 Test Accuracy: {s1_acc*100:.2f}%")


# ==== CELL 5: STAGE 2a MODEL ====
# STAGE 2a: MULTI-OUTPUT LEL REGRESSION MODEL
# Input: 6 ratios → Output: 6 LEL values (0-100% for each gas)

def Stage2a_LEL_Regression():
    """Multi-output regression: predicts LEL% for each of 6 gases simultaneously"""
    model = keras.models.Sequential([
        keras.Input(shape=(6,)),  # 6 normalized ratios
        keras.layers.Dense(24, activation="tanh", name="s2a_dense1"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation="tanh", name="s2a_dense2"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(24, activation="tanh", name="s2a_dense3"),
        keras.layers.Dense(16, activation="tanh", name="s2a_dense4"),
        keras.layers.Dense(6, activation="sigmoid", name="s2a_output")  # 6 outputs, scaled 0-1
    ])
    model.compile(
        loss="mse",  # Mean Squared Error for regression
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["mae"]  # Mean Absolute Error
    )
    return model

# Build and show Stage 2a model
stage2a_model = Stage2a_LEL_Regression()
print("STAGE 2a: Multi-Output LEL Regression")
stage2a_model.summary()

# Prepare Stage 2a data (only gas samples, not clean)
gas_mask = Y_pattern_mixed == 1  # Only samples with gas
X_s2a = X_mixed[gas_mask]
Y_s2a = Y_lel_mixed[gas_mask] / 100.0  # Normalize to 0-1 for sigmoid output

shuffle_idx = np.random.permutation(len(X_s2a))
X_s2a, Y_s2a = X_s2a[shuffle_idx], Y_s2a[shuffle_idx]

split = int(0.8 * len(X_s2a))
X_s2a_train, X_s2a_test = X_s2a[:split], X_s2a[split:]
Y_s2a_train, Y_s2a_test = Y_s2a[:split], Y_s2a[split:]

print(f"\n✓ Stage 2a Data: Train={len(X_s2a_train)}, Test={len(X_s2a_test)}")
print(f"  - Output range: 0-1 (multiply by 100 for LEL%)")


# ==== CELL 6: TRAIN STAGE 2a ====
# TRAIN STAGE 2a MODEL
s2a_callbacks = [
    ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=5, verbose=1),
    EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True),
    ModelCheckpoint('stage2a_lel_regression.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_mae')
]

history_s2a = stage2a_model.fit(
    X_s2a_train, Y_s2a_train,
    batch_size=64,
    epochs=100,
    validation_split=0.2,
    callbacks=s2a_callbacks,
    verbose=1
)

# Evaluate
s2a_loss, s2a_mae = stage2a_model.evaluate(X_s2a_test, Y_s2a_test, verbose=0)
print(f"\n✓ Stage 2a Test MAE: {s2a_mae*100:.2f}% LEL")


# ==== CELL 7: STAGE 2b MODEL ====
# STAGE 2b: INTERFERENCE CLASSIFIER (Structure Only - needs real data)
# Classifies interference type: 0=Dirty, 1=Drops, 2=Fog

def Stage2b_Interference_Classifier():
    """Classifies interference type when Stage 1 detects interference pattern"""
    model = keras.models.Sequential([
        keras.Input(shape=(6,)),  # 6 ratios (expand later for camera features)
        keras.layers.Dense(12, activation="relu", name="s2b_dense1"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(8, activation="relu", name="s2b_dense2"),
        keras.layers.Dense(3, activation="softmax", name="s2b_output")  # Dirty/Drops/Fog
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model

Labels_Interference = ["Dirty", "Drops", "Fog"]

# Build Stage 2b model (training requires real interference data)
stage2b_model = Stage2b_Interference_Classifier()
print("STAGE 2b: Interference Classifier (structure only - needs real data)")
stage2b_model.summary()

print("\n⚠️ Stage 2b requires real recorded data. Use load_real_interference_data():")
print("   X_interf, Y_interf = load_real_interference_data('dirty.xlsx', 'drops.xlsx', 'fog.xlsx')")


# ==== CELL 8: FULL PIPELINE ====
# FULL PIPELINE: COMBINED PREDICTION FUNCTION

def predict_full_pipeline(ratios_input, stage1_model, stage2a_model, threshold=0.6):
    """
    Full detection pipeline:
    1. Stage 1: Classify pattern type (Clean/Gas/Interference)
    2. If Gas → Stage 2a: Predict LEL% for each gas
    3. If Interference → Report interference type (future: Stage 2b)
    """
    if ratios_input.ndim == 1:
        ratios_input = ratios_input.reshape(1, -1)
    
    # Stage 1: Pattern classification
    s1_probs = stage1_model.predict(ratios_input, verbose=0)
    s1_pred = np.argmax(s1_probs, axis=1)
    s1_conf = np.max(s1_probs, axis=1)
    
    results = []
    for i in range(len(ratios_input)):
        result = {
            'pattern': ['Clean', 'Gas', 'Interference'][s1_pred[i]],
            'confidence': float(s1_conf[i]),
            'lel_values': None,
            'interference_type': None
        }
        
        if s1_pred[i] == 1 and s1_conf[i] >= threshold:  # Gas pattern
            lel_pred = stage2a_model.predict(ratios_input[i:i+1], verbose=0)[0]
            result['lel_values'] = {
                Labels_Mixed[j]: round(float(lel_pred[j] * 100), 2) 
                for j in range(N_GASES)
            }
        elif s1_pred[i] == 2:  # Interference pattern
            result['interference_type'] = "Dirty/Drops/Fog (Stage 2b not implemented)"
        
        results.append(result)
    
    return results[0] if len(results) == 1 else results

# Test the pipeline with a synthetic sample
test_concentrations = np.array([0.3, 0.0, 0.2, 0.0, 0.0, 0.0])  # 30% CH4, 20% C3H8
test_ratios = generate_mixed_gas_sample(test_concentrations, add_noise=False)

print("=" * 60)
print("PIPELINE TEST")
print("=" * 60)
print(f"True concentrations: {dict(zip(Labels_Mixed, test_concentrations * 100))} % LEL")
print(f"Input ratios: {np.round(test_ratios, 4)}")
print()

result = predict_full_pipeline(test_ratios, stage1_model, stage2a_model)
print(f"Stage 1 Prediction: {result['pattern']} (confidence: {result['confidence']*100:.1f}%)")
if result['lel_values']:
    print(f"Stage 2a LEL Predictions:")
    for gas, lel in result['lel_values'].items():
        print(f"  {gas}: {lel:.1f}% LEL")


# ==== CELL 9: SAVE MODELS ====
# SAVE ALL MODELS
stage1_model.save_weights('stage1_pattern_classifier_final.weights.h5')
stage2a_model.save_weights('stage2a_lel_regression_final.weights.h5')
# stage2b_model.save_weights('stage2b_interference_classifier.weights.h5')  # After training

print("✓ Models saved:")
print("  - stage1_pattern_classifier_final.weights.h5")
print("  - stage2a_lel_regression_final.weights.h5")
print("\nPipeline Architecture Summary:")
print("=" * 60)
print("INPUT: 6 normalized sensor ratios (after calibration)")
print("  │")
print("  ▼")
print("STAGE 1: Pattern Classifier")
print("  ├─ Clean (0)      → No action")
print("  ├─ Gas (1)        → STAGE 2a: LEL Regression")
print("  └─ Interference (2) → STAGE 2b: Interference Type")
print("  ")
print("STAGE 2a OUTPUT: 6 LEL values [CH4, C4H8, C3H8, C2H6, n-Butane, CO2]")
print("STAGE 2b OUTPUT: Interference type [Dirty, Drops, Fog]")
