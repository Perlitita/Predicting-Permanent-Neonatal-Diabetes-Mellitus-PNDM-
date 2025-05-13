import numpy as np

# Pesos y bias extraídos de Octave
W1 = np.array([
    [1.1753, 0.7176, 0.1175, -0.2596, 0.1302, 0.2927, -1.2174],
    [-1.8271, 0.9173, -0.3442, -1.3091, -2.2416, 0.5192, -0.3837],
    [1.1940, -0.5651, -0.3317, -0.9073, -0.9369, -0.7414, 1.1180],
    [-1.4146, -0.0321, 0.5486, 1.0120, 0.0639, -0.5931, 1.7304],
    [-0.1364, -0.2106, -0.5381, -1.4346, -0.5420, -1.3171, -1.2116],
    [0.8595, 0.4864, -1.3072, 0.9523, 0.4640, 0.9722, 1.1640],
    [0.1953, 0.8525, -1.3453, -0.6933, -1.0993, -0.6208, -1.7489],
    [0.5760, -2.0490, 0.6114, 0.0452, -0.2022, -0.5546, 0.2512],
    [-0.5604, -0.5873, -0.9809, 0.4783, 0.5182, 1.4133, 0.1774],
    [0.6363, 0.2076, 0.7301, -0.5632, 0.3208, 0.6449, -0.1110]
])

W2 = np.array([
    [0.2631, 0.5424, -1.1326, 0.3986, 0.5077, -0.3812, -1.0748, 1.6856, -0.1728, 1.6557],
    [0.9572, -0.7438, -0.8231, -0.6004, -1.1599, -2.5148, -0.1002, 1.8766, -0.4463, 1.9423]
])

b1 = np.array([
    -0.030655, -0.126340, 0.038274, -0.096568, -0.017835,
    0.067295, 0.170643, -0.012968, 0.101388, -0.119145
])

b2 = np.array([0.082652, -0.156192])

# Normalización para columnas 0,1,4,6 (Age, HbA1c, BirthWeight, Insulin)
ymin, ymax = 0.1, 1.0
min_vals = [1.0, 4.788392, None, None, 1.301862, None, 0.746370]
max_vals = [11.0, 9.607762, None, None, 4.434027, None, 10.610798]

def normalize(value, i):
    if min_vals[i] is None or max_vals[i] is None:
        return value
    return ((value - min_vals[i]) / (max_vals[i] - min_vals[i])) * (ymax - ymin) + ymin

def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / np.sum(e_x)

def predict_result(data):
    try:
        X = np.array([
            data["age"],
            data["hba1c"],
            float(data["genetic_info"]),
            float(data["family_history"]),
            data["birth_weight"],
            float(data["developmental_delay"]),
            data["insulin"]
        ], dtype=np.float64)

        # Normalizar 
        for i in [0, 1, 4, 6]:
            X[i] = normalize(X[i], i)

        # Capa oculta
        h = np.tanh(np.dot(W1, X) + b1)

        # Capa de salida 
        logits = np.dot(W2, h) + b2
        probs = softmax(logits)

        pred_label = np.argmax(probs)
        pred_prob = probs[pred_label]

        result = "PNDM Detected" if pred_label == 1 else "No PNDM"
        return f"{result} (confidence: {pred_prob:.2f})"
    except Exception as e:
        return f"Error in prediction: {e}"
