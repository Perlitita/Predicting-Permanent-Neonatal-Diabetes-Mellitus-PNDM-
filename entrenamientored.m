clc; close all; clear all;

% ==================== CARGA DE DATOS ====================
% Cargar datos de entrenamiento
bien = csvread('pndmBien_train.csv');
mal  = csvread('pndmMal_train.csv');

Dt = 105;  % número de datos por clase

Bien = bien(:, 1:7);
Mal  = mal(:, 1:7);

% Normalización min-max
datos_total = [Bien; Mal];
min_val = min(datos_total);
max_val = max(datos_total);

ymax = 1;
ymin = 0.1;

datos_normal = ((ymax - ymin) * (datos_total - min_val)) ./ (max_val - min_val) + ymin;

% ==================== FUNCIONES ====================
function A = softmax(z)
  expZ = exp(z - max(z, [], 1));
  A = expZ ./ sum(expZ, 1);
end

% ==================== PREPARAR DATOS ====================
input = datos_normal';

T_Bien = [ones(1, Dt); zeros(1, Dt)];
T_Mal  = [zeros(1, Dt); ones(1, Dt)];
targets = [T_Bien, T_Mal];

% ==================== HIPERPARÁMETROS ====================
num_inputs = size(input, 1);
num_hidden = 12;
num_outputs = size(targets, 1);
learning_rate = 0.000009;
epochs = 200;
lambda = 0.05;

% ==================== INICIALIZACIÓN ====================
W1 = randn(num_hidden, num_inputs);
W2 = randn(num_outputs, num_hidden);
b1 = randn(num_hidden, 1) * 0.1;
b2 = randn(num_outputs, 1) * 0.1;

% ==================== ENTRENAMIENTO ====================
for epoch = 1:epochs
  Z1 = W1 * input + b1;
  A1 = tanh(Z1);
  Z2 = W2 * A1 + b2;
  A2 = softmax(Z2);

  loss = -sum(sum(targets .* log(A2))) / size(targets, 2);

  % Backpropagation
  dZ2 = A2 - targets;
  dW2 = dZ2 * A1';
  db2 = sum(dZ2, 2);

  dA1 = W2' * dZ2;
  dZ1 = dA1 .* (1 - A1.^2);
  dW1 = dZ1 * input';
  db1 = sum(dZ1, 2);

  % Actualización
  W1 = W1 - learning_rate * (dW1 + lambda * W1);
  W2 = W2 - learning_rate * (dW2 + lambda * W2);
  b1 = b1 - learning_rate * db1;
  b2 = b2 - learning_rate * db2;

  if mod(epoch, 100) == 0
    fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
  end
end

% ==================== EVALUACIÓN ENTRENAMIENTO ====================
Z1 = W1 * input + b1;
A1 = tanh(Z1);
Z2 = W2 * A1 + b2;
A2 = softmax(Z2);

[~, y_pred] = max(A2, [], 1);
[~, y_test] = max(targets, [], 1);

confMat = zeros(num_outputs, num_outputs);
for i = 1:length(y_test)
  confMat(y_test(i), y_pred(i)) = confMat(y_test(i), y_pred(i)) + 1;
end

accuracy = sum(diag(confMat)) / sum(confMat(:)) * 100;

TN = confMat(1,1); FP = confMat(1,2);
FN = confMat(2,1); TP = confMat(2,2);

precision_0 = TN / (TN + FN);
recall_0 = TN / (TN + FP);
f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0);

precision_1 = TP / (TP + FP);
recall_1 = TP / (TP + FN);
f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1);

precision_macro = (precision_0 + precision_1) / 2;
recall_macro = (recall_0 + recall_1) / 2;
f1_macro = (f1_0 + f1_1) / 2;

fprintf('\n--- ENTRENAMIENTO ---\n');
disp('Matriz de confusión:');
disp(confMat);
fprintf('Precisión entrenamiento: %.2f%%\n', accuracy);
fprintf('F1 Score Promedio: %.2f%%\n', f1_macro * 100);

% ==================== GUARDAR RED ====================
red.W1 = W1; red.W2 = W2;
red.b1 = b1; red.b2 = b2;
save('red_prueba8.mat', 'red');

% ==================== EVALUACIÓN EN PRUEBA ====================
Test_bien = csvread("pndmBien_test.csv");
Test_mal  = csvread("pndmMal_test.csv");

Test_bien = Test_bien(:, 1:7);
Test_mal  = Test_mal(:, 1:7);

Norm_bien_test = ((ymax - ymin) * (Test_bien - min_val)) ./ (max_val - min_val) + ymin;
Norm_mal_test  = ((ymax - ymin) * (Test_mal - min_val)) ./ (max_val - min_val) + ymin;

TP = 0; FP = 0; TN = 0; FN = 0;

for i = 1:size(Norm_bien_test, 1)
  Z1 = red.W1 * Norm_bien_test(i,:)' + red.b1;
  A1 = tanh(Z1);
  Z2 = red.W2 * A1 + red.b2;
  A2 = softmax(Z2);
  if A2(2) > A2(1)
    TN = TN + 1;
  else
    FP = FP + 1;
  end
end

for i = 1:size(Norm_mal_test, 1)
  Z1 = red.W1 * Norm_mal_test(i,:)' + red.b1;
  A1 = tanh(Z1);
  Z2 = red.W2 * A1 + red.b2;
  A2 = softmax(Z2);
  if A2(1) > A2(2)
    TP = TP + 1;
  else
    FN = FN + 1;
  end
end

confMat_test = [TN, FP; FN, TP];

precision = TP / (TP + FP);
exactitud = (TP + TN) / (TP + TN + FP + FN);
recall = TP / (TP + FN);
F1 = 2 * (precision * recall) / (precision + recall);

% ==================== MOSTRAR Y GUARDAR RESULTADOS ====================
fprintf('\n--- PRUEBA ---\n');
fprintf('Matriz de confusión (prueba):\n');
disp(confMat_test);
fprintf('Precisión: %.2f%%\n', precision * 100);
fprintf('Exactitud: %.2f%%\n', exactitud * 100);
fprintf('Recall: %.2f%%\n', recall * 100);
fprintf('F1 Score: %.2f%%\n', F1 * 100);

% Guardar en archivo
fid = fopen('resultados_test8.txt', 'w');
fprintf(fid, '--- MATRIZ DE CONFUSIÓN (PRUEBA) ---\n');
fprintf(fid, '           Predicho 0    Predicho 1\n');
fprintf(fid, 'Real 0     %10d %12d\n', TN, FP);
fprintf(fid, 'Real 1     %10d %12d\n\n', FN, TP);
fprintf(fid, '--- MÉTRICAS DE PRUEBA ---\n');
fprintf(fid, 'Precisión: %.2f%%\n', precision * 100);
fprintf(fid, 'Exactitud: %.2f%%\n', exactitud * 100);
fprintf(fid, 'Recall:    %.2f%%\n', recall * 100);
fprintf(fid, 'F1 Score:  %.2f%%\n', F1 * 100);
fclose(fid);

