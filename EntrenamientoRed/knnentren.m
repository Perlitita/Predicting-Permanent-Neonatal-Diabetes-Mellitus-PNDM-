clc; clear all; close all;
pkg load io;
pkg load statistics;

% === Cargar y preparar datos de entrenamiento ===
train_data = dlmread('pndm_train.csv', ',', 1, 0);  % omitir encabezado
X_train = train_data(:, 1:7);
Y_train = zeros(size(X_train, 1), 1);
Y_train(106:210) = 1;

% === Cargar y preparar datos de prueba ===
test_data = dlmread('pndm_test.csv', ',', 1, 0);  % omitir encabezado
X_test = test_data(:, 1:7);
Y_test = zeros(size(X_test, 1), 1);
Y_test(46:90) = 1;

% === KNN clasificación ===
k = 8;
y_pred = zeros(size(Y_test));

for i = 1:size(X_test, 1)
    distances = sum((X_train - X_test(i, :)).^2, 2);
    [~, idx] = sort(distances);
    vecinos = Y_train(idx(1:k));
    y_pred(i) = mode(vecinos);
end

% === Evaluación ===
TP = sum((y_pred == 1) & (Y_test == 1));
TN = sum((y_pred == 0) & (Y_test == 0));
FP = sum((y_pred == 1) & (Y_test == 0));
FN = sum((y_pred == 0) & (Y_test == 1));
accuracy = (TP + TN) / length(Y_test) * 100;
F_score = 2 * TP / (2 * TP + FP + FN);

% === Resultados ===
disp('--- Evaluación del modelo KNN ---');
fprintf('Precisión: %.2f%%\n', accuracy);
fprintf('F-score: %.4f\n', F_score);
disp('Matriz de Confusión:');
disp([TP, FP; FN, TN]);

