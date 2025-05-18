clc; clear all; close all;
pkg load io;
pkg load statistics;

% === Cargar base de entrenamiento ===
data_train = dlmread('pndm_train.csv');
data_train = data_train(2:211, 1:8);

% === Normalización ===
cols_to_normalize = [1, 2, 5, 7];
max_vals = struct();
min_vals = struct();
ymax = 1; ymin = 0;

for i = 1:length(cols_to_normalize)
    col_idx = cols_to_normalize(i);
    col = data_train(:, col_idx);
    valMax = max(col);
    valMin = min(col);
    max_vals.(sprintf('col%d', col_idx)) = valMax;
    min_vals.(sprintf('col%d', col_idx)) = valMin;
    data_train(:, col_idx) = ((ymax - ymin) * (col - valMin)) ./ (valMax - valMin) + ymin;
end

% === Separar características y etiquetas ===
X_train = data_train(:, 1:7);
Y_train = zeros(size(X_train, 1), 1);
Y_train(106:210) = 1;

% === Entrenamiento Naive Bayes ===
P_class0 = sum(Y_train == 0) / length(Y_train);
P_class1 = sum(Y_train == 1) / length(Y_train);
means_0 = mean(X_train(Y_train == 0, :), 1);
means_1 = mean(X_train(Y_train == 1, :), 1);
vars_0 = var(X_train(Y_train == 0, :), 0, 1) + 1e-3;
vars_1 = var(X_train(Y_train == 1, :), 0, 1) + 1e-3;

modelo = struct('P_class0', P_class0, 'P_class1', P_class1, ...
                'means_0', means_0, 'means_1', means_1, ...
                'vars_0', vars_0, 'vars_1', vars_1, ...
                'min_vals', min_vals, 'max_vals', max_vals);

% === Cargar base de prueba ===
data_test = dlmread('pndm_test.csv');
data_test = data_test(2:91, 1:8);

% === Normalización con clipping ===
for i = 1:length(cols_to_normalize)
    col_idx = cols_to_normalize(i);
    col = data_test(:, col_idx);
    valMin = modelo.min_vals.(sprintf('col%d', col_idx));
    valMax = modelo.max_vals.(sprintf('col%d', col_idx));
    norm_col = ((ymax - ymin) * (col - valMin)) ./ (valMax - valMin) + ymin;
    norm_col = max(0, min(1, norm_col));  % Clipping
    data_test(:, col_idx) = norm_col;
end

% === Características y etiquetas reales ===
X_test = data_test(:, 1:7);
Y_test = zeros(size(X_test, 1), 1);
Y_test(46:90) = 1;

% === Clasificación ===
predicciones = zeros(size(Y_test));
eps_val = 1e-6;
umbral = 0.5;

for i = 1:length(Y_test)
    log_prob_0 = log(modelo.P_class0 + eps_val) + ...
        sum(log(normpdf(X_test(i, :), modelo.means_0, sqrt(modelo.vars_0) + eps_val)));
    log_prob_1 = log(modelo.P_class1 + eps_val) + ...
        sum(log(normpdf(X_test(i, :), modelo.means_1, sqrt(modelo.vars_1) + eps_val)));
    total_prob = exp(log_prob_0) + exp(log_prob_1) + eps_val;
    posterior_1 = exp(log_prob_1) / total_prob;
    predicciones(i) = double(posterior_1 > umbral);
end

% === Evaluación ===
accuracy = sum(predicciones == Y_test) / length(Y_test) * 100;
TP = sum((predicciones == 1) & (Y_test == 1));
TN = sum((predicciones == 0) & (Y_test == 0));
FP = sum((predicciones == 1) & (Y_test == 0));
FN = sum((predicciones == 0) & (Y_test == 1));
F_score = 2 * TP / (2 * TP + FP + FN);

% === Mostrar resultados ===
disp(['Precisión del modelo: ', num2str(accuracy), '%']);
disp(['F-score: ', num2str(F_score)]);
disp('Matriz de confusión:');
disp([TP, FP; FN, TN]);

% === Guardar resultados ===
fid = fopen('resultadosPNDB.txt', 'w');
fprintf(fid, 'Ingenuo Bayesiano - PNDB\n\n');
fprintf(fid, 'Precisión: %.2f%%\n', accuracy);
fprintf(fid, 'F-score: %.4f\n', F_score);
fprintf(fid, '\nMatriz de Confusión:\n');
fprintf(fid, '           Pred 0   Pred 1\n');
fprintf(fid, 'Real 0     %6d   %6d\n', TN, FP);
fprintf(fid, 'Real 1     %6d   %6d\n', FN, TP);
fclose(fid);

