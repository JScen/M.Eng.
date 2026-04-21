%% LOOCV_LogisticRegression.m
% ロジスティック回帰モデルによる LOOCV 失踪予測
%
% MATLABネイティブな方法:
%   fitclinear + CVPartition でLOOCVを内部処理．
%   クラス不均衡はサンプル重みで対応．
%
% 必要 Toolbox: Statistics and Machine Learning Toolbox
% 実行前に同フォルダに: clean2.csv, prepare_data.m, print_metrics.m

clc; clear; close all;

%% 1. データ準備
[X, y, data_orig, ~] = prepare_data();
N = size(X, 1);

%% 2. サンプル重み（class_weight='balanced' と同等）
%    少数クラス（失踪）を多く重み付けすることでクラス不均衡を補正
n_pos = sum( y);
n_neg = sum(~y);
weights = zeros(N, 1);
weights( y) = N / (2 * n_pos);   % 失踪クラスの重み
weights(~y) = N / (2 * n_neg);   % 非失踪クラスの重み

%% 3. LOOCV の設定
cv = cvpartition(y, 'LeaveOut');

%% 4. モデル学習 + LOOCV
fprintf('ロジスティック回帰 LOOCV 実行中...\n');
tic;

mdl_cv = fitclinear(X, double(y), ...
    'CVPartition',    cv, ...
    'Learner',        'logistic', ...
    'Regularization', 'ridge', ...
    'Lambda',         1e-4, ...
    'Weights',        weights, ...
    'ClassNames',     [0, 1]);

%% 5. 予測結果の取得
[y_pred_raw, y_scores] = kfoldPredict(mdl_cv);
y_pred = logical(y_pred_raw);
y_prob = y_scores(:, 2);

fprintf('完了！総時間: %.1f 秒\n', toc);

%% 6. 評価指標の表示 & FN 保存
print_metrics('LogisticRegression', y, y_pred, y_prob, data_orig);
