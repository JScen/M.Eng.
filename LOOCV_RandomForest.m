%% LOOCV_RandomForest.m
% ランダムフォレストモデルによる LOOCV 失踪予測
%
% MATLABネイティブな方法:
%   fitcensemble + CVPartition でLOOCVを内部処理．
%   'Method','Bag' がランダムフォレストに相当．
%
% 必要 Toolbox: Statistics and Machine Learning Toolbox
% 実行前に同フォルダに: clean2.csv, prepare_data.m, print_metrics.m

clc; clear; close all;

%% 1. データ準備
[X, y, data_orig, ~] = prepare_data();

%% 2. LOOCV の設定
cv = cvpartition(y, 'LeaveOut');

%% 3. 決定木テンプレート（アンサンブルの各木の設定）
tree_template = templateTree( ...
    'Reproducible', true);

%% 4. モデル学習 + LOOCV
fprintf('ランダムフォレスト LOOCV 実行中...\n');
fprintf('（時間がかかります）\n');
tic;

mdl_cv = fitcensemble(X, y, ...
    'CVPartition',       cv, ...
    'Method',            'Bag', ...
    'NumLearningCycles', 300, ...
    'Learners',          tree_template, ...
    'Prior',             'uniform');

%% 5. 予測結果の取得
[y_pred_raw, y_scores] = kfoldPredict(mdl_cv);
y_pred = logical(y_pred_raw);
y_prob = y_scores(:, 2);

fprintf('完了！総時間: %.1f 秒\n', toc);

%% 6. 評価指標の表示 & FN 保存
print_metrics('RandomForest', y, y_pred, y_prob, data_orig);
