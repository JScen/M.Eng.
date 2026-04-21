%% LOOCV_SVM.m
% SVM（線形カーネル）モデルによる LOOCV 失踪予測
%
% MATLABネイティブな方法:
%   fitcsvm + CVPartition でLOOCVを内部処理．
%   fitSVMPosterior で確率スコアを取得（PythonのSVC(probability=True)と同等）．
%
% 必要 Toolbox: Statistics and Machine Learning Toolbox
% 実行前に同フォルダに: clean2.csv, prepare_data.m, print_metrics.m

clc; clear; close all;

%% 1. データ準備
[X, y, data_orig, ~] = prepare_data();

%% 2. LOOCV の設定
cv = cvpartition(y, 'LeaveOut');

%% 3. モデル学習 + LOOCV
fprintf('SVM LOOCV 実行中...\n');
fprintf('（線形カーネル，時間がかかります）\n');
tic;

mdl_cv = fitcsvm(X, y, ...
    'CVPartition',    cv, ...
    'KernelFunction', 'linear', ...
    'Standardize',    false, ...    % prepare_data で標準化済み
    'Prior',          'uniform');

%% 4. Plattスケーリングで確率スコアを取得
%    fitSVMPosterior（旧: fitPosterior）でSVMの決定値を確率に変換
mdl_cv = fitSVMPosterior(mdl_cv);

%% 5. 予測結果の取得
[y_pred_raw, y_scores] = kfoldPredict(mdl_cv);
y_pred = logical(y_pred_raw);
y_prob = y_scores(:, 2);

fprintf('完了！総時間: %.1f 秒\n', toc);

%% 6. 評価指標の表示 & FN 保存
print_metrics('SVM', y, y_pred, y_prob, data_orig);
