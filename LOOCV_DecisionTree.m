%% LOOCV_DecisionTree.m
% 決定木モデルによる LOOCV 失踪予測
%
% MATLABネイティブな方法:
%   cvpartition + fitctree の 'CVPartition' オプションを使用．
%   手動ループ不要で，内部最適化が効く．
%
% 必要 Toolbox: Statistics and Machine Learning Toolbox
% 実行前に同フォルダに: clean2.csv, prepare_data.m, print_metrics.m

clc; clear; close all;

%% 1. データ準備
[X, y, data_orig, ~] = prepare_data();
N = size(X, 1);

%% 2. クラス不均衡の対応
%    'Prior','uniform' でクラス比率を均等に扱う（Pythonのclass_weight='balanced'と同等）
prior_setting = 'uniform';

%% 3. LOOCV の設定
%    cvpartition で Leave-One-Out を定義
cv = cvpartition(y, 'LeaveOut');

%% 4. モデル学習 + LOOCV
%    fitctree に CVPartition を渡すだけで自動的にLOOCVを実行
fprintf('決定木 LOOCV 実行中...\n');
tic;

mdl_cv = fitctree(X, y, ...
    'CVPartition',   cv, ...
    'SplitCriterion','gdi', ...
    'Prior',         prior_setting);

%% 5. 予測結果の取得
%    kfoldPredict で全サンプルの予測ラベル・スコアを一括取得
[y_pred_raw, y_scores] = kfoldPredict(mdl_cv);
y_pred  = logical(y_pred_raw);
y_prob  = y_scores(:, 2);   % 失踪クラス（列2）のスコア

fprintf('完了！総時間: %.1f 秒\n', toc);

%% 6. 評価指標の表示 & FN 保存
print_metrics('DecisionTree', y, y_pred, y_prob, data_orig);
