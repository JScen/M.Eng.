function print_metrics(model_name, y, y_pred, y_prob, data_orig)
%PRINT_METRICS  LOOCV結果の評価指標を表示し，FNデータをCSVに保存する．
%
%  使用するMATLAB機能:
%    - confusionmat() : 混同行列
%    - perfcurve()    : AUC計算

%% 混同行列
%    confusionmat(actual, predicted) → [TN FP; FN TP] の順
cm = confusionmat(double(y), double(y_pred));
TN = cm(1,1); FP = cm(1,2);
FN = cm(2,1); TP = cm(2,2);

%% 評価指標
accuracy    = (TP+TN) / sum(cm(:));
precision   = TP / max(TP+FP, 1);
recall      = TP / max(TP+FN, 1);      % Sensitivity
f1          = 2*precision*recall / max(precision+recall, 1e-9);
specificity = TN / max(TN+FP, 1);

% 非失踪クラス（0）の指標
prec0 = TN / max(TN+FN, 1);
rec0  = TN / max(TN+FP, 1);
f1_0  = 2*prec0*rec0 / max(prec0+rec0, 1e-9);

% AUC（perfcurve はMATLAB標準関数）
[~, ~, ~, auc] = perfcurve(double(y), y_prob, 1);

%% 表示
fprintf('\n========================================\n');
fprintf('【%s：LOOCV 結果】\n', model_name);
fprintf('========================================\n');

fprintf('\n--- 混同行列 ---\n');
fprintf('              予測:非失踪  予測:失踪\n');
fprintf('  実際:非失踪    %5d       %5d\n', TN, FP);
fprintf('  実際:失踪      %5d       %5d\n', FN, TP);

fprintf('\n--- クラス別レポート ---\n');
fprintf('%-12s  %9s  %6s  %8s  %7s\n', 'クラス','Precision','Recall','F1-score','Support');
fprintf('%-12s  %9.3f  %6.3f  %8.3f  %7d\n', '非失踪(0)', prec0, rec0, f1_0, sum(~y));
fprintf('%-12s  %9.3f  %6.3f  %8.3f  %7d\n', '失踪(1)',   precision, recall, f1, sum(y));

fprintf('\nAccuracy             : %.3f\n', accuracy);
fprintf('Sensitivity（感度）  : %.3f\n', recall);
fprintf('Specificity（特異度）: %.3f\n', specificity);
fprintf('AUC                  : %.3f\n', auc);

%% FNサンプルの保存
fn_mask  = (y == 1) & (y_pred == 0);
fn_data  = data_orig(fn_mask, :);
out_file = sprintf('LOOCV_FN_%s_2.csv', model_name);
writetable(fn_data, out_file, 'Encoding', 'UTF-8');
fprintf('\nFNデータ保存: %s（%d件）\n', out_file, sum(fn_mask));
end
