function [X, y, data_orig, feature_names] = prepare_data()
%PREPARE_DATA  clean2.csv を読み込み，MATLABネイティブな方法で前処理する．
%
%  使用するMATLAB機能:
%    - readtable          : CSV読み込み
%    - normalize()        : 標準化（Pythonのstandard scalerと同等）
%    - onehotencode()     : One-Hot Encoding
%    - fillmissing()      : 欠損値補完
%
%  返り値:
%    X             : [N x F] 前処理済み特徴量行列（double）
%    y             : [N x 1] 目的変数（logical, 1=失踪）
%    data_orig     : 元の table（FN 保存用）
%    feature_names : 特徴量名のセル配列

%% 1. 読み込み
data_orig = readtable('clean2.csv', ...
    'Encoding',           'UTF-8', ...
    'VariableNamingRule', 'preserve');

%% 2. 目的変数
y = logical(data_orig{:, '失踪の有無'});

%% 3. 説明変数（目的変数列を除外）
X_raw      = removevars(data_orig, '失踪の有無');
feat_names = X_raw.Properties.VariableNames;

%% 4. 数値列・カテゴリ列を自動判別
num_mask = varfun(@isnumeric, X_raw, 'OutputFormat', 'uniform');
num_cols = feat_names( num_mask);
cat_cols = feat_names(~num_mask);

fprintf('数値列: %s\n',   strjoin(num_cols, ', '));
fprintf('カテゴリ列数: %d\n', length(cat_cols));

%% 5. 数値列：欠損を中央値で補完 → 標準化
%    fillmissing + normalize でPythonのPipelineと同等
X_num     = zeros(height(X_raw), length(num_cols));
num_names = cell(1, length(num_cols));
for i = 1:length(num_cols)
    col = X_raw{:, num_cols{i}};
    col = fillmissing(col, 'constant', median(col, 'omitnan'));
    X_num(:, i) = normalize(col, 'zscore');   % (x-mean)/std
    num_names{i} = [num_cols{i}, '_scaled'];
end

%% 6. カテゴリ列：欠損を 'N/A' で補完 → One-Hot Encoding
%    onehotencode() はMATLAB R2020b以降で使用可能
X_ohe     = [];
ohe_names = {};
for i = 1:length(cat_cols)
    col = string(X_raw{:, cat_cols{i}});
    col(col == "" | col == "NaN" | ismissing(col)) = "N/A";
    col = categorical(col);

    % onehotencode: 各カテゴリをダミー変数に変換
    ohe = onehotencode(col, 2);          % [N x nCats] logical行列
    ohe = double(ohe(:, 2:end));         % 最初の列をドロップ（多重共線性回避）

    cats = categories(col);
    for k = 2:length(cats)
        ohe_names{end+1} = sprintf('%s_%s', cat_cols{i}, cats{k}); %#ok<AGROW>
    end
    X_ohe = [X_ohe, ohe]; %#ok<AGROW>
end

%% 7. 結合
X = [X_num, X_ohe];
feature_names = [num_names, ohe_names];

fprintf('=== データ読み込み完了 ===\n');
fprintf('サンプル数  : %d\n', size(X,1));
fprintf('特徴量数    : %d\n', size(X,2));
fprintf('失踪者数    : %d\n', sum(y));
fprintf('非失踪者数  : %d\n', sum(~y));
fprintf('========================\n\n');
end
