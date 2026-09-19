# EMS Mesh Checker

`ems_mesh_checker` は、有限要素メッシュのトポロジーをチェックし、EMSolution のメッシュから代表的な Property 形状を再構築するための Python ベースの概念実証（PoC）ツールです。PyVista/VTK を用いて特徴エッジを抽出した後、独自のトポロジー処理と曲線再構築を行い、CAD やメッシャーで再利用できる入力データとして出力します。

## 特長

- 外表面および Property ID 間の境界面を抽出
- `--properties` により指定した Property のみを選択
- `--exclude-properties` により空気領域など指定した Property を除外
- `--merge-properties` により複数の Property を1つの領域として扱い、それら相互間の境界面を除外
- 1つの四角形と2つの三角形から構成される、同一 Property 内の六面体／四面体結合境界面をオプションで抑制
- 特徴エッジから代表的な直線および円弧を再構築
- 再構築した同一の曲線を DXF、Gmsh GEO、Femap Neutral の各形状ファイルとして出力
- 大規模な Femap Neutral ファイルに対して、節点／要素の読み込み進捗を表示

現在の PoC は一次要素を対象としています。高次要素への対応は、現時点のリリース目標には含まれていません。

## インストール

Python 3.10 以上が必要です。通常は、本ツールと依存するプロジェクトをそれぞれ Python パッケージとしてインストールします。2つのリポジトリを同じ親ディレクトリに配置する必要はありません。

PyPI からインストールできます。依存関係として `ems-file-format-converter` 0.6.0 以上が自動的にインストールされます。

```powershell
python -m pip install ems-mesh-checker
```

両方のプロジェクトを開発する場合は、それぞれのリポジトリを clone してインストールします。

```powershell
git clone https://github.com/EMSolution-SSIL/ems_file_format_converter.git
git clone https://github.com/EMSolution-SSIL/ems_mesh_checker.git

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .\ems_file_format_converter
.\.venv\Scripts\python.exe -m pip install -e ".\ems_mesh_checker[dev]"
```

editable install では、各リポジトリは任意のディレクトリに配置できます。隣接して配置する構成は、あくまで管理しやすくするための例です。

## コマンドラインでの使用方法

コマンドは任意の入力メッシュを受け取ります。出力時には、同じファイル名の stem を持つ代表形状 `.dxf`、`.geo`、Femap 形状 `.neu` の3ファイルを常に生成します。

```powershell
ems-mesh-checker INPUT_MESH [OUTPUT.dxf]
```

公開サンプル IronCoil の例:

```powershell
ems-mesh-checker .\data\3D\IronCoil\post_geom.atl .\generated\IronCoil.dxf `
  --circle-elements-per-turn 48
```

公開サンプル QuadTri の例。異なる Property 間の境界は保持しつつ、同一 Property 内の hex/tet 結合面を抑制します。

```powershell
ems-mesh-checker .\data\3D\QuadTri\post_geom.neu .\generated\QuadTri.dxf `
  --suppress-quad-tri-interfaces
```

Property 2 と 5 のみを出力する例:

```powershell
ems-mesh-checker model.neu geometry.dxf --properties 2,5
```

空気領域の Property 1 を除外し、Property 4 と 5 を1つの抽出領域として結合する例:

```powershell
ems-mesh-checker model.neu geometry.dxf `
  --exclude-properties 1 `
  --merge-properties 4,5
```

独立した複数の Property グループを結合する場合は、`--merge-properties` を繰り返し指定します。許容誤差、プレビュー、Gmsh 検証などのオプションについては `ems-mesh-checker --help` を参照してください。

PyVista は、まれに小さな三角形や多角形状の特徴エッジノイズを返すことがあります。以下のオプションによるクリーンアップでは、境界を構成するエッジ数が10以下の閉じたエッジループ、および孤立した開放表面コンポーネントを除去します。必要に応じてモデルに合わせて `--max-loop-edges` を調整してください。

```powershell
ems-mesh-checker model.neu geometry.dxf `
  --remove-small-loops `
  --max-loop-edges 10
```

小さなループや開放表面が実際の形状特徴である場合もあるため、このクリーンアップはデフォルトでは無効です。閉じた表面コンポーネント、複数表面から構成されるコンポーネント、および2次元の閉領域は保持されます。CLI の診断出力には、除去されたエッジループ数と開放表面コンポーネント数の両方が表示されます。

## 出力形式

- DXF: CAD やメッシャーとの連携を目的とした、点／直線／円弧ベースの代表形状
- Gmsh GEO: 再構築可能な範囲で、点、直線、円、平面／ruled surface、surface loop、volume を出力
- Femap Neutral: 再構築した点、直線、円弧の形状を出力

特徴エッジの抽出はヒューリスティックな処理です。複雑なメッシュやノイズを含むメッシュでは、開いたシェルや未対応の平面領域が残る場合があります。本番のメッシュ生成に使用する前に、CLI の診断結果と生成された形状を確認してください。

## Python API

低レベル API は `ems_mesh_checker` から利用できます。

```python
from ems_file_format_converter import read_mesh
from ems_mesh_checker import BoundaryExtractor, FeatureEdgeConfig
from ems_mesh_checker import extract_property_group_feature_edges

mesh = read_mesh("model.neu", progress=True)
boundaries = BoundaryExtractor(mesh).extract()
features = extract_property_group_feature_edges(
    boundaries,
    (2, 5),
    config=FeatureEdgeConfig(
        feature_angle_degrees=30.0,
        remove_small_loops=True,
        max_loop_edges=10,
    ),
)
```

## 公開サンプルデータ

レビュー済みの公開サンプルとして、以下のモデルのみを含めています。

- `data/3D/IronCoil/post_geom.atl`
- `data/3D/QuadTri/post_geom.neu`

その他の開発用モデルおよび顧客モデルは意図的に含めていません。`.gitignore` では、`data/` 以下に追加されるその他のファイルをデフォルトで非公開扱いとしています。

## テスト

```powershell
python -m pytest -q
```

## ライセンス

MIT License. Copyright (c) 2026 Hiroyuki Kaimori. 詳細は `LICENSE` を参照してください。

## 英語版README

英語版は [`README.md`](README.md) を参照してください。
