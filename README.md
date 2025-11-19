# `shunk031/pytest-ml-tdd-example`

## 概要

このリポジトリは、機械学習（ML）の実験コードを、**テスト駆動開発（TDD）の考え方と `pytest` を用いて改善するためのサンプルコード**です。目的は、「**オレオレ最強 main.py**」のスタイル（手続き的で密結合、依存が隠蔽されている）から脱却し、筋の良い Python コードを書くことです。

## プロジェクトの目的と特徴

多くの実験コードが陥る「再現性の罠」「拡張性の罠」「テストの罠」を `pytest` で解決します。

1.  **再現性の確保と設計の明示**:
    *   **テストを書くことは実験設計を明示すること**であり、関数の入出力が明確になります。
    *   `pytest`の**fixture機構**が依存性注入（DI）機能を提供し、実験条件（`seed`、`dataset`、`model`など）を分離・明示的に注入できます。
    *   `fix_seed` fixtureにより、乱数シード（例: 19950815）の固定化が容易になります。

2.  **軽量な実験実行くん**:
    *   `pytest` を**実験のタスクランナー**として使用します。
    *   実行パラメータは fixture として明示され、複雑なコマンドを覚える必要がなく、「**`$ pytest`**」の**1コマンドだけ**で全ての実験を再現可能です。

3.  **実験探索の自動化**:
    *   `@pytest.mark.parametrize` デコレータを利用することで、学習率（`lr`）やバッチサイズ（`batch_size`）など、**設定の組み合わせを自動で総当たり**し、実験探索を容易に行えます。

## 実行方法

実験の実行は以下のコマンドで行えます:

```bash
$ uv run pytest
```

これにより、実験環境が自動で構築・注入され、テスト（スモークテスト）が研究の最小構成単位となります。

## uv 初心者向け：本レポジトリをゼロから構築したときの手順

```shell
uv init --lib -p 3.11

# Initialized project `pytest-ml-tdd-example`
```

```shell
uv add --group dev 'ruff>=0.1.5' 'pytest>=6.0.0' 'mypy>=1.0.0'

# Using CPython 3.11.10
# Creating virtual environment at: .venv
# Resolved 12 packages in 48ms
# Prepared 3 packages in 707ms
# Installed 10 packages in 46ms
#  + iniconfig==2.3.0
#  + mypy==1.18.2
#  + mypy-extensions==1.1.0
#  + packaging==25.0
#  + pathspec==0.12.1
#  + pluggy==1.6.0
#  + pygments==2.19.2
#  + pytest==9.0.1
#  + ruff==0.14.5
#  + typing-extensions==4.15.0
```

```shell
uv add torch torchvision

# Resolved 40 packages in 102ms
# Prepared 6 packages in 12.80s
# Installed 27 packages in 163ms
#  + filelock==3.20.0
#  + fsspec==2025.10.0
#  + jinja2==3.1.6
#  + markupsafe==3.0.3
#  + mpmath==1.3.0
#  + networkx==3.5
#  + numpy==2.3.4
#  + nvidia-cublas-cu12==12.8.4.1
#  + nvidia-cuda-cupti-cu12==12.8.90
#  + nvidia-cuda-nvrtc-cu12==12.8.93
#  + nvidia-cuda-runtime-cu12==12.8.90
#  + nvidia-cudnn-cu12==9.10.2.21
#  + nvidia-cufft-cu12==11.3.3.83
#  + nvidia-cufile-cu12==1.13.1.3
#  + nvidia-curand-cu12==10.3.9.90
#  + nvidia-cusolver-cu12==11.7.3.90
#  + nvidia-cusparse-cu12==12.5.8.93
#  + nvidia-cusparselt-cu12==0.7.1
#  + nvidia-nccl-cu12==2.27.5
#  + nvidia-nvjitlink-cu12==12.8.93
#  + nvidia-nvshmem-cu12==3.3.20
#  + nvidia-nvtx-cu12==12.8.90
#  + pillow==12.0.0
#  + sympy==1.14.0
#  + torch==2.9.1
#  + torchvision==0.24.1
#  + triton==3.5.1
```

## ソースコードの書き始め

- `tests` ディレクトリの作成

```shell
mkdir tests
```

- テストファイル `tests/main_test.py` の作成

```shell
touch tests/main_test.py
```

- テストコードの実行

```shell
CUDA_VISIBLE_DEVICES=0 uv run pytest -vsx tests
```
