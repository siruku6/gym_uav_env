# Docker での環境構築

流れを以下に記載する

## 1. Python 環境の準備

- docker コンテナの build

    ```bash
    $ cd environments/uav_env
    $ cp .env.example .env

    $ docker compose build
    ```

これにより、 docker で jupyter と tensorboard を起動できるようになるので、実施しておくことを推奨


## 2. 必要ファイルの準備

- `data` ディレクトリ直下に `drone.urdf` を配置する。


## 3. 学習〜評価

- docker 環境の準備

    ```bash
    $ docker compose up
    $ docker compose exec uav_env bash
    ```

- 実行
    `localhost:5001` にアクセスして、 notebokks ディレクトリ下にある ipynb ファイルを実行することで、学習、推論が行える。
